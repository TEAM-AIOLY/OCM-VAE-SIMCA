# ============================================================
# IMPORTS 
# ============================================================
import os
import random
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim as optim
import torch.utils.data as data_utils
import itertools
from pathlib import Path

from vae_model import ConvVAE1D, beta_vae_bce_loss, compute_q_h_f
from utils.data_utils import object_aware_splits

# ============================================================
# SEED + DEVICE 
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = os.getcwd()

# ============================================================
# HELPERS 
# ============================================================
def save_model(model, path, name):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / name)

def save_metrics(obj, path, name):
    path.mkdir(parents=True, exist_ok=True)
    with open(path / name, "w") as f:
        json.dump(obj, f, indent=2)

def save_plot(fig, path, name, fmt="pdf"):
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f"{name}.{fmt}")
    plt.close(fig)

# ============================================================
# DATA LOADING 
# ============================================================
data_root = "C:\\00_aioly\\GitHub\\HyperNuts\\data\\Nuts data\\HSI Data"
data_folder = "SWIR camera (842–2532 nm)"
data_file = "nut_objects.h5"
file_path = os.path.join(data_root, data_folder, data_file)

if not os.path.exists(file_path):
    raise FileNotFoundError(file_path)

data = {}
with h5py.File(file_path, 'r') as h5f:
    for nut_type in sorted(h5f.keys()):
        data[nut_type] = []
        for img_key in sorted(h5f[nut_type].keys()):
            for obj_key in sorted(h5f[nut_type][img_key].keys()):
                grp = h5f[nut_type][img_key][obj_key]
                spec = grp['spectra'][()]
                data[nut_type].append({
                    'spectral_data': spec,
                    'obj_idx': int(grp.attrs.get('obj_idx', -1)),
                    'img_idx': int(grp.attrs.get('img_idx', -1))
                })

nut_types = list(data.keys())
n_wavelengths = data[nut_types[0]][0]['spectral_data'].shape[1]

# ============================================================
# SPLITS 
# ============================================================
TARGET_NUT = "peanut"

_, _, _, X_cal, X_val, X_test_in, X_test_out = object_aware_splits(
    data, nut_types, TARGET_NUT, n_wavelengths
)

# labels for test
y_test_in = np.zeros(len(X_test_in), dtype=int)
y_test_out = np.ones(len(X_test_out), dtype=int)

X_test = np.vstack([X_test_in, X_test_out])
y_test = np.concatenate([y_test_in, y_test_out])

# ============================================================
# OUTPUT
# ============================================================
output_dir = Path(data_root) / data_folder / "vae_simca_analysis" / TARGET_NUT
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# PARAMS 
# ============================================================
base_params = {
    "batch_size": 512,
    "n_epochs": 2000,
    "weight_decay": 0.003 / 2,
    "beta": 1.0,
}

param_grid = {
    "latent_dim": [32],
    "hidden_fc": [32, 64],
    "learning_rate": [1e-4],
    "conv_blocks": [1, 3],
    "n_filters": [1, 3],
    "kernel_size": [5, 7],
    "dropout": [0.1],
}

keys, values = zip(*param_grid.items())
param_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

all_metrics = []

# ============================================================
# Benchmark
# ============================================================
for run_idx, params in enumerate(param_list):

    run_name = f"run_{run_idx:03d}"
    base_path = output_dir / run_name
    base_path.mkdir(parents=True, exist_ok=True)

    mean = np.mean(X_cal, axis=0)
    std = np.std(X_cal, axis=0) + 1e-8

    vae = ConvVAE1D(
        input_length=n_wavelengths,
        latent_dim=params["latent_dim"],
        mean=mean,
        std=std,
        conv_blocks=params["conv_blocks"],
        n_filters=params["n_filters"],
        kernel_size=params["kernel_size"],
        hidden_fc=params["hidden_fc"],
        dropout=params["dropout"],
        activation="elu",
        beta=base_params["beta"]
    ).to(device)

    optimizer = optim.Adam(
        vae.parameters(),
        lr=params["learning_rate"],
        weight_decay=base_params["weight_decay"]
    )

    cal_loader = data_utils.DataLoader(
        data_utils.TensorDataset(torch.tensor(X_cal, dtype=torch.float32)),
        batch_size=base_params["batch_size"], shuffle=True
    )

    val_loader = data_utils.DataLoader(
        data_utils.TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
        batch_size=base_params["batch_size"], shuffle=False
    )

    train_losses, val_losses = [], []
    best_val_loss = np.inf
    best_epoch = -1

    # ========================================================
    # TRAINING
    # ========================================================
    for epoch in range(base_params["n_epochs"]):
        vae.train()
        tot_loss = 0.0

        for xb, in cal_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            xb_rec, mu, logvar = vae(xb)
            loss, _, _ = beta_vae_bce_loss(xb, xb_rec, mu, logvar)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * xb.size(0)

        train_loss = tot_loss / len(cal_loader.dataset)
        train_losses.append(train_loss)

        vae.eval()
        tot_loss = 0.0
        with torch.no_grad():
            for xb, in val_loader:
                xb = xb.to(device)
                xb_rec, mu, logvar = vae(xb)
                loss, _, _ = beta_vae_bce_loss(xb, xb_rec, mu, logvar)
                tot_loss += loss.item() * xb.size(0)

        val_loss = tot_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            # ---------- thresholds
            q_list, h_list, f_list = [], [], []
            with torch.no_grad():
                for xb, in cal_loader:
                    x = xb.to(device)
                    x_rec, z, _ = vae(x)
                    q, h, f, q_c, h_c, f_c = compute_q_h_f(x, x_rec, z)
                    q_list.append(q.cpu().numpy())
                    h_list.append(h.cpu().numpy())
                    f_list.append(f.cpu().numpy())

            vae.threshold_q.copy_(torch.tensor(q_c))
            vae.threshold_h.copy_(torch.tensor(h_c))
            vae.threshold_f.copy_(torch.tensor(f_c))

            save_model(vae, base_path, "VAE_class0_best.pth")

    save_metrics(
        {"train_losses": train_losses, "val_losses": val_losses},
        base_path,
        "losses.json"
    )

    # ========================================================
    # TEST (UNCHANGED LOGIC)
    # ========================================================
    test_loader = data_utils.DataLoader(
        data_utils.TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test)
        ),
        batch_size=base_params["batch_size"], shuffle=False
    )

    vae.load_state_dict(torch.load(base_path / "VAE_class0_best.pth"))
    vae.eval()

    f_values, labels_true = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            x = xb.to(device)
            x_rec, z, _ = vae(x)
            _, _, f, _, _, _ = compute_q_h_f(x, x_rec, z)
            f_values.append(f.cpu().numpy())
            labels_true.append(yb.numpy())

    f_all = np.concatenate(f_values)
    labels_true = np.concatenate(labels_true)

    pred_class0 = f_all <= vae.threshold_f.item()
    pred_labels = np.where(pred_class0, 0, 1)

    conf_mat = np.zeros((2, 2), dtype=int)
    for i in range(len(labels_true)):
        conf_mat[pred_labels[i], labels_true[i]] += 1

    # ========================================================
    # FIGURE
    # ========================================================
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
                xticklabels=["in", "out"],
                yticklabels=["conform", "unconform"],
                ax=ax)
    ax.set_xlabel("True class")
    ax.set_ylabel("Predicted")
    save_plot(fig, base_path, "confusion_matrix_anomaly")

    # ========================================================
    # METRICS (UNCHANGED FORMULAS)
    # ========================================================
    TP = conf_mat[0, 0]
    FN = conf_mat[1, 0]
    FP = conf_mat[0, 1]
    TN = conf_mat[1, 1]

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-12)
    precision = TP / (TP + FP + 1e-12)
    recall = TP / (TP + FN + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    metrics = {
        "run": run_name,
        "latent_dim": params["latent_dim"],
        "hidden_fc": params["hidden_fc"],
        "conv_blocks": params["conv_blocks"],
        "n_filters": params["n_filters"],
        "kernel_size": params["kernel_size"],
        "F1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "best_epoch": best_epoch
    }

    all_metrics.append(metrics)

    with open(base_path / "metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

# ============================================================
# GLOBAL SUMMARY
# ============================================================
with open(output_dir / "all_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

print("SWEEP FINISHED")

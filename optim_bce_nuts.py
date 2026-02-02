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
import optuna
from pathlib import Path
from sklearn.metrics import roc_auc_score

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

y_test_in = np.zeros(len(X_test_in), dtype=int)
y_test_out = np.ones(len(X_test_out), dtype=int)

X_test = np.vstack([X_test_in, X_test_out])
y_test = np.concatenate([y_test_in, y_test_out])

# ============================================================
# OUTPUT
# ============================================================
output_dir = Path(data_root) / data_folder / "vae_simca_optuna" / TARGET_NUT
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# FIXED PARAMS 
# ============================================================
BASE_PARAMS = {
    "batch_size": 512,
    "n_epochs": 400,
    "beta": 1.0,
     "weight_decay": 0.003 / 2,
}

# ============================================================
# OPTUNA OBJECTIVE
# ============================================================
def objective(trial):

    params = {
        "latent_dim": trial.suggest_categorical("latent_dim", [20, 30]),
        "hidden_fc": trial.suggest_categorical("hidden_fc", [ 64, 128, 256]),
        "conv_blocks": trial.suggest_int("conv_blocks", 1, 5),
        "n_filters": trial.suggest_int("n_filters", 1, 5),
        "kernel_size": trial.suggest_categorical("kernel_size", [3,5, 7, 9]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 3e-4),
    }

    run_name = f"trial_{trial.number:03d}"
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
        beta=BASE_PARAMS["beta"]
    ).to(device)

    optimizer = optim.Adam(
        vae.parameters(),
        lr=params["learning_rate"],
        weight_decay=BASE_PARAMS["weight_decay"]
    )

    cal_loader = data_utils.DataLoader(
        data_utils.TensorDataset(torch.tensor(X_cal, dtype=torch.float32)),
        batch_size=BASE_PARAMS["batch_size"], shuffle=True
    )

    val_loader = data_utils.DataLoader(
        data_utils.TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
        batch_size=BASE_PARAMS["batch_size"], shuffle=False
    )

    best_val_loss = np.inf
    best_epoch = -1

    # ========================================================
    # TRAINING
    # ========================================================
    for epoch in range(BASE_PARAMS["n_epochs"]):

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

        vae.eval()
        tot_loss = 0.0
        with torch.no_grad():
            for xb, in val_loader:
                xb = xb.to(device)
                xb_rec, mu, logvar = vae(xb)
                loss, _, _ = beta_vae_bce_loss(xb, xb_rec, mu, logvar)
                tot_loss += loss.item() * xb.size(0)

        val_loss = tot_loss / len(val_loader.dataset)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            with torch.no_grad():
                for xb, in cal_loader:
                    x = xb.to(device)
                    x_rec, z, _ = vae(x)
                    _, _, _, _, _, f_c = compute_q_h_f(x, x_rec, z)

            vae.threshold_f.copy_(torch.tensor(f_c))
            save_model(vae, base_path, "VAE_class0_best.pth")

    # ========================================================
    # TEST
    # ========================================================
    vae.load_state_dict(torch.load(base_path / "VAE_class0_best.pth"))
    vae.eval()

    test_loader = data_utils.DataLoader(
        data_utils.TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test)
        ),
        batch_size=BASE_PARAMS["batch_size"], shuffle=False
    )

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

    auc = roc_auc_score(labels_true, f_all)

    # ========================================================
    # CONFUSION MATRIX + FIGURE (UNCHANGED)
    # ========================================================
    pred_class0 = f_all <= vae.threshold_f.item()
    pred_labels = np.where(pred_class0, 0, 1)

    conf_mat = np.zeros((2, 2), dtype=int)
    for i in range(len(labels_true)):
        conf_mat[pred_labels[i], labels_true[i]] += 1

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

    # store secondary metrics
    trial.set_user_attr("F1", f1)
    trial.set_user_attr("accuracy", accuracy)
    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("auc", auc)

    return accuracy

# ============================================================
# RUN STUDY
# ============================================================
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
)

study.optimize(objective, n_trials=50)

# ============================================================
# SAVE GLOBAL RESULTS
# ============================================================
with open(output_dir / "study_results.json", "w") as f:
    json.dump(
        [{"trial": t.number, **t.params, **t.user_attrs} for t in study.trials if t.value is not None],
        f,
        indent=2
    )

print("OPTUNA SWEEP FINISHED")
print("Best trial:", study.best_trial.number)
print("Best params:", study.best_params)
print("Best AUC:", study.best_value)

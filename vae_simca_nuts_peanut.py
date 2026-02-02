import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from vae_model import ConvVAE1D
from copy import deepcopy
import itertools

# ===========================
# Loss functions
# ===========================

def beta_vae_cosine_loss(x, x_recon, mu, logvar, beta=1.0, eps=1e-8):
    x_flat = x.view(x.size(0), -1)
    x_recon_flat = x_recon.view(x_recon.size(0), -1)
    x_norm = F.normalize(x_flat, p=2, dim=1)
    recon_norm = F.normalize(x_recon_flat, p=2, dim=1)
    cos_theta = torch.clamp(torch.sum(x_norm * recon_norm, dim=1), -1.0 + eps, 1.0 - eps)
    recon_loss = torch.mean(torch.sqrt(2.0 * (1.0 - cos_theta)))
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    return recon_loss + beta*kl, recon_loss.detach().cpu().item(), kl.detach().cpu().item()

# Import data loading from simca_nuts
data_root = "C:\\00_aioly\\GitHub\\HyperNuts\\data\\Nuts data\\HSI Data"
data_folder = "SWIR camera (842–2532 nm)"
data_file = "nut_objects.json"
file_path = os.path.join(data_root, data_folder, data_file)

with open(file_path, 'r') as f:
    data = json.load(f)

# Extract unique nut types
nut_types = list(data.keys())
nut_type_to_label = {nut_type: idx for idx, nut_type in enumerate(nut_types)}
print(f"Nut types found: {nut_types}")
print(f"Label mapping: {nut_type_to_label}\n")

# Number of wavelengths (assume consistent across objects)
example_nut = nut_types[0]
example_obj = data[example_nut][0]

n_wavelengths = np.array(example_obj['spectral_data']).shape[1]

# Target nut type
TARGET_NUT = 'peanut'
print(f"Target nut type: {TARGET_NUT}\n")

# ===========================
# Data preparation for peanut
# ===========================

print(f"Processing {TARGET_NUT}:")

#
splits = {}
for nut_type in nut_types:
    # Stack raw spectra for this nut
    all_spectra = [np.array(obj['spectral_data'], dtype=np.float32) for obj in data[nut_type]]
    X_nut = np.vstack(all_spectra)

    # Remove NaN/inf from raw data
    bad_mask_nut = np.isnan(X_nut).any(axis=1) | np.isinf(X_nut).any(axis=1)
    if np.any(bad_mask_nut):
        print(f"  WARNING: {nut_type}: Found {np.sum(bad_mask_nut)} NaN/inf samples. Removing them.")
        X_nut = X_nut[~bad_mask_nut]

    # Prepare a preprocessed copy for outlier detection only (SNV + Savitzky-Golay)
    X_proc = (X_nut - np.mean(X_nut, axis=1, keepdims=True)) / (np.std(X_nut, axis=1, keepdims=True) + 1e-8)
    try:
        X_proc = savgol_filter(X_proc, window_length=5, polyorder=2, deriv=1, axis=1)
    except Exception:
        pass

    # Outlier detection (PCA score-space) on X_proc
    X_clean = X_nut.copy()
    if X_proc.shape[0] > 3 and X_proc.shape[0] > 1:
        n_comp = min(10, X_proc.shape[1], max(1, X_proc.shape[0]-1))
        if X_proc.shape[0] > n_comp:
            pca_tmp = PCA(n_components=n_comp)
            T = pca_tmp.fit_transform(X_proc)
            mean_scores = np.mean(T, axis=0)
            cov_scores = np.cov(T, rowvar=False)
            cov_inv = np.linalg.pinv(cov_scores)
            mahal = np.array([np.sqrt((t - mean_scores) @ cov_inv @ (t - mean_scores).T) for t in T])
            out_thr = np.percentile(mahal, 95)
            mask = mahal <= out_thr
            n_removed = np.sum(~mask)
            if n_removed > 0:
                print(f"  {nut_type}: removed {n_removed} outliers (threshold {out_thr:.3f})")
            X_clean = X_nut[mask]
    else:
        print(f"  {nut_type}: too few samples for PCA-outlier removal, skipping outlier removal")

    X_cal_nut, X_temp_nut = train_test_split(X_clean, test_size=0.3, random_state=42)
    X_val_nut, X_test_nut = train_test_split(X_temp_nut, test_size=0.5, random_state=42)

    splits[nut_type] = {'cal': X_cal_nut, 'val': X_val_nut, 'test': X_test_nut}
    print(f"  {nut_type}: raw samples after cleaning={X_clean.shape[0]} -> cal={X_cal_nut.shape}, val={X_val_nut.shape}, test={X_test_nut.shape}")

# Build global test set (concatenate all per-nut test parts)
Xts_data_list = []
Xts_label_list = []
for nut_type in nut_types:
    X_test_nut = splits[nut_type]['test']
    if X_test_nut.shape[0] == 0:
        continue
    test_labels = np.zeros(X_test_nut.shape[0], dtype=int)
    if nut_type != TARGET_NUT:
        test_labels[:] = 1
    Xts_data_list.append(X_test_nut)
    Xts_label_list.append(test_labels)

if len(Xts_data_list) == 0:
    Xts_data = np.empty((0, n_wavelengths), dtype=np.float32)
    Xts_label = np.array([], dtype=int)
else:
    Xts_data = np.vstack(Xts_data_list)
    Xts_label = np.concatenate(Xts_label_list)

# Select target-specific calibration/validation sets
X_cal = splits[TARGET_NUT]['cal']
X_val = splits[TARGET_NUT]['val']
X_test_in = splits[TARGET_NUT]['test']

print(f"Calibration shape: {X_cal.shape}")
print(f"Validation shape: {X_val.shape}")
print(f"Test (in-class) shape: {X_test_in.shape}")
print(f"Test data shape (global): {Xts_data.shape}")
print(f"Test labels distribution: {np.bincount(Xts_label) if Xts_label.size else 'empty'}\n")

# Note: outlier removal is performed per-nut before splitting (see above). Skipping additional calibration outlier step.
print(f"Calibration shape after preprocessing: {X_cal.shape}\n")

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ===========================
# Parameter sweep (grid search)
# ===========================

print("Starting parameter sweep for VAE-SIMCA on peanut data...\n")

n_wavelengths = X_cal.shape[1]

# Output base dir
output_dir = os.path.join(data_root, data_folder, "vae_simca_analysis", TARGET_NUT)
os.makedirs(output_dir, exist_ok=True)
print(f"Results will be saved to: {output_dir}\n")

# Base training params (can be adjusted)
base_params = {
    'batch_size': 512,
    'n_epochs': 2000,
    'weight_decay': 1e-5,
    'beta': 1.0,
    'early_stop_patience': 30,
}

# Parameter grid (tune these lists as needed)
param_grid = {
    'latent_dim': [16, 32],
    'hidden_fc': [32, 64,128],
    'learning_rate': [1e-4],
    'conv_blocks': [1, 2,3],
    'n_filters': [1, 3],
    'kernel_size': [5, 7],
    'dropout': [0.1],
}

# Build list of parameter combinations
keys, values = zip(*param_grid.items())
param_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

all_metrics = []

for run_idx, params in enumerate(param_list):
    run_name = f"run_{run_idx:03d}"
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Running {run_name}: {params}")

    # Prepare model
    mean_cal = np.mean(X_cal, axis=0)
    std_cal = np.std(X_cal, axis=0) + 1e-8

    vae = ConvVAE1D(
        input_length=n_wavelengths,
        latent_dim=params['latent_dim'],
        mean=mean_cal,
        std=std_cal,
        conv_blocks=params['conv_blocks'],
        n_filters=params['n_filters'],
        kernel_size=params['kernel_size'],
        stride=2,
        hidden_fc=params['hidden_fc'],
        activation="elu",
        dropout=params['dropout'],
        use_batchnorm=True,
        beta=base_params['beta']
    ).to(device)

    optimizer = optim.Adam(vae.parameters(), lr=params['learning_rate'], weight_decay=base_params['weight_decay'])

    train_loader = data_utils.DataLoader(
        data_utils.TensorDataset(torch.tensor(X_cal, dtype=torch.float32)),
        batch_size=base_params['batch_size'], shuffle=True
    )
    val_loader = data_utils.DataLoader(
        data_utils.TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
        batch_size=base_params['batch_size'], shuffle=False
    )

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    # Training loop with simple early stopping
    for epoch in range(base_params['n_epochs']):
        vae.train()
        train_loss = 0.0
        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            x_rec, mu, logvar = vae(x_batch)
            loss, recon_loss, kl_loss = beta_vae_cosine_loss(x_batch, x_rec, mu, logvar, beta=vae.beta)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        vae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x_batch,) in val_loader:
                x_batch = x_batch.to(device)
                x_rec, mu, logvar = vae(x_batch)
                loss, recon_loss, kl_loss = beta_vae_cosine_loss(x_batch, x_rec, mu, logvar, beta=vae.beta)
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Verbose
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{base_params['n_epochs']} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        # No early stopping: keep training for all epochs
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(vae.state_dict())
        # (early stopping disabled for long training runs)

    # Save training history
    np.save(os.path.join(run_dir, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(run_dir, 'val_losses.npy'), np.array(val_losses))

    # Load best model
    if best_state is not None:
        vae.load_state_dict(best_state)

    # ----- Compute latent stats on calibration set (batched encoding)
    vae.eval()
    mus_train_list = []
    with torch.no_grad():
        for (x_batch,) in train_loader:
            x = x_batch.to(device)
            x_std = (x - vae.spec_mean) / vae.spec_std
            mu_t, _ = vae.encode(x_std)
            mus_train_list.append(mu_t.cpu().numpy())
    mus_train = np.concatenate(mus_train_list, axis=0)
    mu_train_mean = np.mean(mus_train, axis=0)
    cov = np.cov(mus_train, rowvar=False) + np.eye(mus_train.shape[1]) * 1e-6
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)
    threshold = float(np.percentile(np.einsum('ij,jk,ik->i', mus_train - mu_train_mean, cov_inv, mus_train - mu_train_mean), 95))

    # ----- (Q computation commented out for single-criterion testing)
    # q_cal_list = []
    # with torch.no_grad():
    #     for (x_batch,) in train_loader:
    #         x = x_batch.to(device)
    #         x_std = (x - vae.spec_mean) / vae.spec_std
    #         mu_t, _ = vae.encode(x_std)
    #         x_rec_std = vae.decode(mu_t)
    #         x_rec = x_rec_std * vae.spec_std + vae.spec_mean
    #         q_batch = np.mean((x.cpu().numpy() - x_rec.cpu().numpy()) ** 2, axis=1)
    #         q_cal_list.append(q_batch)
    # q_cal = np.concatenate(q_cal_list)
    # q_threshold = float(np.percentile(q_cal, 95))

    # ----- Save latent stats inside model buffers
    vae.latent_mean.copy_(torch.tensor(mu_train_mean, dtype=torch.float32))
    vae.latent_cov_inv.copy_(torch.tensor(cov_inv, dtype=torch.float32))
    vae.threshold.copy_(torch.tensor(threshold, dtype=torch.float32))
    # Q threshold intentionally NOT saved while testing single D^2 criterion

    # ----- Evaluate on test set using Mahalanobis D^2 in latent space
    test_loader = data_utils.DataLoader(
        data_utils.TensorDataset(torch.tensor(Xts_data, dtype=torch.float32), torch.tensor(Xts_label, dtype=torch.long)),
        batch_size=base_params['batch_size'], shuffle=False
    )

    d2_list = []
    # q_ts_list = []  # commented out (not using Q for now)
    labels_true_list = []
    with torch.no_grad():
        for xb, yb in test_loader:
            x = xb.to(device)
            x_std = (x - vae.spec_mean) / vae.spec_std
            mu_t, _ = vae.encode(x_std)
            mu_np = mu_t.cpu().numpy()
            diff = mu_np - vae.latent_mean.cpu().numpy()[None, :]
            d2 = np.einsum('ij,jk,ik->i', diff, vae.latent_cov_inv.cpu().numpy(), diff)
            d2_list.append(d2)

            # Q computation commented out for single-criterion testing
            # x_rec_std = vae.decode(mu_t)
            # x_rec = x_rec_std * vae.spec_std + vae.spec_mean
            # q = np.mean((x.cpu().numpy() - x_rec.cpu().numpy()) ** 2, axis=1)
            # q_ts_list.append(q)

            labels_true_list.append(yb.cpu().numpy())

    d2_ts = np.concatenate(d2_list)
    # q_ts = np.concatenate(q_ts_list)  # commented out
    labels_true = np.concatenate(labels_true_list)

    # Predictions using D^2 threshold saved in model
    pred_class0 = d2_ts <= vae.threshold.item()
    pred_labels = np.where(pred_class0, 0, 1)

    # For plotting, use d2_ts as T^2 and q_ts as Q

    # Confusion matrix
    conf_mat = np.zeros((2, 2), dtype=int)
    conf_mat[0, 0] = np.sum((pred_labels == 0) & (Xts_label == 0))  # TP
    conf_mat[0, 1] = np.sum((pred_labels == 0) & (Xts_label == 1))  # FP
    conf_mat[1, 0] = np.sum((pred_labels == 1) & (Xts_label == 0))  # FN
    conf_mat[1, 1] = np.sum((pred_labels == 1) & (Xts_label == 1))  # TN

    TP = conf_mat[0, 0]
    FP = conf_mat[0, 1]
    FN = conf_mat[1, 0]
    TN = conf_mat[1, 1]

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    metrics = {
        'run': run_name,
        'params': params,
        'best_val_loss': float(best_val_loss),
        'd2_threshold': float(vae.threshold.item()),
        'confusion_matrix': conf_mat.tolist(),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'precision': precision,
        'f1': f1,
    }

    # Save metrics and model
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as fh:
        json.dump(metrics, fh, indent=2)
    torch.save(vae.state_dict(), os.path.join(run_dir, 'best_model.pt'))

    # Save plots: confusion matrix
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=[TARGET_NUT, "Others"],
                yticklabels=["conform", "unconform"], ax=ax)
    ax.set_xlabel("True class")
    ax.set_ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "confusion_matrix.pdf"))
    plt.close()

    # T-Q scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ['green', 'red']
    labels = [TARGET_NUT, 'Others']
    # Plot D^2 per sample (log scale) colored by true label
    idxs = np.arange(len(d2_ts))
    for j, true_label in enumerate([0, 1]):
        mask = labels_true == true_label
        ax.scatter(idxs[mask], d2_ts[mask],
                   s=40, edgecolor='k', linewidth=0.5, alpha=0.7,
                   color=colors[j], label=labels[j])
    ax.axhline(y=vae.threshold.item(), color='b', linestyle='--', lw=2, label='D² threshold')
    ax.set_xlabel('Sample index')
    ax.set_ylabel(r"$D^2$")
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-4)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "T2_Q_plot.pdf"))
    plt.close()

    # Training history
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label='Train loss')
    ax.plot(val_losses, label='Val loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "training_history.pdf"))
    plt.close()

    all_metrics.append(metrics)

# Save summary
with open(os.path.join(output_dir, 'all_metrics.json'), 'w') as fh:
    json.dump(all_metrics, fh, indent=2)

print("Parameter sweep completed. Summary saved to all_metrics.json")


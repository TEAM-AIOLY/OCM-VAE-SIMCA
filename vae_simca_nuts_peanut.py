import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from vae_model import ConvVAE1D

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

# Target nut type
TARGET_NUT = 'peanut'
print(f"Target nut type: {TARGET_NUT}\n")

# ===========================
# Data preparation for peanut
# ===========================

print(f"Processing {TARGET_NUT}:")

# Collect all spectral data for peanut
all_spectra = []
for obj in data[TARGET_NUT]:
    spectral_data = np.array(obj['spectral_data'], dtype=np.float32)
    all_spectra.append(spectral_data)

# Stack all spectra into one array
X = np.vstack(all_spectra)
print(f"  Total spectra shape: {X.shape}")


# Check for NaN/inf values
nan_mask = np.isnan(X).any(axis=1)
inf_mask = np.isinf(X).any(axis=1)
bad_mask = nan_mask | inf_mask
if np.any(bad_mask):
    print(f"  WARNING: Found {np.sum(bad_mask)} samples with NaN/inf values. Removing them.")
    X = X[~bad_mask]
    print(f"  Shape after cleaning: {X.shape}")

# Create labels (all 0 = in-class for peanut)
y = np.zeros(X.shape[0], dtype=int)

# Split into calibration (70%), validation (15%), test (15%)
X_cal, X_temp, y_cal, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test_in, y_val, y_test_in = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"  Calibration shape: {X_cal.shape}")
print(f"  Validation shape: {X_val.shape}")
print(f"  Test (in-class) shape: {X_test_in.shape}\n")

# Prepare test data with all nut types (anomalies)
Xts_data_list = []
Xts_label_list = []

for nut_type in nut_types:
    # Get test data from this nut type
    all_spectra_test = []
    for obj in data[nut_type]:
        spectral_data = np.array(obj['spectral_data'], dtype=np.float32)
        all_spectra_test.append(spectral_data)
    X_nut = np.vstack(all_spectra_test)
    
    # Clean NaN/inf
    bad_mask_nut = np.isnan(X_nut).any(axis=1) | np.isinf(X_nut).any(axis=1)
    X_nut = X_nut[~bad_mask_nut]
    
    # Get test portion only
    _, X_test_nut, _, y_test_nut = train_test_split(
        X_nut, np.full(X_nut.shape[0], nut_type_to_label[nut_type]), 
        test_size=0.15, random_state=42
    )
    
    # Set label: 0 if peanut, 1 if other
    test_labels = np.zeros(X_test_nut.shape[0], dtype=int)
    if nut_type != TARGET_NUT:
        test_labels[:] = 1
    
    Xts_data_list.append(X_test_nut)
    Xts_label_list.append(test_labels)

Xts_data = np.vstack(Xts_data_list)
Xts_label = np.concatenate(Xts_label_list)

print(f"Test data shape: {Xts_data.shape}")
print(f"Test labels distribution: {np.bincount(Xts_label)}\n")

# Remove outliers from calibration in PCA space
print("Removing outliers from calibration data...")
pca_temp = PCA(n_components=10)
T_cal = pca_temp.fit_transform(X_cal)

mean_scores = np.mean(T_cal, axis=0)
cov_scores = np.cov(T_cal, rowvar=False)
cov_scores_inv = np.linalg.pinv(cov_scores)

mahal_dists = []
for i in range(T_cal.shape[0]):
    diff = T_cal[i] - mean_scores
    md = np.sqrt(diff @ cov_scores_inv @ diff.T)
    mahal_dists.append(md)
mahal_dists = np.array(mahal_dists)

outlier_threshold = np.percentile(mahal_dists, 95)
outlier_mask = mahal_dists <= outlier_threshold
X_cal = X_cal[outlier_mask]

n_outliers = np.sum(~outlier_mask)
print(f"Removed {n_outliers} outliers")
print(f"Calibration shape after cleaning: {X_cal.shape}\n")

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ===========================
# Training
# ===========================

print("Training VAE on peanut calibration data...\n")

n_wavelengths = X_cal.shape[1]
batch_size = 512
n_epochs = 1000
learning_rate = 0.0001

# Compute mean and std on calibration data
mean_cal = np.mean(X_cal, axis=0)
std_cal = np.std(X_cal, axis=0) + 1e-8

vae = ConvVAE1D(
    input_length=n_wavelengths, 
    latent_dim=20,
    mean=mean_cal,
    std=std_cal,
    conv_blocks=2,
    n_filters=3,
    kernel_size=5,
    stride=2,
    hidden_fc=32,
    activation="elu",
    dropout=0.1,
    use_batchnorm=True,
    beta=1.0
).to(device)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate, weight_decay=1e-5)

train_loader = data_utils.DataLoader(
    data_utils.TensorDataset(torch.tensor(X_cal, dtype=torch.float32)),
    batch_size=batch_size, shuffle=True
)

val_loader = data_utils.DataLoader(
    data_utils.TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
    batch_size=batch_size, shuffle=False
)

train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(n_epochs):
    # Training
    vae.train()
    train_loss = 0.0
    for (x_batch,) in train_loader:
        x_batch = x_batch.to(device)
        optimizer.zero_grad()
        
        x_rec, mu, logvar = vae(x_batch)
        
        # Cosine loss (same as VAE_cheese)
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
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{n_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss

print(f"\nTraining completed. Best val loss: {best_val_loss:.6f}\n")

# ===========================
# Compute SIMCA statistics
# ===========================

print("Computing SIMCA statistics on calibration data...")

vae.eval()
with torch.no_grad():
    X_cal_tensor = torch.tensor(X_cal, dtype=torch.float32).to(device)
    _, mu_cal, _ = vae(X_cal_tensor)
    mu_cal = mu_cal.cpu().numpy()
    
    # Reconstruction errors
    x_rec_cal, _, _ = vae(X_cal_tensor)
    x_rec_cal = x_rec_cal.cpu().numpy()
    q_cal = np.mean((X_cal - x_rec_cal) ** 2, axis=1)

# Latent space statistics
mu_mean = np.mean(mu_cal, axis=0)
mu_cov = np.cov(mu_cal, rowvar=False) + np.eye(mu_cal.shape[1]) * 1e-6
mu_cov_inv = np.linalg.inv(mu_cov)

# T² thresholds
h_cal = np.sum((mu_cal - mu_mean) ** 2, axis=1) / np.mean(np.sum((mu_cal - mu_mean) ** 2, axis=1))
t2_threshold = np.percentile(h_cal, 95)

# Q threshold
q_threshold = np.percentile(q_cal, 95)

print(f"T² threshold (95%): {t2_threshold:.4f}")
print(f"Q threshold (95%): {q_threshold:.4f}\n")

# ===========================
# Test on full dataset
# ===========================

print("Evaluating on test data...")

with torch.no_grad():
    Xts_tensor = torch.tensor(Xts_data, dtype=torch.float32).to(device)
    _, mu_ts, _ = vae(Xts_tensor)
    mu_ts = mu_ts.cpu().numpy()
    
    x_rec_ts, _, _ = vae(Xts_tensor)
    x_rec_ts = x_rec_ts.cpu().numpy()
    q_ts = np.mean((Xts_data - x_rec_ts) ** 2, axis=1)

# Compute distances
h_ts = np.sum((mu_ts - mu_mean) ** 2, axis=1) / np.mean(np.sum((mu_ts - mu_mean) ** 2, axis=1))

# Combined distance
f_ts = h_ts + q_ts

# Predictions
pred_ts = f_ts <= (t2_threshold + q_threshold)
pred_labels = np.where(pred_ts, 0, 1)

# ===========================
# Metrics and plots
# ===========================

output_dir = os.path.join(data_root, data_folder, "vae_simca_analysis", TARGET_NUT)
os.makedirs(output_dir, exist_ok=True)

print(f"\nResults saved to: {output_dir}\n")

# Confusion matrix
conf_mat = np.zeros((2, 2), dtype=int)
conf_mat[0, 0] = np.sum((pred_labels == 0) & (Xts_label == 0))  # TP
conf_mat[0, 1] = np.sum((pred_labels == 0) & (Xts_label == 1))  # FP
conf_mat[1, 0] = np.sum((pred_labels == 1) & (Xts_label == 0))  # FN
conf_mat[1, 1] = np.sum((pred_labels == 1) & (Xts_label == 1))  # TN

print("Confusion Matrix:")
print(conf_mat)
print()

# Metrics
TP = conf_mat[0, 0]
FP = conf_mat[0, 1]
FN = conf_mat[1, 0]
TN = conf_mat[1, 1]

sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1: {f1:.4f}\n")

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=[TARGET_NUT, "Others"],
            yticklabels=["conform", "unconform"], ax=ax)
ax.set_xlabel("True class")
ax.set_ylabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.pdf"))
plt.close()

# Plot T-Q scatter
fig, ax = plt.subplots(figsize=(8, 8))
colors = ['green', 'red']
labels = [TARGET_NUT, 'Others']

for j, true_label in enumerate([0, 1]):
    ax.scatter(h_ts[Xts_label == true_label], q_ts[Xts_label == true_label],
               s=40, edgecolor='k', linewidth=0.5, alpha=0.7,
               color=colors[j], label=labels[j])

# Decision boundaries
ax.axhline(y=q_threshold, color='b', linestyle='--', lw=2, label='Q threshold')
ax.axvline(x=t2_threshold, color='b', linestyle='--', lw=2, label='T² threshold')

ax.set_xlabel(r"$T^2$")
ax.set_ylabel(r"$Q$")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(left=1e-2)
ax.set_ylim(bottom=1e-2)
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "T2_Q_plot.pdf"))
plt.close()

# Plot training history
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_losses, label='Train loss')
ax.plot(val_losses, label='Val loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "training_history.pdf"))
plt.close()

print("Plots saved!")

import sys
import os
import numpy as np
import scipy as sp
import seaborn as sns
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
from pathlib import Path
import itertools
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils
from torch import nn, optim

# ---------------------------
# Repro / device / root
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = os.getcwd()

# ---------------------------
# File helpers
# ---------------------------
def make_save_path(*args):
    path = Path(os.getcwd()).joinpath(*args)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_model(model, path: Path, filename="model.pth", overwrite=True):
    file_path = Path(path) / filename
    if file_path.exists() and overwrite:
        try: os.remove(file_path)
        except Exception as e: print(f"Could not remove old model: {e}")
    torch.save(model.state_dict(), file_path)
    return file_path

def save_metrics(metrics: dict, path: Path, filename="metrics.json", overwrite=True):
    file_path = Path(path) / filename
    if file_path.exists() and overwrite:
        try: os.remove(file_path)
        except Exception as e: print(f"Could not remove old metrics: {e}")
    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return file_path

def save_plot(fig, path: Path, filename: str, fmt="pdf", overwrite=True):
    file_path = Path(path) / f"{filename}.{fmt}"
    if file_path.exists() and overwrite:
        try: os.remove(file_path)
        except Exception as e: print(f"Could not remove old plot: {e}")
    fig.savefig(file_path, format=fmt)
    plt.close(fig)
    return file_path

class ConvVAE1D(nn.Module):
    def __init__(
        self,
        input_length,
        latent_dim,
        mean,
        std,
        conv_blocks=3,
        n_filters=32,
        kernel_size=9,
        stride=2,
        hidden_fc=256,
        activation="elu",
        dropout=0.0,
        use_batchnorm=True,
        beta=1.0,
    ):
        super().__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim
        self.beta = beta
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.register_buffer("threshold", torch.tensor(0.0))
        self.register_buffer("q_threshold", torch.tensor(0.0))  # <-- Q threshold buffer

        act = nn.ELU if activation == "elu" else nn.GELU
        padding = kernel_size // 2

        # Encoder conv
        enc_blocks = []
        in_ch = 1
        out_len = input_length
        filters = n_filters
        for b in range(conv_blocks):
            stride_b = 1 if b == 0 else stride
            enc_blocks.append(nn.Conv1d(in_ch, filters, kernel_size, stride=stride_b, padding=padding))
            if use_batchnorm:
                enc_blocks.append(nn.BatchNorm1d(filters))
            enc_blocks.append(act())
            if dropout > 0:
                enc_blocks.append(nn.Dropout(dropout))
            in_ch = filters
            filters = min(filters * 2, 1024)
            out_len = (out_len + 2*padding - (kernel_size-1) -1)//stride_b + 1
        self.encoder_conv = nn.Sequential(*enc_blocks)
        self._enc_out_channels = in_ch
        self._enc_out_length = out_len

        # Fully connected
        fc_in = self._enc_out_channels * self._enc_out_length
        self.fc = nn.Sequential(nn.Linear(fc_in, hidden_fc), act(), nn.Dropout(dropout) if dropout>0 else nn.Identity())
        self.fc_mu = nn.Linear(hidden_fc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_fc, latent_dim)

        # Decoder
        self.fc_dec = nn.Sequential(nn.Linear(latent_dim, hidden_fc), act(), nn.Dropout(dropout) if dropout>0 else nn.Identity(),
                                    nn.Linear(hidden_fc, fc_in), act())

        dec_blocks = []
        in_ch = self._enc_out_channels
        filters = in_ch
        for b in range(conv_blocks):
            next_filters = max(filters//2, n_filters)
            stride_b = stride if b < conv_blocks-1 else 1
            dec_blocks.append(nn.ConvTranspose1d(filters, next_filters, kernel_size, stride=stride_b, padding=padding, output_padding=(stride_b-1)))
            if use_batchnorm:
                dec_blocks.append(nn.BatchNorm1d(next_filters))
            dec_blocks.append(act())
            if dropout > 0:
                dec_blocks.append(nn.Dropout(dropout))
            filters = next_filters
        dec_blocks.append(nn.Conv1d(filters, 1, kernel_size=1))
        self.decoder_conv = nn.Sequential(*dec_blocks)

        self.register_buffer("spec_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("spec_std", torch.tensor(std, dtype=torch.float32))

        # Latent stats
        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("latent_cov_inv", torch.eye(latent_dim))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, x):
        x = x.unsqueeze(1)  # (B,1,L)
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5*logvar)

    def decode(self, z):
        h = self.fc_dec(z)
        B = h.size(0)
        h = h.view(B, self._enc_out_channels, self._enc_out_length)
        x_rec = self.decoder_conv(h).squeeze(1)
        if x_rec.shape[-1] > self.input_length:
            x_rec = x_rec[..., :self.input_length]
        elif x_rec.shape[-1] < self.input_length:
            pad = x_rec.new_zeros(B, self.input_length - x_rec.shape[-1])
            x_rec = torch.cat([x_rec, pad], dim=1)
        return x_rec

    def forward(self, x):
        x_std = (x - self.spec_mean) / self.spec_std
        mu, logvar = self.encode(x_std)
        z = self.reparameterize(mu, logvar)
        x_rec_std = self.decode(z)
        x_rec = x_rec_std * self.spec_std + self.spec_mean
        return x_rec, mu, logvar

def compute_rec_error(x: torch.Tensor, x_rec: torch.Tensor, mode: str = "euclidean") -> np.ndarray:
    """
    Compute per-sample reconstruction error.

    Args:
        x (torch.Tensor): Original input, shape (B, L)
        x_rec (torch.Tensor): Reconstructed input, shape (B, L)
        mode (str): "euclidean" or "cosine"

    Returns:
        np.ndarray: Reconstruction error per sample
    """
    if mode == "euclidean":
        # standard L2 squared error per sample
        rec_err = ((x - x_rec) ** 2).sum(dim=1).cpu().numpy()
    elif mode == "cosine":
        # cosine distance per sample
        x_flat = x.view(x.size(0), -1)
        xrec_flat = x_rec.view(x_rec.size(0), -1)
        x_norm = F.normalize(x_flat, p=2, dim=1)
        xrec_norm = F.normalize(xrec_flat, p=2, dim=1)
        cos_sim = torch.sum(x_norm * xrec_norm, dim=1)
        rec_err = torch.sqrt(2*(1 - cos_sim)).cpu().numpy()
    else:
        raise ValueError(f"Unknown mode {mode}, choose 'euclidean' or 'cosine'.")
    return rec_err
# ---------------------------
# Cosine-based VAE loss
def beta_vae_cosine_loss(x, x_recon, mu, logvar, beta=1.0, eps=1e-8):
    x_flat = x.view(x.size(0), -1)
    x_recon_flat = x_recon.view(x_recon.size(0), -1)
    x_norm = F.normalize(x_flat, p=2, dim=1)
    recon_norm = F.normalize(x_recon_flat, p=2, dim=1)
    cos_theta = torch.clamp(torch.sum(x_norm * recon_norm, dim=1), -1.0 + eps, 1.0 - eps)
    recon_loss = torch.mean(torch.sqrt(2.0 * (1.0 - cos_theta)))
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return recon_loss + beta*kl, recon_loss.detach().cpu().item(), kl.detach().cpu().item()

def beta_vae_euclidean_loss(x, x_recon, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return recon_loss + beta*kl, recon_loss.detach().cpu().item(), kl.detach().cpu().item()

def beta_vae_bce_loss(x, x_recon, mu, logvar, beta=1.0, eps=1e-8):
    # x and x_recon: raw input, same as current code
    # scale to [0,1] per sample for BCE only
    x_min = x.min(dim=1, keepdim=True)[0]
    x_max = x.max(dim=1, keepdim=True)[0]
    x_scaled = (x - x_min) / (x_max - x_min + 1e-12)
    x_recon_scaled = (x_recon - x_min) / (x_max - x_min + 1e-12)
    # BCE reconstruction
    recon_loss = F.binary_cross_entropy(x_recon_scaled, x_scaled, reduction='mean')
    # KL divergence same as before
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return recon_loss + beta*kl, recon_loss.detach().cpu().item(), kl.detach().cpu().item()
# ---------------------------
# Load data
# ---------------------------
data_path="C:/00_aioly/GitHub/Deep-chemometrics/data/dataset/Ale/IR_ML.mat"
data = sp.io.loadmat(data_path)
data_dict = {k:v for k,v in data.items() if not k.startswith('_')}

Xtr = data_dict['Xtr']
Xts = data_dict['Xts']
Xtr_dict = {key:Xtr[0][0][i] for i,key in enumerate(Xtr.dtype.names)}
Xts_dict = {key:Xts[0][0][i] for i,key in enumerate(Xts.dtype.names)}
Xtr_data = Xtr_dict['data']
Xts_data = Xts_dict['data']
Xtr_label = np.squeeze(Xtr_dict['class'][0][0]).astype(int)-1
Xts_label = np.squeeze(Xts_dict['class'][0][0]).astype(int)-1

n_classes = len(np.unique(np.concatenate([Xtr_label,Xts_label])))
ytr = np.eye(n_classes, dtype=np.float32)[Xtr_label]
yts = np.eye(n_classes, dtype=np.float32)[Xts_label]

# ---------------------------
# Calibration/validation split
# ---------------------------
X_cal, X_val, y_cal, y_val = train_test_split(
    Xtr_data, ytr, test_size=0.2, random_state=42, stratify=np.argmax(ytr,axis=1)
)

# ---------------------------
# One-class VAE on class0
# ---------------------------
class_idx = 0
X_cal_class0 = X_cal[np.argmax(y_cal, axis=1) == class_idx]
X_val_class0 = X_val[np.argmax(y_val, axis=1) == class_idx]

mean = np.mean(X_cal_class0, axis=0)
std = np.std(X_cal_class0, axis=0) + 1e-12

# ---------------------------
# Sweep parameters
# ---------------------------
base_params = {
    "spec_dims": None,
    "mean": None,
    "std": None,
    "DP" : 0.2,
    "LR": 0.001,
    "EPOCH": 3000,
    "WD": 0.003/2,
    "batch_size" : 512
}

LATENT_DIMS = [21,30]
HIDDEN_DIMS = [64,128,256]
LRS = [0.0001]
CONV_BLOCKS = [1,2]
N_FILTERS   = [1,3]
KERNELS     = [3,5]
DP=[0.1,0.2]

param_variations = [
    {
        "latent_dim": ld,
        "hidden_dim": hd,
        "LR": lr,
        "conv_blocks": cb,
        "n_filters": nf,
        "kernel_size": ks,
        "DP": dp
    }
    for ld,hd,lr,cb,nf,ks,dp in itertools.product(
        LATENT_DIMS,HIDDEN_DIMS,LRS,CONV_BLOCKS,N_FILTERS,KERNELS, DP
    )
]

paramsets = [{**base_params, **v} for v in param_variations]
loss_type = "X_cosine"
model_type = f"VAE_cheeseQ_{loss_type}"
process_id = os.path.join("Ale","cheese",model_type)
# ---------------------------
# Training / evaluation loop
all_params = []
all_metrics = []
spec_dims = X_cal.shape[1]

for i,param in enumerate(paramsets):

    param_id = f"Run_{i:02d}"
    print(f"running {param_id} with parameters: {param}")
    param_with_id = param.copy(); param_with_id["Run_ID"]=param_id
    all_params.append(param_with_id)

    local_run = os.path.join(process_id,param_id)
    base_path = os.path.join(root,local_run)
    os.makedirs(base_path,exist_ok=True)
    save_metrics(param, Path(base_path),"params.json")

    cal_loader_class0 = data_utils.DataLoader(
        data_utils.TensorDataset(torch.tensor(X_cal_class0,dtype=torch.float32)),
        batch_size=param["batch_size"],shuffle=True
    )
    val_loader_class0 = data_utils.DataLoader(
        data_utils.TensorDataset(torch.tensor(X_val_class0,dtype=torch.float32)),
        batch_size=param["batch_size"],shuffle=False
    )
    test_loader = data_utils.DataLoader(
        data_utils.TensorDataset(torch.tensor(Xts_data,dtype=torch.float32), torch.tensor(yts,dtype=torch.float32)),
        batch_size=param["batch_size"],shuffle=False
    )

    vae = ConvVAE1D(
        input_length=spec_dims,
        latent_dim=param["latent_dim"],
        conv_blocks=param["conv_blocks"],
        n_filters=param["n_filters"],
        kernel_size=param["kernel_size"],
        hidden_fc=param["hidden_dim"],
        dropout=param["DP"],
        mean=mean,
        std=std,
        activation="elu"
    ).to(device)
    
    nb_train_params = sum(p.numel() for p in vae.parameters())
    optimizer = optim.Adam(vae.parameters(), lr=param["LR"], weight_decay=param["WD"])

    train_losses, val_losses, val_metrics = [], [], []
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(int(param["EPOCH"])):
        # ---------------------------
        # Training
        vae.train()
        epoch_loss = 0.0
        for xb in cal_loader_class0:
            xb = xb[0].to(device)
            optimizer.zero_grad()
            xb_recon, mu, logvar = vae(xb)
            if loss_type == "X_cosine":
               loss, _, _ = beta_vae_cosine_loss(xb, xb_recon, mu, logvar)
            if loss_type == "X_euclidean":  
                loss, _, _ = beta_vae_euclidean_loss(xb, xb_recon, mu, logvar)
            elif loss_type == "X_bce":
                 loss, _, _ = beta_vae_bce_loss(xb, xb_recon, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= max(1, len(cal_loader_class0.dataset))
        train_losses.append(epoch_loss)

        # ---------------------------
        # Validation
        vae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb in val_loader_class0:
                xb = xb[0].to(device)
                xrec, mu, logvar = vae(xb)
                if loss_type == "X_cosine":
                    loss, _, _ = beta_vae_cosine_loss(xb, xrec, mu, logvar)
                if loss_type == "X_euclidean":  
                    loss, _, _ = beta_vae_euclidean_loss(xb, xrec, mu, logvar)
                elif loss_type == "X_bce":
                    loss, _, _ = beta_vae_bce_loss(xb, xrec, mu, logvar)
                val_loss += loss.item() * xb.size(0)     
                       
        val_loss /= max(1, len(val_loader_class0.dataset))
        val_losses.append(val_loss)
        val_metrics.append(val_loss)

        if (epoch+1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{param['EPOCH']} | Train: {epoch_loss:.6f} | Val: {val_loss:.6f}")

        # ---------------------------
        # Save best model automatically
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            # ----- Compute latent stats on calibration set
            vae.eval()
            mus_train_list = []
            rec_errors_list = []  # <-- Q errors
            with torch.no_grad():
                for xb in cal_loader_class0:
                    x = xb[0].to(device)
                    x_std = (x - vae.spec_mean) / vae.spec_std
                    mu_t, _ = vae.encode(x_std)
                    mus_train_list.append(mu_t.cpu().numpy())
                    x_rec, _, _ = vae(x)
                    rec_err = ((x - x_rec)**2).sum(dim=1).cpu().numpy()
                    rec_errors_list.append(rec_err)
            mus_train = np.concatenate(mus_train_list, axis=0)
            mu_train_mean = np.mean(mus_train, axis=0)
            cov = np.cov(mus_train, rowvar=False) + np.eye(mus_train.shape[1]) * 1e-6
            try:
                cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov)
            threshold = float(np.percentile(np.einsum('ij,jk,ik->i', mus_train - mu_train_mean, cov_inv, mus_train - mu_train_mean), 95))
            q_threshold = float(np.percentile(np.concatenate(rec_errors_list), 95))  # <-- Q threshold 95%

            # ----- Save latent stats inside model
            vae.latent_mean.copy_(torch.tensor(mu_train_mean, dtype=torch.float32))
            vae.latent_cov_inv.copy_(torch.tensor(cov_inv, dtype=torch.float32))
            vae.threshold.copy_(torch.tensor(threshold, dtype=torch.float32))
            vae.q_threshold.copy_(torch.tensor(q_threshold, dtype=torch.float32))  # <-- store Q

            # ----- Save best model with updated stats
            save_model(vae, Path(base_path), "VAE_class0_best.pth")

    # ---------------------------
    # Save final loss history
    save_metrics({"train_losses": train_losses, "val_losses": val_losses}, Path(base_path), "losses.json")

    # ---------------------------
    # Load best model for evaluation
    vae_best = ConvVAE1D(
        input_length=spec_dims,
        latent_dim=param["latent_dim"],
        conv_blocks=param["conv_blocks"],
        n_filters=param["n_filters"],
        kernel_size=param["kernel_size"],
        hidden_fc=param["hidden_dim"],
        dropout=param["DP"],
        mean=mean,
        std=std,
        activation="elu"
    ).to(device)

    vae_best.load_state_dict(torch.load(Path(base_path) / "VAE_class0_best.pth", map_location=device))
    vae_best.eval()

    # ---------------------------
    # Benchmark on test set
    test_d2_list = []
    test_q_list = []  # <-- Q list
    labels_true_list = []
    with torch.no_grad():
        for xb, yb in test_loader:
            x = xb.to(device)
            x_std = (x - vae_best.spec_mean) / vae_best.spec_std
            mu_t, _ = vae_best.encode(x_std)
            mu_np = mu_t.cpu().numpy()
            diff = mu_np - vae_best.latent_mean.cpu().numpy()[None, :]
            d2 = np.einsum('ij,jk,ik->i', diff, vae_best.latent_cov_inv.cpu().numpy(), diff)
            test_d2_list.append(d2)
            x_rec, _, _ = vae_best(x)
            q = ((x - x_rec)**2).sum(dim=1).cpu().numpy()
            test_q_list.append(q)
            labels_true_list.append(np.argmax(yb.numpy(), axis=1))
    recon_errors = np.concatenate(test_d2_list)
    q_errors = np.concatenate(test_q_list)  # <-- Q
    labels_true = np.concatenate(labels_true_list)

    pred_class0 = (recon_errors <= vae_best.threshold.item()) & (q_errors <= vae_best.q_threshold.item())
    pred_labels = np.where(pred_class0, 0, 1)

    unique_true = np.unique(labels_true)
    n_true = len(unique_true)
    conf_mat_full = np.zeros((2, n_true), dtype=int)

    for i, pred in enumerate([0, 1]):  # 0=conform, 1=unconform
        for j, true_class in enumerate(unique_true):
            conf_mat_full[i, j] = np.sum((pred_labels == pred) & (labels_true == true_class))


    # Plot as before
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(conf_mat_full, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=[f"class{c+1}" for c in unique_true], yticklabels=["conform","unconform"], ax=ax)
    ax.set_xlabel("True class")
    ax.set_ylabel("Predicted")
    plt.tight_layout()
    save_plot(fig, Path(base_path), "   confusion_matrix", fmt="pdf")   
    
    TP = conf_mat_full[0, 0]          # conform correctly predicted
    FN = conf_mat_full[1, 0]          # conform rejected
    FP = conf_mat_full[0, 1:].sum()   # anomalies accepted as conform
    TN = conf_mat_full[1, 1:].sum()   # anomalies correctly rejected

    # binary metrics (conform=positive)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-12)
    precision = TP / (TP + FP + 1e-12)
    recall = TP / (TP + FN + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    # per-anomaly-class false acceptance rate
    fa_rates = conf_mat_full[0, 1:] / (conf_mat_full[:, 1:].sum(axis=0) + 1e-12)
    fa_mean = np.mean(fa_rates)

    metrics_dict = {
        'num_epochs': int(param['EPOCH']),
        'batch_size': int(param['batch_size']),
        'LR': float(param['LR']),
        'WD': float(param['WD']),
        'latent_dim': int(param['latent_dim']),
        'hidden_dim': int(param['hidden_dim']),
        'conv_blocks': int(param['conv_blocks']),
        'n_filters': int(param['n_filters']),
        'kernel_size': int(param['kernel_size']),
        'F1': float(f1),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'mean_false_acceptance': float(fa_mean),
        'false_acceptance_per_class': fa_rates.tolist(),
        'N_parameters': int(nb_train_params),
        'model_name': model_type,
        'Run_ID': param_id,
        'best_epoch': int(best_epoch),
        'Q_threshold': float(vae_best.q_threshold.item())  # <-- Q threshold
    }
    all_metrics.append(metrics_dict)

    with open(os.path.join(base_path,'metrics.txt'),'w') as f:
        for k,v in metrics_dict.items():
            f.write(f"{k}: {v}\n")

# ---------------------------
# Save all params & metrics summary
summary_dir = os.path.join(root,process_id)
os.makedirs(summary_dir, exist_ok=True)
with open(os.path.join(summary_dir,"all_params.json"),"w") as f:
    json.dump(all_params,f,indent=2)
with open(os.path.join(summary_dir,"all_metrics.json"),"w") as f:
    json.dump(all_metrics,f,indent=2)

print("Sweep finished. Summaries saved to:", summary_dir)

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
from sklearn.metrics import confusion_matrix,accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils
from torch import nn, optim
from scipy import special
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

# ---------------------------
# ConvVAE1D
# ---------------------------
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
        self.threshold = None

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
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
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


# ---------------------------
# SIMCA wrapper
# ---------------------------
import numpy as np
import torch

class VAESIMCA:
    def __init__(self, vae, type='alt', t2lim='Fdist', t2cl=0.95, qlim='jm', qcl=0.95, dcl=0.95, device='cpu', verbose=True):
        self.vae = vae
        self.device = device
        self.type = type
        self.t2lim = t2lim
        self.t2cl = t2cl
        self.qlim = qlim
        self.qcl = qcl
        self.dcl = dcl
        self.verbose = verbose

        self._model = {}  # dict[class_label] = model info
        self.model_class = None

    @torch.no_grad()
    def fit_thresholds(self, loader, class_label=0):
        """Fit latent SIMCA thresholds for a single class."""
        self.vae.eval()
        self.model_class = [class_label]

        # Collect latent vectors
        zs = []
        for xb in loader:
            x = xb[0].to(self.device)
            x_std = (x - self.vae.spec_mean) / self.vae.spec_std
            mu, _ = self.vae.encode(x_std)
            zs.append(mu.cpu().numpy())
        zs_np = np.concatenate(zs, axis=0)

        n_components = zs_np.shape[1]
        x_mean = np.mean(zs_np, axis=0)
        cov = np.cov(zs_np, rowvar=False) + np.eye(n_components) * 1e-12
        invcovT = np.linalg.pinv(cov)

        # T²
        diff = zs_np - x_mean[None, :]
        T2 = np.einsum('ij,jk,ik->i', diff, invcovT, diff)
        T2_limit, t2dof, t2scfact = self._compute_T2_limit(T2, n_components)

        # Q (latent residual)
        z_tensor = torch.tensor(zs_np, dtype=torch.float32).to(self.device)
        x_hat = self.vae.decode(z_tensor)
        z_hat, _ = self.vae.encode((x_hat - self.vae.spec_mean)/self.vae.spec_std)
        Q = torch.sum((z_tensor - z_hat)**2, dim=1).cpu().numpy()
        Q_limit, qdof, qscfact = self._compute_Q_limit(Q)

        # D limit
        D_limit = self._compute_D_limit(T2_limit, Q_limit, T2, Q, n_components, t2dof, t2scfact, qdof, qscfact)

        # Save class model
        self._model[class_label] = {
            "latent_mean": x_mean,
            "invcovT": invcovT,
            "T2": T2,
            "Q": Q,
            "T2_limit": T2_limit,
            "Q_limit": Q_limit,
            "D_limit": D_limit,
            "T2dof": t2dof,
            "T2scfact": t2scfact,
            "Qdof": qdof,
            "Qscfact": qscfact,
            "n_components": n_components
        }

    def _compute_T2_limit(self, T2, n_components):
        n_samples = len(T2)
        t2dof, t2scfact = None, None
        if self.t2lim == 'perc':
            T2_limit = np.percentile(T2, self.t2cl*100)
        elif self.t2lim == 'Fdist':
            F_value = np.percentile(T2, self.t2cl*100)
            T2_limit = n_components * (n_samples-1) / (n_samples - n_components) * F_value
        elif self.t2lim == 'chi2':
            T2_limit = np.percentile(T2, self.t2cl*100)
        elif self.t2lim == 'chi2pom':
            h0 = float(np.mean(T2))
            var_t2 = float(np.var(T2, ddof=1)) if len(T2)>1 else 0.0
            Nh = max(int(np.round(2*(h0**2)/var_t2)) if var_t2>0 else 1,1)
            T2_limit = h0 * np.percentile(T2, self.t2cl*100)/Nh
            t2dof = Nh
            t2scfact = h0
        else:
            raise ValueError(f"T2 limit type {self.t2lim} not implemented")
        return T2_limit, t2dof, t2scfact

    def _compute_Q_limit(self, Q):
        qdof, qscfact = None, None
        if self.qlim == 'perc':
            Q_limit = np.percentile(Q, self.qcl*100)
        elif self.qlim == 'jm':
            theta1 = Q.sum()
            theta2 = np.sum(Q**2)
            theta3 = np.sum(Q**3)
            if theta1 == 0:
                Q_limit = 0
            else:
                h0 = 1 - (2*theta1*theta3)/(3*theta2**2)
                h0 = max(h0, 1e-3)
                ca = np.sqrt(2)*special.erfinv(2*self.qcl-1)
                h1 = ca*np.sqrt(2*theta2*h0**2)/theta1
                h2 = theta2*h0*(h0-1)/(theta1**2)
                Q_limit = theta1*(1+h1+h2)**(1/h0)
        elif self.qlim == 'chi2pom':
            v0 = np.mean(Q)
            Nv = max(round(2*(v0**2)/np.var(Q, ddof=1)),1)
            Q_limit = v0 * np.percentile(Q, self.qcl*100)/Nv
            qdof = Nv
            qscfact = v0
        else:
            raise ValueError(f"Q limit type {self.qlim} not implemented")
        return Q_limit, qdof, qscfact

    def _compute_D_limit(self, T2_limit, Q_limit, T2, Q, n_components, t2dof=None, t2scfact=None, qdof=None, qscfact=None):
        if self.type == 'sim':
            D_limit = 1
        elif self.type == 'alt':
            D_limit = np.sqrt(2)
        elif self.type == 'ci':
            tr1 = (n_components/T2_limit) + (np.sum(Q)/Q_limit)
            tr2 = (n_components/T2_limit**2) + (np.sum(Q**2)/Q_limit**2)
            gd = tr2/tr1
            hd = tr1**2/tr2
            D_limit = gd * np.percentile(Q, self.dcl*100)
        elif self.type == 'dd':
            if t2dof is None or qdof is None:
                raise ValueError("t2dof/qdoff must be set for dd")
            D_limit = t2dof + qdof
        else:
            raise ValueError(f"D type {self.type} not implemented")
        return D_limit

    @torch.no_grad()
    def predict(self, loader):
        self.vae.eval()
        class_label = self.model_class[0]
        model_info = self._model[class_label]

        T2_all, Q_all = [], []
        for xb in loader:
            x = xb[0].to(self.device)
            x_std = (x - self.vae.spec_mean)/self.vae.spec_std
            mu, _ = self.vae.encode(x_std)
            mu_np = mu.cpu().numpy()

            diff = mu_np - model_info["latent_mean"][None,:]
            T2 = np.einsum('ij,jk,ik->i', diff, model_info["invcovT"], diff)
            x_hat = self.vae.decode(mu)
            z_hat, _ = self.vae.encode((x_hat - self.vae.spec_mean)/self.vae.spec_std)
            Q = torch.sum((mu - z_hat)**2, dim=1).cpu().numpy()

            T2_all.append(T2)
            Q_all.append(Q)

        T2_all = np.concatenate(T2_all)
        Q_all = np.concatenate(Q_all)

        if self.type == 'alt':
            D = np.sqrt((T2_all / model_info["T2_limit"])**2 + (Q_all / model_info["Q_limit"])**2)
        elif self.type == 'dd':
            D = (T2_all*model_info["T2dof"]/model_info["T2scfact"]) + (Q_all*model_info["Qdof"]/model_info["Qscfact"])
        else:
            D = np.maximum(T2_all/model_info["T2_limit"], Q_all/model_info["Q_limit"])

        y_pred = D < model_info["D_limit"]

        return y_pred, T2_all, Q_all


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

LATENT_DIMS = [31,41]
HIDDEN_DIMS = [64,128,256]
LRS = [0.001,0.00001]
CONV_BLOCKS = [1,2]
N_FILTERS   = [1,2]
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
model_type = "VAESIMCA_class0_dd"
process_id = os.path.join("Ale","cheese",model_type)

# ---------------------------
# Training / evaluation loop
# ---------------------------
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
        vae.train()
        epoch_loss = 0.0
        for xb in cal_loader_class0:
            xb = xb[0].to(device)
            optimizer.zero_grad()
            xb_recon, mu, logvar = vae(xb)
            loss,_,_ = beta_vae_cosine_loss(xb, xb_recon, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()*xb.size(0)
        epoch_loss /= max(1,len(cal_loader_class0.dataset))
        train_losses.append(epoch_loss)

        vae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb in val_loader_class0:
                xb = xb[0].to(device)
                val_loss += beta_vae_cosine_loss(xb, vae(xb)[0], vae(xb)[1], vae(xb)[2])[0].item()*xb.size(0)
        val_loss /= max(1,len(val_loader_class0.dataset))
        val_losses.append(val_loss)
        val_metrics.append(val_loss)

        if (epoch+1)%50==0 or epoch==0:
            print(f"Epoch {epoch+1}/{param['EPOCH']} | Train: {epoch_loss:.6f} | Val: {val_loss:.6f}")
        
        # ---- Save best model automatically
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_model(vae, Path(base_path), "VAE_class0_best.pth")

    # ---------------------------
    # Train SIMCA thresholds
    save_metrics({"train_losses":train_losses,"val_losses":val_losses}, Path(base_path),"losses.json")
    
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
    
    vae_best.load_state_dict(torch.load(Path(base_path)/"VAE_class0_best.pth", map_location=device))
    vae_best.eval()
    
    simca = VAESIMCA(vae_best, device=device,type='dd', t2lim='Fdist', t2cl=0.95, qlim='jm', qcl=0.95, dcl=0.95, verbose=False)
    simca.fit_thresholds(cal_loader_class0)

    # ---------------------------
    # Benchmark on test using SIMCA
    pred_class0, T2_test, Q_test = simca.predict(test_loader)
    # pred_class0, T2_test, Q_test,score = simca.predict_balanced(test_loader)
    

        # --- Build predicted labels (conform=0, unconform=1)
    pred_labels = np.where(pred_class0, 0, 1)

    # --- True labels (flattened)
    labels_true_list = []
    for _, yb in test_loader:
        labels_true_list.append(np.argmax(yb.numpy(), axis=1))
    labels_true = np.concatenate(labels_true_list)

    unique_true = np.unique(labels_true)
    n_true = len(unique_true)
    conf_mat_full = np.zeros((2, n_true), dtype=int)

    for i, pred in enumerate([0, 1]):  # 0=conform, 1=unconform
        for j, true_class in enumerate(unique_true):
            conf_mat_full[i, j] = np.sum((pred_labels == pred) & (labels_true == true_class))

    # Plot as before
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(conf_mat_full, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=[f"class{c}" for c in unique_true], yticklabels=["conform","unconform"], ax=ax)
    ax.set_xlabel("True class")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Confusion Matrix  {param_id}")
    plt.tight_layout()
    manual_path = os.path.join(base_path, "confusion_matrix_anomaly.pdf")
    fig.savefig(manual_path)
    plt.close()


    # --- Metrics
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
    'best_epoch': int(best_epoch)
    }
    
    all_metrics.append(metrics_dict)
    with open(os.path.join(base_path,'metrics.txt'),'w') as f:
        for k,v in metrics_dict.items():
            f.write(f"{k}: {v}\n")

# ---------------------------
# Save all params & metrics summary
summary_dir = os.path.join(root,process_id)
with open(os.path.join(summary_dir,"all_params.json"),"w") as f:
    json.dump(all_params,f,indent=2)
with open(os.path.join(summary_dir,"all_metrics.json"),"w") as f:
    json.dump(all_metrics,f,indent=2)

print("Sweep finished. Summaries saved to:", summary_dir)

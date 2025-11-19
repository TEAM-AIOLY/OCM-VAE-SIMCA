import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
from utils.alt_simca import ALTSIMCA  
import seaborn as sns
from scipy.signal import savgol_filter

data_path = "C:/00_aioly/GitHub/Deep-chemometrics/data/dataset/Ale/IR_ML.mat"
base_path = "C:/00_aioly/GitHub/Deep-chemometrics/Ale/cheese/figures"
os.makedirs(base_path, exist_ok=True)
data = sp.io.loadmat(data_path)

data_dict = {k: v for k, v in data.items() if not k.startswith('_')}

# Load training and test data
Xtr = data_dict['Xtr']
Xts = data_dict['Xts']
Xtr_dict = {key: Xtr[0][0][i] for i, key in enumerate(Xtr.dtype.names)}
Xts_dict = {key: Xts[0][0][i] for i, key in enumerate(Xts.dtype.names)}

Xtr_data = Xtr_dict['data']
Xts_data = Xts_dict['data']
Xtr_label = np.squeeze(Xtr_dict['class'][0][0]).astype(int) - 1
Xts_label = np.squeeze(Xts_dict['class'][0][0]).astype(int) - 1

# Determine number of classes and one-hot encode
n_classes = len(np.unique(np.concatenate([Xtr_label, Xts_label])))
ytr = np.eye(n_classes, dtype=np.float32)[Xtr_label]
yts = np.eye(n_classes, dtype=np.float32)[Xts_label]

# Spectral axis and preprocessing
wv = np.linspace(2500, 4000, Xtr_data.shape[1])
w = 15
d = 1
pol = 2

Xtr_data = savgol_filter(Xtr_data, window_length=w, polyorder=pol, deriv=d, axis=1)
Xts_data = savgol_filter(Xts_data, window_length=w, polyorder=pol, deriv=d, axis=1)

# Fit ALT-SIMCA model on class 0
n_pc = 10
simca = ALTSIMCA(n_components=n_pc, alpha=0.95)
simca.fit(Xtr_data, ytr, target_class=0)

# Predict on test data
conf_mat, metrics, conform = simca.predict(Xts_data, yts)

# Plot confusion matrix
unique_true = np.arange(conf_mat.shape[1])
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=[f"class{c}" for c in unique_true],
            yticklabels=["conform", "unconform"], ax=ax)
ax.set_xlabel("True class")
ax.set_ylabel("Predicted (conform/unconform)")
ax.set_title("ALT-SIMCA Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(base_path, 'cheese_altsimca_confmat.pdf'))

# Metrics
sensitivity = metrics['recall']
print("Sensitivity (recall) for class 0:", sensitivity)
specificity = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[0, 1])
print("Specificity for class 0:", specificity)

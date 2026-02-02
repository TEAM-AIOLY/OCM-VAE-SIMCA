import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
from utils import SIMCA
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

# # Determine number of classes and one-hot encode
# n_classes = len(np.unique(np.concatenate([Xtr_label, Xts_label])))
# ytr = np.eye(n_classes, dtype=np.float32)[Xtr_label]
# yts = np.eye(n_classes, dtype=np.float32)[Xts_label]

# Spectral axis and preprocessing
wv = np.linspace(2500, 4000, Xtr_data.shape[1])
w = 15
d = 1
pol = 2

Xtr_data = savgol_filter(Xtr_data, window_length=w, polyorder=pol, deriv=d, axis=1)
Xts_data = savgol_filter(Xts_data, window_length=w, polyorder=pol, deriv=d, axis=1)

model = SIMCA(n_components=10, model_class=
0 ,type='alt',t2lim='Fdist',qlim='jm').fit(Xtr_data,Xtr_label)

y_pred = model.predict(Xts_data, y_true=Xts_label)


y_pred = np.ravel(y_pred).astype(int)
y_true = np.ravel(Xts_label).astype(int)

# --- True labels (flattened)
unique_true = np.unique(Xts_label)
n_true = len(unique_true)
print(unique_true)
conf_mat_full = np.zeros((2, n_true), dtype=int)

for i, pred in enumerate([1, 0]):  
        for j, true_class in enumerate(unique_true):
            conf_mat_full[i, j] = np.sum((y_pred == pred) & (y_true == true_class))

print( conf_mat_full)

# # Plot as before
fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(conf_mat_full, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=[f"class{c}" for c in unique_true], yticklabels=["conform","unconform"], ax=ax)
ax.set_xlabel("True class")
ax.set_ylabel("Predicted class")
plt.tight_layout()
plt.savefig(os.path.join(base_path,"Alt-Simac_cm.pdf"))
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



T2,T2red,Q,Qred = model.transform(Xtr_data)


plt.rcParams.update({
    'font.size': 18,          # global font size
    'axes.labelsize': 18,     # x and y labels
    'xtick.labelsize': 18,    # x tick labels
    'ytick.labelsize': 18,    # y tick labels
    'legend.fontsize': 18,    # legend text
})


cmap=plt.get_cmap('tab10')
for i, cls in enumerate(model._model):
    T2, T2red, Q, Qred = model.transform(Xts_data) 
    Dlim = model._model[cls]['D_limit']
    a = np.arange(0, Dlim + 0.0001, 0.0001)
    curve = np.sqrt(np.maximum(Dlim**2 - a**2, 0))

    plt.figure(figsize=(8, 8))  # taller figure

    # scatter for each class
    scatter_handles = []
    for j in range(len(np.unique(Xts_label))):
        sc = plt.scatter(
            T2red[Xts_label == j],
            Qred[Xts_label == j],
            s=40,
            edgecolor='k',
            linewidth=0.5,
            alpha=0.7,
            color=cmap(j)
        )
        scatter_handles.append(sc)

    # confine line
    line_handle, = plt.plot(a, curve, 'b-', lw=2)

    # two-row legend
    class_labels = [f'Class {j}' for j in range(len(np.unique(Xts_label)))]
    line_labels = [f'Confine classe {cls}']

    first_legend = plt.legend(
        scatter_handles, class_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.10),
        ncol=len(class_labels),
        frameon=False
    )
    plt.gca().add_artist(first_legend)

    plt.legend(
        [line_handle], line_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.16),
        frameon=False
    )

    plt.xlabel(r"$T^2_{red}$")
    plt.ylabel(r"$Q_{red}$")
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, f"Alt-Simac_TQ_{cls}.pdf"))
    plt.close()




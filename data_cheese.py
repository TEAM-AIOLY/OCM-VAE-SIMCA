import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler  
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.signal import savgol_filter


data_path="C:/00_aioly/GitHub/Deep-chemometrics/data/dataset/Ale/IR_ML.mat"
base_path = "C:/00_aioly/GitHub/Deep-chemometrics/Ale/cheese/figures"
os.makedirs(base_path, exist_ok=True)
data = sp.io.loadmat(data_path)

data_dict = {k: v for k, v in data.items() if not k.startswith('_')}

# Example: access Xtr and Xts
Xtr = data_dict['Xtr']
Xts = data_dict['Xts']

Xtr_obj = Xtr[0][0]
Xts_obj = Xts[0][0]

Xtr_dict = {key: Xtr[0][0][i] for i, key in enumerate(Xtr.dtype.names)}
Xts_dict = {key: Xts[0][0][i] for i, key in enumerate(Xts.dtype.names)}

Xts_data= Xts_dict['data']
Xtr_data= Xtr_dict['data']
Xts_label= Xts_dict['class'][0][0]
Xtr_label= Xtr_dict['class'][0][0]

Xtr_label = np.squeeze(Xtr_label).astype(int) - 1
Xts_label = np.squeeze(Xts_label).astype(int) - 1

# --- Determine number of classes ---
n_classes = len(np.unique(np.concatenate([Xtr_label, Xts_label])))

# --- One-hot encode directly ---
ytr = np.eye(n_classes, dtype=np.float32)[Xtr_label]
yts = np.eye(n_classes, dtype=np.float32)[Xts_label]

cmap = plt.get_cmap('tab10', 10)

wv = np.linspace(2500,4000,Xtr_data.shape[1])



# plt.figure()
# for i in range(n_classes):
#     plt.plot(wv, Xtr_data[Xtr_label==i][::20].T, color=cmap(i), alpha=0.5)

# # one legend entry per class (no other changes)
# custom_lines = [Line2D([0], [0], color=cmap(i), lw=3) for i in range(n_classes)]
# plt.legend(custom_lines, [f"Class {i+1}" for i in range(n_classes)], title="", frameon=False)

# plt.xlabel('Wavelength (nm)')    
# plt.ylabel('Absorbance (a.u.)')
# plt.tight_layout()
# plt.savefig(os.path.join(base_path, 'cheese_train_spectra.pdf'))
# plt.close()

# plt.figure()
# for i in range(n_classes):
#      plt.plot(wv,Xts_data[Xts_label==i][::20].T, color=cmap(i), alpha=0.5);
     
# plt.xlabel('Wavelength')    
# plt.ylabel('Absorbance')
# plt.tight_layout()
# plt.savefig(os.path.join(base_path, 'cheese_test_spectra.pdf'))


n_pc =10
pca = PCA(n_components=n_pc)
Xtr_std = StandardScaler().fit_transform(Xtr_data)
Xts_std = StandardScaler().fit_transform(Xts_data)
pca.fit(Xtr_std)
pca_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)


scores_tr = pca.transform(Xtr_std)
scores_ts = pca.transform(Xts_std)
explained_var = pca.explained_variance_ratio_

import matplotlib.pyplot as plt
import os


plt.rcParams.update({
    'font.size': 18,          # global font size
    'axes.labelsize': 18,     # x and y labels
    'xtick.labelsize': 18,    # x tick labels
    'ytick.labelsize': 18,    # y tick labels
    'legend.fontsize': 18,    # legend text
})

# --- First PCA scores plot ---
plt.figure(figsize=(8,6))
for i in range(n_classes):
    plt.scatter(
        scores_tr[Xtr_label == i, 0],
        scores_tr[Xtr_label == i, 1],
        alpha=0.6,
        label=f'Class {i}',
        color=cmap(i)
    )

plt.xlabel(f'PC1 ({explained_var[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({explained_var[1]*100:.1f}%)')
plt.grid()

plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.20),
    ncol=3,
    frameon=False,
    handletextpad=0.5,
    columnspacing=1.5
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(os.path.join(base_path, 'cheese_pca_scores.pdf'), bbox_inches='tight')
plt.close()

# --- Second PCA scores plot ---
plt.figure(figsize=(8,6))
for i in range(n_classes):
    plt.scatter(
        scores_tr[Xtr_label == i, 1],
        scores_tr[Xtr_label == i, 2],
        alpha=0.6,
        label=f'Class {i}',
        color=cmap(i)
    )

plt.xlabel(f'PC2 ({explained_var[1]*100:.1f}%)')
plt.ylabel(f'PC3 ({explained_var[2]*100:.1f}%)')
plt.grid()

plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.20),
    ncol=3,
    frameon=False,
    handletextpad=0.5,
    columnspacing=1.5
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(os.path.join(base_path, 'cheese_pca_scores_2.pdf'), bbox_inches='tight')
plt.close()



plt.figure(figsize=(8,6))
for k in range(5):
    plt.plot(wv, pca_loadings[:, k], label=f'PC{k+1} ({explained_var[k]*100:.1f}%)')
plt.xlabel('Wavelength')
plt.ylabel('Loading')   
plt.title('PCA Loadings')
plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,  
    frameon=False
)
plt.tight_layout()
plt.savefig(os.path.join(base_path, 'cheese_pca_loadings.pdf'))
plt.close()

# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111, projection='3d')
# for i in range(n_classes):
#     ax.scatter(scores_tr[Xtr_label==i,0], scores_tr[Xtr_label==i,1], scores_tr[Xtr_label==i,2],
#                alpha=0.5, label=f'Class {i}', color=cmap(i))
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
# ax.set_title('PCA Scores 3D (Training data)')
# ax.legend()
# plt.show()



n_folds = 5
max_components = 25  # or any upper limit you want to try
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

f1c_curve = []
f1cv_curve = []

for n_comp in range(1, max_components + 1):
    # --- Calibration F1 (train full set) ---
    pls = PLSRegression(n_components=n_comp)
    pls.fit(Xtr_data, Xtr_label)
    lda = LDA()
    X_scores = pls.transform(Xtr_data)
    lda.fit(X_scores, Xtr_label)
    y_pred = lda.predict(X_scores)
    f1c_curve.append(f1_score(Xtr_label, y_pred, average='macro'))

    # --- Cross-validation F1 ---
    f1_folds = []
    for train_idx, val_idx in skf.split(Xtr_data, Xtr_label):
        X_train, X_val = Xtr_data[train_idx], Xtr_data[val_idx]
        y_train, y_val = Xtr_label[train_idx], Xtr_label[val_idx]

        pls = PLSRegression(n_components=n_comp)
        pls.fit(X_train, y_train)
        lda = LDA()
        lda.fit(pls.transform(X_train), y_train)

        y_val_pred = lda.predict(pls.transform(X_val))
        f1_folds.append(f1_score(y_val, y_val_pred, average='macro'))

    f1cv_curve.append(np.mean(f1_folds))
    
    
offset = 0.005  # small vertical shift for visibility

plt.figure(figsize=(8,6))
plt.plot(range(1, max_components + 1), 
         np.array(f1c_curve) + offset, 
         marker='o', linewidth=2, label='F1 cal')
plt.plot(range(1, max_components + 1), 
         f1cv_curve, 
         marker='s', linewidth=2, label='F1 CV')
plt.xlabel('Number of Latent Variables')
plt.ylabel('F1 Score')
plt.grid(True, linestyle='--', alpha=0.6)

plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.20),
    ncol=2,
    frameon=False
)

plt.tight_layout()
plt.savefig(os.path.join(base_path, 'cheese_plsda_f1_curve.pdf'), bbox_inches='tight')
plt.close()





# Select best number of components
best_n_comp = np.argmax(f1cv_curve) + 1
print(f"Best number of PLS components: {best_n_comp}")

# Train final model with best_n_comp and evaluate on test set
pls = PLSRegression(n_components=best_n_comp)
pls.fit(Xtr_data, Xtr_label)
lda = LDA()
Xtr_scores = pls.transform(Xtr_data)
Xts_scores = pls.transform(Xts_data)

lda.fit(Xtr_scores, Xtr_label)
yts_pred = lda.predict(Xts_scores)

confmat_test = confusion_matrix(Xts_label, yts_pred)
print("Test Confusion Matrix:\n", confmat_test)

fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(confmat_test, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=[f"class{c+1}" for c in range(confmat_test.shape[0])],
            yticklabels=[f"class{c+1}" for c in range(confmat_test.shape[0])], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(base_path, 'cheese_plsda_confmat.pdf'))
plt.close()



pls_loadings = pls.x_loadings_  # shape (n_features, n_components)
wv = np.linspace(2500, 4000, pls_loadings.shape[0])

plt.figure(figsize=(10,8))
for i in range(5):
    plt.plot(wv, pls_loadings[:, i], label=f'LV{i+1}')
plt.xlabel('Wavelength')
plt.ylabel('magnitude')
plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.20),
    ncol=3,
    frameon=False
)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(os.path.join(base_path, 'cheese_pls_loadings.pdf'), bbox_inches='tight')
plt.close()




P = pls.x_loadings_  # shape (L, k)
W = lda.coef_        # shape (n_classes-1, k)
D = P @ W.T          # shape (L, n_classes-1)

var = np.var(pls.x_scores_, axis=0, ddof=1)
P_scaled = P * np.sqrt(var)  # scales by LV variance
D_var = P_scaled @ W.T
D_norm = D_var / np.linalg.norm(D_var, axis=0, keepdims=True)

plt.figure(figsize=(8,6))
for i in range(D_norm.shape[1]):
    plt.plot(wv, D_norm[:, i], label=f'DV {i+1}')
plt.xlabel('Wavelength')
plt.ylabel('Discriminant vectors')

plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.20),
    ncol=3,
    frameon=False
)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(os.path.join(base_path, 'cheese_lda_discriminant_vectors_norm.pdf'), bbox_inches='tight')
plt.close()

import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import SIMCA
import seaborn as sns
from scipy.signal import savgol_filter
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA

data_root ="C:\\00_aioly\\GitHub\\HyperNuts\\data\\Nuts data\\HSI Data"
data_folder ="SWIR camera (842–2532 nm)"
data_file ="nut_objects.json"
file_path = os.path.join(data_root, data_folder, data_file)
with open(file_path, 'r') as f:
    data = json.load(f)

# Extract unique nut types
nut_types = list(data.keys())
n_nut_types = len(nut_types)
print(f"Nut types found: {nut_types}")
print(f"Number of nut types: {n_nut_types}\n")

# Create a mapping of nut_type to label
nut_type_to_label = {nut_type: idx for idx, nut_type in enumerate(nut_types)}
print(f"Label mapping: {nut_type_to_label}\n")

# Prepare data: extract spectral data for each nut type
datasets = {}  # Will store calibration, validation, test sets

for nut_type in nut_types:
    print(f"Processing {nut_type}:")
    
    # Collect all spectral data for this nut type
    all_spectra = []
    for obj in data[nut_type]:
        spectral_data = np.array(obj['spectral_data'], dtype=np.float32)
        all_spectra.append(spectral_data)
    
    # Stack all spectra into one array (n_total_pixels, n_wavelengths)
    X = np.vstack(all_spectra)
    print(f"  Total spectra shape: {X.shape}")
    
    
    # Apply SNV (Standard Normal Variate) preprocessing

    X_mean = np.mean(X, axis=1, keepdims=True)
    X_std = np.std(X, axis=1, keepdims=True)
    X = (X - X_mean) / (X_std + 1e-8)  
    # Apply Savitzky-Golay filter
    X = savgol_filter(X, window_length=5, polyorder=2, deriv=1, axis=1)
    
    # Check for NaN values
    nan_mask = np.isnan(X).any(axis=1)
    if np.any(nan_mask):
        print(f"  WARNING: Found {np.sum(nan_mask)} samples with NaN values. Removing them.")
        X = X[~nan_mask]
        print(f"  Shape after removing NaNs: {X.shape}")
    
    # Check for inf values
    inf_mask = np.isinf(X).any(axis=1)
    if np.any(inf_mask):
        print(f"  WARNING: Found {np.sum(inf_mask)} samples with inf values. Removing them.")
        X = X[~inf_mask]
        print(f"  Shape after removing infs: {X.shape}")
    
    # Create labels for this nut type (use the nut type's ID)
    nut_label = nut_type_to_label[nut_type]
    y = np.full(X.shape[0], nut_label, dtype=int)
    
    # Split into calibration (70%), validation (15%), test (15%)
    X_cal, X_temp, y_cal, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"  Calibration shape: {X_cal.shape}, labels: {np.unique(y_cal)}")
    print(f"  Validation shape: {X_val.shape}, labels: {np.unique(y_val)}")
    print(f"  Test shape: {X_test.shape}, labels: {np.unique(y_test)}\n")
    
    datasets[nut_type] = {
        'calibration': X_cal,
        'validation': X_val,
        'test': X_test,
        'calibration_labels': y_cal,
        'validation_labels': y_val,
        'test_labels': y_test,
        'n_objects': len(data[nut_type]),
        'n_wavelengths': X.shape[1]
    }
    
# ===========================
# SIMCA Analysis - One-class per nut type
# ===========================
nb_comp=12
# Create output directory for SIMCA results
simca_output_root = os.path.join(data_root, data_folder, "simca_analysis")
os.makedirs(simca_output_root, exist_ok=True)

print(f"\n{'='*60}")
print("SIMCA One-class Analysis")
print(f"{'='*60}\n")

# Store results
all_metrics = []

# Iterate through each nut type for one-class SIMCA
for target_nut_idx, target_nut in enumerate(nut_types):
    print(f"\n{'='*60}")
    print(f"Processing {target_nut} (class {nut_type_to_label[target_nut]})")
    print(f"{'='*60}")
    
    # Create target-specific directory
    target_path = os.path.join(simca_output_root, target_nut)
    os.makedirs(target_path, exist_ok=True)
    
    # Prepare calibration data (in-class)
    Xtr_data = datasets[target_nut]['calibration'].copy()
    Xtr_label = datasets[target_nut]['calibration_labels'].copy()
    
    # Relabel calibration data to 0 (in-class) for SIMCA training
    Xtr_label[:] = 0
    
    # Remove outliers from calibration data using PCA score space
    print(f"Calibration data shape before outlier removal: {Xtr_data.shape}")
    
   
    pca_temp = PCA(n_components=nb_comp)
    T_cal = pca_temp.fit_transform(Xtr_data)  # PCA scores
    
    # Compute mean and covariance in PCA score space
    mean_scores = np.mean(T_cal, axis=0)
    cov_scores = np.cov(T_cal, rowvar=False)
    cov_scores_inv = np.linalg.pinv(cov_scores)  # Use pseudo-inverse for stability
    
    # Compute Mahalanobis distance for each sample in PCA space
    mahal_dists = []
    for i in range(T_cal.shape[0]):
        diff = T_cal[i] - mean_scores
        md = np.sqrt(diff @ cov_scores_inv @ diff.T)
        mahal_dists.append(md)
    mahal_dists = np.array(mahal_dists)
    
    # Remove samples with Mahalanobis distance > 95th percentile
    outlier_threshold = np.percentile(mahal_dists, 95)
    outlier_mask = mahal_dists <= outlier_threshold
    Xtr_data = Xtr_data[outlier_mask]
    Xtr_label = Xtr_label[outlier_mask]
    
    n_outliers = np.sum(~outlier_mask)
    print(f"Removed {n_outliers} outliers (threshold: {outlier_threshold:.2f})")
    print(f"Calibration data shape after outlier removal: {Xtr_data.shape}")
    print(f"Calibration labels: {np.unique(Xtr_label)}")
    
    # Check for NaN/inf in calibration data
    if np.any(np.isnan(Xtr_data)) or np.any(np.isinf(Xtr_data)):
        nan_count = np.sum(np.isnan(Xtr_data))
        inf_count = np.sum(np.isinf(Xtr_data))
        print(f"ERROR: Calibration data contains {nan_count} NaNs and {inf_count} infs")
        continue
    
    # Prepare test data (all nut types)
    Xts_data_list = []
    Xts_label_list = []
    
    for nut_type in nut_types:
        test_data = datasets[nut_type]['test'].copy()
        test_labels = datasets[nut_type]['test_labels'].copy()
        
        # If it's the target nut, keep label as 0 (in-class)
        # Otherwise, set label to 1 (out-of-class/anomaly)
        if nut_type == target_nut:
            test_labels[:] = 0
        else:
            test_labels[:] = 1
        
        Xts_data_list.append(test_data)
        Xts_label_list.append(test_labels)
    
    Xts_data = np.vstack(Xts_data_list)
    Xts_label = np.concatenate(Xts_label_list)
    
    print(f"Test data shape: {Xts_data.shape}")
    print(f"Test labels distribution: {np.bincount(Xts_label)}")
    
    # Fit SIMCA model
    model = SIMCA(n_components=nb_comp, model_class=0, type='alt', t2lim='Fdist', qlim='jm').fit(Xtr_data, Xtr_label)
    
    # Predict on test set
    y_pred = model.predict(Xts_data, y_true=Xts_label)
    y_pred = np.ravel(y_pred).astype(int)
    y_true = np.ravel(Xts_label).astype(int)
    
    # Confusion matrix (binary: conform/unconform)
    unique_true = np.unique(Xts_label)
    n_true = len(unique_true)
    conf_mat_full = np.zeros((2, n_true), dtype=int)
    
    for i, pred in enumerate([1, 0]):
        for j, true_class in enumerate(unique_true):
            conf_mat_full[i, j] = np.sum((y_pred == pred) & (y_true == true_class))
    
    print(f"\nConfusion matrix:\n{conf_mat_full}")
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_mat_full, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=[target_nut, "Others"],
                yticklabels=["conform", "unconform"], ax=ax)
    ax.set_xlabel("True class")
    ax.set_ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(target_path, f"confusion_matrix.pdf"))
    plt.close()
    
    # Metrics
    TP = conf_mat_full[0, 0]
    FN = conf_mat_full[1, 0]
    FP = conf_mat_full[0, 1:].sum()
    TN = conf_mat_full[1, 1:].sum()
    
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-12)
    precision = TP / (TP + FP + 1e-12)
    recall = TP / (TP + FN + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    
    fa_rates = conf_mat_full[0, 1:] / (conf_mat_full[:, 1:].sum(axis=0) + 1e-12)
    fa_mean = np.mean(fa_rates)
    
    print(f"\nMetrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  Mean FA rate: {fa_mean:.4f}")
    
    metrics_dict = {
        'nut_type': target_nut,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'mean_false_acceptance': float(fa_mean),
        'TP': int(TP),
        'FN': int(FN),
        'FP': int(FP),
        'TN': int(TN)
    }
    all_metrics.append(metrics_dict)
    
    # T-Q plots
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
    })
    
    cmap = plt.get_cmap('tab10')
    
    for i, cls in enumerate(model._model):
        T2, T2red, Q, Qred = model.transform(Xts_data)
        Dlim = model._model[cls]['D_limit']
        a = np.arange(0, Dlim + 0.0001, 0.0001)
        curve = np.sqrt(np.maximum(Dlim**2 - a**2, 0))
        
        plt.figure(figsize=(8, 8))
        
        # Scatter plot for in-class vs out-of-class
        colors = ['green', 'red']
        labels_plot = ['In-class', 'Out-of-class']
        
        for j, true_class in enumerate(unique_true):
            plt.scatter(
                T2red[Xts_label == true_class],
                Qred[Xts_label == true_class],
                s=40,
                edgecolor='k',
                linewidth=0.5,
                alpha=0.7,
                color=colors[j],
                label=labels_plot[j]
            )
        
        # Confine line
        plt.plot(a, curve, 'b-', lw=2, label=f'Confine {target_nut}')
        
        plt.xlabel(r"$T^2_{red}$")
        plt.ylabel(r"$Q_{red}$")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3, which='both')
        
        # Apply log scale to both axes
        plt.xscale('log')
        plt.yscale('log')
        
        # Set limits with small offset for log scale
        plt.xlim(left=1e-2)
        plt.ylim(bottom=1e-2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(target_path, f"T2_Q_plot.pdf"))
        plt.close()
    
    print(f"Plots saved to: {target_path}")

# Save summary metrics
with open(os.path.join(simca_output_root, "all_metrics.json"), "w") as f:
    json.dump(all_metrics, f, indent=2)

print(f"\n{'='*60}")
print("SIMCA analysis completed!")
print(f"Results saved to: {simca_output_root}")
print(f"{'='*60}")



import numpy as np
import matplotlib.pyplot as plt
import os
import json
from utils import SIMCA
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split

# Import datasets from simca_nuts
import sys
sys.path.insert(0, 'c:\\00_aioly\\GitHub\\OCM-VAE-SIMCA')
from simca_nuts import datasets, nut_types, nut_type_to_label

# Create output directory
data_root = "C:\\00_aioly\\GitHub\\HyperNuts\\data\\Nuts data\\HSI Data"
data_folder = "SWIR camera (842–2532 nm)"
base_path = os.path.join(data_root, data_folder, "simca_analysis")
os.makedirs(base_path, exist_ok=True)

print(f"Output directory: {base_path}\n")

# Store results
all_metrics = []

# Iterate through each nut type for one-class SIMCA
for target_nut_idx, target_nut in enumerate(nut_types):
    print(f"\n{'='*60}")
    print(f"Processing {target_nut} (class {nut_type_to_label[target_nut]})")
    print(f"{'='*60}")
    
    # Create target-specific directory
    target_path = os.path.join(base_path, target_nut)
    os.makedirs(target_path, exist_ok=True)
    
    # Prepare calibration data (in-class)
    Xtr_data = datasets[target_nut]['calibration'].copy()
    Xtr_label = datasets[target_nut]['calibration_labels'].copy()
    
    print(f"Calibration data shape: {Xtr_data.shape}")
    print(f"Calibration labels: {np.unique(Xtr_label)}")
    
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
    model = SIMCA(n_components=10, model_class=0, type='alt', t2lim='Fdist', qlim='jm').fit(Xtr_data, Xtr_label)
    
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
                xticklabels=[f"In-class" if c == 0 else f"{nut_types[c-1]}" for c in unique_true],
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
        scatter_handles = []
        colors = ['green', 'red']
        labels_plot = ['In-class', 'Out-of-class']
        
        for j, true_class in enumerate(unique_true):
            sc = plt.scatter(
                T2red[Xts_label == true_class],
                Qred[Xts_label == true_class],
                s=40,
                edgecolor='k',
                linewidth=0.5,
                alpha=0.7,
                color=colors[j],
                label=labels_plot[j]
            )
            scatter_handles.append(sc)
        
        # Confine line
        line_handle, = plt.plot(a, curve, 'b-', lw=2, label=f'Confine {target_nut}')
        
        plt.xlabel(r"$T^2_{red}$")
        plt.ylabel(r"$Q_{red}$")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(os.path.join(target_path, f"T2_Q_plot.pdf"))
        plt.close()
    
    print(f"Plots saved to: {target_path}")

# Save summary metrics
with open(os.path.join(base_path, "all_metrics.json"), "w") as f:
    json.dump(all_metrics, f, indent=2)

print(f"\n{'='*60}")
print("SIMCA analysis completed!")
print(f"Results saved to: {base_path}")
print(f"{'='*60}")

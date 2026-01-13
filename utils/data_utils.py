"""Data utility helpers for OCM-VAE-SIMCA

Contains object_aware_splits() which performs object-aware dataset splitting
(i.e., ensures spectra from the same object do not appear in multiple splits).
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter


def object_aware_splits(data, nut_types, target_nut, n_wavelengths,
                        cal_frac=0.7, val_frac=0.15, test_frac=0.15,
                        random_state=42, outlier_percentile=95, use_pca=True):
    """Split dataset by objects so that spectra from the same object do not cross splits.

    Parameters
    - data: dict mapping nut_type -> list of entries where each entry has 'spectral_data' (ndarray)
    - nut_types: list of nut type keys (order preserved)
    - target_nut: which nut is the target class
    - n_wavelengths: number of wavelength channels (used for empty arrays)

    Returns
    - splits: dict mapping nut_type to {'cal','val','test'} arrays
    - Xts_data: concatenated global test set (all nuts)
    - Xts_label: labels for Xts_data (0=target,1=others)
    - X_cal, X_val, X_test_in: calibration/validation/test sets for target nut
    - X_test_out: concatenated test set for "other" nuts
    """
    assert abs(cal_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1.0"

    splits = {}
    for nut_type in nut_types:
        objs = data[nut_type]
        obj_spectra = [np.asarray(obj['spectral_data'], dtype=np.float32) for obj in objs]

        if len(obj_spectra) == 0:
            print(f"  {nut_type}: no objects found, skipping")
            splits[nut_type] = {'cal': np.empty((0, n_wavelengths), dtype=np.float32),
                                'val': np.empty((0, n_wavelengths), dtype=np.float32),
                                'test': np.empty((0, n_wavelengths), dtype=np.float32)}
            continue

        obj_lengths = [s.shape[0] for s in obj_spectra]
        X_nut = np.vstack(obj_spectra)
        obj_ids = np.concatenate([np.full(l, idx, dtype=int) for idx, l in enumerate(obj_lengths)])

        # Remove NaN/inf samples
        bad_mask_nut = np.isnan(X_nut).any(axis=1) | np.isinf(X_nut).any(axis=1)
        if np.any(bad_mask_nut):
            print(f"  WARNING: {nut_type}: Found {np.sum(bad_mask_nut)} NaN/inf samples. Removing them.")
            keep_mask = ~bad_mask_nut
            X_nut = X_nut[keep_mask]
            obj_ids = obj_ids[keep_mask]

        # Preprocessing copy for outlier detection (SNV + Savitzky-Golay)
        X_proc = (X_nut - np.mean(X_nut, axis=1, keepdims=True)) / (np.std(X_nut, axis=1, keepdims=True) + 1e-8)
        try:
            X_proc = savgol_filter(X_proc, window_length=5, polyorder=2, deriv=1, axis=1)
        except Exception:
            pass

        # Outlier detection (PCA score-space) on X_proc -> remove pixel-level outliers then re-group by object
        X_clean = X_nut.copy()
        if use_pca and X_proc.shape[0] > 3 and X_proc.shape[0] > 1:
            n_comp = min(10, X_proc.shape[1], max(1, X_proc.shape[0]-1))
            if X_proc.shape[0] > n_comp:
                pca_tmp = PCA(n_components=n_comp)
                T = pca_tmp.fit_transform(X_proc)
                mean_scores = np.mean(T, axis=0)
                cov_scores = np.cov(T, rowvar=False)
                cov_inv = np.linalg.pinv(cov_scores)
                mahal = np.array([np.sqrt((t - mean_scores) @ cov_inv @ (t - mean_scores).T) for t in T])
                out_thr = np.percentile(mahal, outlier_percentile)
                mask = mahal <= out_thr
                n_removed = np.sum(~mask)
                if n_removed > 0:
                    print(f"  {nut_type}: removed {n_removed} outliers (threshold {out_thr:.3f})")
                X_clean = X_nut[mask]
                obj_ids_clean = obj_ids[mask]
            else:
                print(f"  {nut_type}: not enough samples for PCA components, skipping outlier removal")
                obj_ids_clean = obj_ids
        else:
            obj_ids_clean = obj_ids

        # Group remaining samples by object
        objects_after = {}
        for idx in np.unique(obj_ids_clean):
            rows_mask = obj_ids_clean == idx
            objspectra = X_clean[rows_mask]
            if objspectra.shape[0] > 0:
                objects_after[int(idx)] = objspectra

        n_objects_after = len(objects_after)
        if n_objects_after == 0:
            print(f"  {nut_type}: no objects remaining after cleaning, skipping")
            splits[nut_type] = {'cal': np.empty((0,n_wavelengths),dtype=np.float32),
                                'val': np.empty((0,n_wavelengths),dtype=np.float32),
                                'test': np.empty((0,n_wavelengths),dtype=np.float32)}
            continue

        # Split objects (ensure entire objects go to only one split)
        obj_idxs = list(objects_after.keys())
        if len(obj_idxs) >= 3:
            temp_size = 1.0 - cal_frac
            cal_objs, temp_objs = train_test_split(obj_idxs, test_size=temp_size, random_state=random_state)
            # split temp into val/test in proportion
            if (val_frac + test_frac) > 0:
                test_frac_rel = test_frac / (val_frac + test_frac)
            else:
                test_frac_rel = 0.5
            val_objs, test_objs = train_test_split(temp_objs, test_size=test_frac_rel, random_state=random_state)
        elif len(obj_idxs) == 2:
            cal_objs = [obj_idxs[0]]
            val_objs = []
            test_objs = [obj_idxs[1]]
        else:
            cal_objs = [obj_idxs[0]]
            val_objs = []
            test_objs = []

        def concat_objs(obj_list):
            if len(obj_list) == 0:
                return np.empty((0, n_wavelengths), dtype=np.float32)
            return np.vstack([objects_after[i] for i in obj_list])

        X_cal_nut = concat_objs(cal_objs)
        X_val_nut = concat_objs(val_objs)
        X_test_nut = concat_objs(test_objs)

        splits[nut_type] = {'cal': X_cal_nut, 'val': X_val_nut, 'test': X_test_nut}

        print(f"  {nut_type}: objects after cleaning={n_objects_after}, raw samples after cleaning={X_clean.shape[0]} -> cal={X_cal_nut.shape}, val={X_val_nut.shape}, test={X_test_nut.shape}")

    # Build global test set (concatenate all per-nut test parts)
    Xts_data_list = []
    Xts_label_list = []
    for nut_type in nut_types:
        X_test_nut = splits[nut_type]['test']
        if X_test_nut.shape[0] == 0:
            continue
        test_labels = np.zeros(X_test_nut.shape[0], dtype=int)
        if nut_type != target_nut:
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
    X_cal = splits[target_nut]['cal']
    X_val = splits[target_nut]['val']
    X_test_in = splits[target_nut]['test']

    # Explicitly provide a concatenated 'other nuts' test set (X_test_out)
    other_tests = [splits[n]['test'] for n in nut_types if n != target_nut and splits[n]['test'].shape[0] > 0]
    if len(other_tests) == 0:
        X_test_out = np.empty((0, n_wavelengths), dtype=np.float32)
    else:
        X_test_out = np.vstack(other_tests)

    return splits, Xts_data, Xts_label, X_cal, X_val, X_test_in, X_test_out
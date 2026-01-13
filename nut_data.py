import os
import numpy as np
import scipy.io as sio
from scipy import ndimage   
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import random

data_root ="C:\\00_aioly\\GitHub\\HyperNuts\\data\\Nuts data\\HSI Data"
data_folder ="SWIR camera (842–2532 nm)"
data_file ="SWIR_sb.mat"

data_path = os.path.join(data_root, data_folder, data_file)
data = sio.loadmat(data_path)

nut_classes = ["almond", "walnut", "hazelnut", "peanut"]

# Read all fields in data
print("Fields in data:")
for key, value in data.items():
    if not key.startswith('__'):
        print(f"  {key}: shape={np.array(value).shape}, dtype={np.array(value).dtype}")

# Extract objects from hyperspectral images
objects_dict = {nut_type: [] for nut_type in nut_classes}

h5_path = os.path.join(os.path.dirname(data_path), 'nut_objects.h5')
# parameters
BACKGROUND_THRESHOLD = 1e-6
# Plotting options
SAVE_PLOTS = True
MAX_PLOTS_PER_NUT = 1
FIG_DPI = 900
fig_dir = os.path.join(os.path.dirname(data_path), 'figures')
os.makedirs(fig_dir, exist_ok=True)

# HDF5 file open for writing
with h5py.File(h5_path, 'w') as h5f:
    extracted_counts = {nut_type: 0 for nut_type in nut_classes}
    sample_counts = {nut_type: 0 for nut_type in nut_classes}
    img_idx_tracker = {nut_type: 0 for nut_type in nut_classes} 

    for field_name, hsi_image in data.items():
        if field_name.startswith('__'):
            continue
        hsi_image = np.array(hsi_image, dtype=np.float32)

        # Identify nut type from field name
        nut_type = None
        for nc in nut_classes:
            if nc.lower() in field_name.lower():
                nut_type = nc
                break
        if nut_type is None:
            print(f"Warning: Could not identify nut type for field '{field_name}'. Skipping.")
            continue

        img_idx = img_idx_tracker[nut_type]
        print(f"\nProcessing {field_name} (nut type: {nut_type}, img_idx: {img_idx})")
        print(f"  HSI shape: {hsi_image.shape} (height x width x wavelength)")

        # Background mask (mean intensity across wavelengths)
        background_mask = np.mean(hsi_image, axis=2) < BACKGROUND_THRESHOLD
        foreground_mask = ~background_mask

        # Connected components
        structure = np.ones((3, 3), dtype=int)
        labeled_array, num_objects = ndimage.label(foreground_mask, structure=structure)
        print(f"  Found {num_objects} objects")

        # Optional: save one segmentation figure and one random-object extraction per image
        if SAVE_PLOTS and img_idx < MAX_PLOTS_PER_NUT and num_objects > 0:
            # RGB composite for visualization (pick 3 wavelengths)
            L = hsi_image.shape[2]
            idxs = [int(L*0.1), int(L*0.5), int(L*0.9)]
            rgb = np.stack([hsi_image[..., i] for i in idxs], axis=-1)
            p1, p99 = np.percentile(rgb, (1, 99))
            rgb_vis = np.clip((rgb - p1) / (p99 - p1 + 1e-12), 0, 1)

            # Labelled segmentation image (background black, each object a different color)
            seg_color = np.zeros_like(rgb_vis)
            cmap = cm.get_cmap('tab20')
            slices = ndimage.find_objects(labeled_array)
            for obj_id in range(1, num_objects + 1):
                mask = (labeled_array == obj_id)
                color = cmap((obj_id - 1) % 20)[:3]
                seg_color[mask] = color
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(seg_color)
            ax.axis('off')
            # add labels inside each component
            for obj_id in range(1, num_objects + 1):
                mask = (labeled_array == obj_id)
                coords = np.argwhere(mask)
                if coords.size == 0:
                    continue
                y, x = coords.mean(axis=0)
                ax.text(x, y, str(obj_id), color='white', fontsize=8, ha='center', va='center')
            seg_fig_path = os.path.join(fig_dir, f"seg_labels_{nut_type}_img{img_idx}.png")
            plt.tight_layout()
            plt.savefig(seg_fig_path, dpi=FIG_DPI)
            plt.close(fig)
            print(f"  Saved segmentation labels figure: {seg_fig_path}")

            # Random object extraction visualization (show extracted nut in RGB composited image)
            rand_obj = random.randint(1, num_objects)
            mask = (labeled_array == rand_obj)
            rgb_masked = np.zeros_like(rgb_vis)
            if mask.any():
                rgb_masked[mask] = rgb_vis[mask]
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(rgb_masked)
            ax.axis('off')
            # draw bbox
            sl = slices[rand_obj - 1]
            if sl is not None:
                y0, x0 = sl[0].start, sl[1].start
                h = sl[0].stop - sl[0].start
                w = sl[1].stop - sl[1].start
                rect = Rectangle((x0, y0), w, h, linewidth=1, edgecolor='yellow', facecolor='none')
                ax.add_patch(rect)
                ax.text(x0, y0, str(rand_obj), color='yellow', fontsize=8)
            ext_fig_path = os.path.join(fig_dir, f"extract_{nut_type}_img{img_idx}_obj{rand_obj}.png")
            plt.tight_layout()
            plt.savefig(ext_fig_path, dpi=FIG_DPI)
            plt.close(fig)
            print(f"  Saved extraction figure: {ext_fig_path}")

        # Iterate each connected component (object) and store spectra as 2D arrays (n_pixels x n_wavelengths)
        for obj_idx in range(1, num_objects + 1):
            obj_mask = (labeled_array == obj_idx)
            n_pixels = np.sum(obj_mask)
            if n_pixels == 0:
                continue

            # spectral data (raw) for the object
            spectral_data = hsi_image[obj_mask]  # (n_pixels, n_wavelengths)
            mean_spectrum = spectral_data.mean(axis=0)

            # centroid in image coordinates
            coords = np.argwhere(obj_mask)
            centroid = tuple(np.mean(coords, axis=0).tolist())

            # HDF5 path
            group_path = f"{nut_type}/img_{img_idx}/obj_{obj_idx}"
            grp = h5f.require_group(group_path)
            # Prepare spectra as contiguous float32 2D array and validate
            try:
                spec = np.asarray(spectral_data)
                # If object-dtype (ragged), attempt to stack rows
                if spec.dtype == object:
                    spec = np.vstack([np.asarray(r, dtype=np.float32) for r in spectral_data])
                spec = np.ascontiguousarray(spec, dtype=np.float32)

                if spec.ndim != 2 or spec.size == 0:
                    raise ValueError(f"Invalid spectra shape: {spec.shape}")

                # store spectra compressed (fallback to uncompressed if compression fails)
                if 'spectra' in grp:
                    del grp['spectra']
                try:
                    grp.create_dataset('spectra', data=spec, compression='gzip', compression_opts=4)
                except Exception as e:
                    print(f"  WARNING: failed to create compressed dataset for {group_path}: {e}; retrying without compression")
                    grp.create_dataset('spectra', data=spec)

                # always record the number of pixels and basic attrs
                grp.attrs['n_pixels'] = int(spec.shape[0])
                grp.attrs['centroid'] = centroid
                grp.attrs['img_idx'] = img_idx
                grp.attrs['nut_type'] = nut_type

                # also store the original dict fields as HDF5 metadata (replace JSON index)
                grp.attrs['obj_idx'] = int(obj_idx)
                grp.attrs['h5_path'] = group_path + '/spectra'
                # store mean_spectrum as an attribute if possible (small array);
                # fallback to a dataset if attr assignment fails
                try:
                    grp.attrs['mean_spectrum'] = np.asarray(mean_spectrum, dtype=np.float32)
                except Exception:
                    if 'mean_spectrum' in grp:
                        del grp['mean_spectrum']
                    grp.create_dataset('mean_spectrum', data=np.asarray(mean_spectrum, dtype=np.float32))

            except Exception as e:
                print(f"  ERROR: could not store spectra for {group_path}: {e}")
                import traceback
                traceback.print_exc()
                # skip this object
                continue

            # count this successful object write
            extracted_counts[nut_type] += 1
            # add spectral sample count (total pixels for this object)
            sample_counts[nut_type] += int(spec.shape[0])

        img_idx_tracker[nut_type] += 1
        print(f"  Extracted {extracted_counts[nut_type]} objects for {nut_type}")

# Print HDF5 file size
h5_size = os.path.getsize(h5_path) / (1024*1024)
print(f"\nSaved HDF5 data to: {h5_path} ({h5_size:.2f} MB)")

# Print per-nut spectral sample counts
total_samples = 0
for nt in nut_classes:
    cnt = int(sample_counts.get(nt, 0))
    print(f"Spectral samples for {nt}: {cnt}")
    total_samples += cnt
print(f"Total spectral samples across all nut types: {total_samples}")




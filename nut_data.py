import os
import numpy as np
import scipy.io as sio
from scipy import ndimage   





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
img_idx_tracker = {nut_type: 0 for nut_type in nut_classes}  # Track image index per nut type

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
    
    # Create background mask: all zero vectors (all wavelengths are 0)
    background_mask = np.all(hsi_image == 0, axis=2)
    foreground_mask = ~background_mask
    
    # Extract connected components
    labeled_array, num_objects = ndimage.label(foreground_mask)
    print(f"  Found {num_objects} objects")
    
    # Extract each object
    for obj_idx in range(1, num_objects + 1):
        obj_mask = labeled_array == obj_idx
        
        # Get pixel coordinates of object
        coords = np.argwhere(obj_mask)  # shape: (n_pixels, 2)
        
        # Compute centroid
        centroid = np.mean(coords, axis=0)
        
        # Extract spectral data: each pixel is a row, wavelengths are columns
        spectral_data = hsi_image[obj_mask]  # shape: (n_pixels, n_wavelengths)
        
        # Store in dictionary
        object_info = {
            'nut_type': nut_type,
            'img_idx': img_idx,
            'obj_idx': int(obj_idx),
            'centroid': centroid.tolist(),  # [row, col]
            'spectral_data': spectral_data,  # (n_pixels, n_wavelengths)
            'n_pixels': len(coords)
        }
        objects_dict[nut_type].append(object_info)
    
    # Increment image index for this nut type
    img_idx_tracker[nut_type] += 1
    
    print(f"  Extracted {len(objects_dict[nut_type])} objects for {nut_type}")

# Print summary
print("\n" + "="*60)
print("Summary of extracted objects:")
print("="*60)
for nut_type in nut_classes:
    print(f"{nut_type}: {len(objects_dict[nut_type])} objects")

# Save dictionary as JSON (convert numpy arrays to lists)
import json

objects_dict_json = {}
for nut_type, objects_list in objects_dict.items():
    objects_dict_json[nut_type] = []
    for obj in objects_list:
        obj_json = {
            'nut_type': obj['nut_type'],
            'img_idx': obj['img_idx'],
            'obj_idx': obj['obj_idx'],
            'centroid': obj['centroid'],
            'spectral_data': obj['spectral_data'].tolist(),  # Convert numpy array to list
            'n_pixels': obj['n_pixels']
        }
        objects_dict_json[nut_type].append(obj_json)

# Save to JSON file
output_path = os.path.join(os.path.dirname(data_path), 'nut_objects.json')
with open(output_path, 'w') as f:
    json.dump(objects_dict_json, f, indent=2)

print(f"\nSaved objects dictionary to: {output_path}")




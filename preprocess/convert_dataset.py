import os
import nibabel as nib
import numpy as np

# === НАСТРОЙКИ ===
DATASET_ROOT = r"D:\Dic\MONAI_dataset"
LABELS_DIR = os.path.join(DATASET_ROOT, "labels")

print("=== FIXING DATASET LABEL SHAPES ===")
print("Folder:", LABELS_DIR)
print("-----------------------------------")

def fix_mask(path):
    img = nib.load(path)
    data = img.get_fdata()
    original_shape = data.shape

    print(f"\n{path}  original shape = {original_shape}")

    # --- Удаляем оси размера 1 (squeeze) ---
    squeezed = np.squeeze(data)
    squeezed_shape = squeezed.shape

    if squeezed_shape != original_shape:
        print(f"  AFTER squeeze → {squeezed_shape}")

    # --- Если после squeeze стало 4D (например [X,Y,Z,2]) ---
    if len(squeezed_shape) == 4:
        print("  !!! WARNING: Found 4D mask → keeping only channel 0")
        squeezed = squeezed[..., 0]
        print(f"  NEW shape → {squeezed.shape}")

    # --- Если маска стала не 3D ---
    if len(squeezed.shape) != 3:
        print(f"  !!! ERROR: Mask shape still incorrect ({squeezed.shape}), skipping")
        return

    # --- Сохраняем обратно с тем же affine/header ---
    fixed_img = nib.Nifti1Image(squeezed.astype(np.uint16), img.affine, img.header)

    # Перезаписываем файл
    nib.save(fixed_img, path)
    print(f"  DONE → saved corrected file")



# === ОБРАБОТКА ВСЕХ ФАЙЛОВ ===
files = sorted([f for f in os.listdir(LABELS_DIR) if f.endswith(".nii.gz")])

for filename in files:
    fix_mask(os.path.join(LABELS_DIR, filename))

print("\n=== COMPLETED: Dataset masks normalized ===")

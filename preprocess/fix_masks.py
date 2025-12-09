import os
import nibabel as nib
import numpy as np

# Папка с масками
mask_dir = r"D:\Dic\MONAI_dataset\labels"

print("=== FIX MASKS SCRIPT STARTED ===")
print("Folder:", mask_dir)
print("--------------------------------")

# Получаем все файлы .nii.gz
files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".nii.gz")])

for fname in files:
    path = os.path.join(mask_dir, fname)

    img = nib.load(path)
    data = img.get_fdata()
    original_shape = data.shape

    print(f"\n{path} {original_shape}")

    # === 1. SQUEEZE убирает размерности = 1 ===
    squeezed = np.squeeze(data)
    squeezed_shape = squeezed.shape

    # Логируем изменение после squeeze
    if squeezed_shape != original_shape:
        print("AFTER SQUEEZE:", squeezed_shape)

    # === 2. Если остаётся 4D — берём первый канал ===
    if len(squeezed_shape) == 4:
        print("!!! WARNING: 4D mask → taking channel 0")
        squeezed = squeezed[..., 0]
        print("NEW SHAPE:", squeezed.shape)

    # === 3. Проверяем, что теперь 3D ===
    if len(squeezed.shape) != 3:
        print(f"!!! ERROR: Mask still not 3D ({squeezed.shape}). Skipping.")
        continue

    # === 4. Сохраняем обратно ===
    new_img = nib.Nifti1Image(squeezed.astype(np.uint16), img.affine, img.header)
    nib.save(new_img, path)

    print("DONE → saved:", path)

print("\n=== ALL MASKS FIXED ===")
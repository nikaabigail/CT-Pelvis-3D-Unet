import re
from pathlib import Path
import numpy as np
import SimpleITK as sitk

def stem_key(p: Path) -> str:
    s = p.name
    s = re.sub(r"\.nrrd$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(_LABEL.*)$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(_Seg.*)$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(-Pelvis-Thighs_S.*)$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(-Pelvis-Thighs_Segmentation.*)$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(-Pelvis-Thighs)$", "", s, flags=re.IGNORECASE)
    return s

def build_pairs_from_dirs(ct_dir: str, lab_dir: str):
    ct_dir = Path(ct_dir)
    lab_dir = Path(lab_dir)
    ct_files = sorted(ct_dir.glob("*.nrrd"))
    lab_files = sorted(lab_dir.glob("*.nrrd"))

    ct_map = {stem_key(p): p for p in ct_files}
    lab_map = {stem_key(p): p for p in lab_files}

    keys = sorted(set(ct_map.keys()) & set(lab_map.keys()))
    missing_ct = sorted(set(lab_map.keys()) - set(ct_map.keys()))
    missing_lab = sorted(set(ct_map.keys()) - set(lab_map.keys()))

    pairs = [{"case": k, "image": str(ct_map[k]), "label": str(lab_map[k])} for k in keys]
    return pairs, missing_ct, missing_lab

def ensure_scalar_zyx(arr: np.ndarray, src: str = "") -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 4:
        if a.shape[-1] <= 8:
            nonzeros = [int((a[..., i] != 0).sum()) for i in range(a.shape[-1])]
            best = int(np.argmax(nonzeros))
            if src:
                print(f"[WARN] Vector image '{src}', shape={a.shape}. Choose channel {best} {nonzeros}")
            a = a[..., best]
        else:
            if src:
                print(f"[WARN] 4D image '{src}', shape={a.shape}. Taking [0].")
            a = a[0]

    while a.ndim > 3:
        if a.shape[-1] <= 8:
            if src:
                print(f"[WARN] >3D image '{src}', shape={a.shape}. Taking [...,0].")
            a = a[..., 0]
        else:
            if src:
                print(f"[WARN] >3D image '{src}', shape={a.shape}. Taking [0].")
            a = a[0]

    if a.ndim != 3:
        raise RuntimeError(f"Cannot coerce to (Z,Y,X). Got shape={a.shape} for '{src}'")
    return a

def load_ct_zyx(path: str) -> np.ndarray:
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    return ensure_scalar_zyx(arr, src=path)

def load_lab_zyx(path: str) -> np.ndarray:
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    arr = ensure_scalar_zyx(arr, src=path)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return arr

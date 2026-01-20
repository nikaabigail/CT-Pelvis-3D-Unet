import numpy as np

def scale_intensity_to_01(ct_patch: np.ndarray, a_min: int, a_max: int) -> np.ndarray:
    x = ct_patch.astype(np.float32, copy=False)
    x = np.clip(x, float(a_min), float(a_max))
    x = (x - float(a_min)) / float(a_max - a_min)
    return x.astype(np.float32, copy=False)

def pad_to_roi_zyx(ct: np.ndarray, lab: np.ndarray, roi_zyx):
    Z, Y, X = ct.shape
    rz, ry, rx = roi_zyx
    pad_z = max(0, rz - Z)
    pad_y = max(0, ry - Y)
    pad_x = max(0, rx - X)
    if pad_z or pad_y or pad_x:
        ct = np.pad(ct, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant", constant_values=0)
        lab = np.pad(lab, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant", constant_values=0)
    return ct, lab

def crop_zyx(ct: np.ndarray, lab: np.ndarray, z0: int, y0: int, x0: int, roi_zyx):
    rz, ry, rx = roi_zyx
    return (
        ct[z0:z0 + rz, y0:y0 + ry, x0:x0 + rx],
        lab[z0:z0 + rz, y0:y0 + ry, x0:x0 + rx],
    )

def make_coord_grid_zyx(roi_zyx):
    rz, ry, rx = roi_zyx
    zz = np.linspace(-1.0, 1.0, rz, dtype=np.float32)[:, None, None]
    yy = np.linspace(-1.0, 1.0, ry, dtype=np.float32)[None, :, None]
    xx = np.linspace(-1.0, 1.0, rx, dtype=np.float32)[None, None, :]
    Zc = np.broadcast_to(zz, (rz, ry, rx))
    Yc = np.broadcast_to(yy, (rz, ry, rx))
    Xc = np.broadcast_to(xx, (rz, ry, rx))
    return np.stack([Zc, Yc, Xc], axis=0)  # (3, Z, Y, X)

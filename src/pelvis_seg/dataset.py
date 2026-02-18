# src/pelvis_seg/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

from .io_nrrd import load_ct_zyx, load_lab_zyx
from .preprocess import scale_intensity_to_01, make_coord_grid_zyx
from .sampler import rand_crop_classbalanced_zyx

# --- AUGMENTATIONS (numpy) ---------------------------------
from dataclasses import dataclass


# ============================================================
# Augmentation schedule (epoch-dependent)
# ============================================================

@dataclass
class AugSchedule:
    # epoch ranges (1-based)
    warmup_epochs: int = 10     # 1..warmup: почти без геометрии
    ramp_end: int = 60          # warmup+1..ramp_end: плавный разгон (0->1)

    # intensity
    p_noise: float = 0.20
    noise_sigma_max: float = 0.03   # в шкале [0..1]

    p_gamma: float = 0.20
    gamma_range: tuple = (0.85, 1.15)

    p_shift: float = 0.20
    shift_max: float = 0.05         # add to [0..1]

    p_blur: float = 0.10
    blur_sigma_max: float = 0.8     # в вокселях (очень мягко)

    # geometry (очень аккуратно)
    p_translate: float = 0.20
    translate_frac_max: float = 0.08  # доля от roi (в каждую ось)


def _ramp(epoch: int, warmup: int, ramp_end: int) -> float:
    # epoch: 1-based
    if epoch <= warmup:
        return 0.0
    if epoch >= ramp_end:
        return 1.0
    return float(epoch - warmup) / float(max(1, ramp_end - warmup))


def _gaussian_blur3d_separable(vol: np.ndarray, sigma: float) -> np.ndarray:
    """
    Простая separable 3D blur без scipy.
    ВНИМАНИЕ: это O(N * kernel) по каждому измерению, но для roi 128^3 и sigma<1 обычно терпимо.
    """
    if sigma <= 1e-6:
        return vol

    r = int(max(1, round(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma)).astype(np.float32)
    k /= float(k.sum())

    def conv1d(a: np.ndarray, axis: int) -> np.ndarray:
        pad_width = [(0, 0)] * a.ndim
        pad_width[axis] = (r, r)
        ap = np.pad(a, pad_width, mode="reflect")
        ap = np.moveaxis(ap, axis, 0)  # axis -> 0
        out = np.empty((ap.shape[0] - 2 * r,) + ap.shape[1:], dtype=np.float32)
        for i in range(out.shape[0]):
            window = ap[i:i + 2 * r + 1]     # (2r+1, ...)
            out[i] = np.tensordot(k, window, axes=(0, 0))
        out = np.moveaxis(out, 0, axis)
        return out

    v = vol.astype(np.float32, copy=False)
    v = conv1d(v, 0)
    v = conv1d(v, 1)
    v = conv1d(v, 2)
    return v


def _rand_intensity_aug(ct01: np.ndarray, rng: np.random.Generator, a: float, sch: AugSchedule) -> np.ndarray:
    """
    ct01 in [0..1] float32
    a: 0..1 strength
    """
    x = ct01.astype(np.float32, copy=False)

    if sch.p_noise > 0 and rng.random() < sch.p_noise:
        sigma = float(a) * float(sch.noise_sigma_max)
        if sigma > 0:
            x = x + rng.normal(0.0, sigma, size=x.shape).astype(np.float32)

    if sch.p_shift > 0 and rng.random() < sch.p_shift:
        shift = (float(rng.random()) * 2.0 - 1.0) * float(a) * float(sch.shift_max)
        x = x + np.float32(shift)

    if sch.p_gamma > 0 and rng.random() < sch.p_gamma:
        g0, g1 = sch.gamma_range
        gamma = np.float32(float(g0) + (float(g1) - float(g0)) * float(rng.random()))
        x = np.clip(x, 0.0, 1.0)
        x = np.power(x, gamma, dtype=np.float32)

    if sch.p_blur > 0 and rng.random() < sch.p_blur:
        sigma = float(a) * float(sch.blur_sigma_max)
        if sigma > 1e-6:
            x = _gaussian_blur3d_separable(x, float(sigma))

    return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)


def _rand_geo_aug_shift_only(ct: np.ndarray, lab: np.ndarray, rng: np.random.Generator, a: float, sch: AugSchedule):
    """
    Геометрия максимально безопасно: только небольшие integer translate в пределах ROI.
    Это сохраняет дискретность маски и не требует интерполяции.
    """
    if sch.p_translate <= 0 or rng.random() >= sch.p_translate:
        return ct, lab

    rz, ry, rx = ct.shape
    max_dz = int(round(float(a) * float(sch.translate_frac_max) * rz))
    max_dy = int(round(float(a) * float(sch.translate_frac_max) * ry))
    max_dx = int(round(float(a) * float(sch.translate_frac_max) * rx))

    dz = int(rng.integers(-max_dz, max_dz + 1)) if max_dz > 0 else 0
    dy = int(rng.integers(-max_dy, max_dy + 1)) if max_dy > 0 else 0
    dx = int(rng.integers(-max_dx, max_dx + 1)) if max_dx > 0 else 0
    if dz == 0 and dy == 0 and dx == 0:
        return ct, lab

    def shift3(a3: np.ndarray, dz: int, dy: int, dx: int, fill):
        out = np.full_like(a3, fill)
        z0s, z0d = (0, dz) if dz >= 0 else (-dz, 0)
        y0s, y0d = (0, dy) if dy >= 0 else (-dy, 0)
        x0s, x0d = (0, dx) if dx >= 0 else (-dx, 0)

        zlen = rz - abs(dz)
        ylen = ry - abs(dy)
        xlen = rx - abs(dx)
        if zlen <= 0 or ylen <= 0 or xlen <= 0:
            return out

        out[z0d:z0d+zlen, y0d:y0d+ylen, x0d:x0d+xlen] = a3[z0s:z0s+zlen, y0s:y0s+ylen, x0s:x0s+xlen]
        return out

    ct2 = shift3(ct, dz, dy, dx, fill=float(ct.min()))
    lab2 = shift3(lab, dz, dy, dx, fill=0)
    return ct2, lab2


# ============================================================
# Basic flips / rot90
# ============================================================

def rand_flip_zyx(ct, lab, axis: int, prob: float, rng):
    if prob > 0 and rng.random() < prob:
        ct = np.flip(ct, axis=axis).copy()
        lab = np.flip(lab, axis=axis).copy()
    return ct, lab


def rand_rot90_xy(ct, lab, prob: float, rng):
    if prob > 0 and rng.random() < prob:
        k = int(rng.integers(0, 4))
        ct = np.rot90(ct, k=k, axes=(1, 2)).copy()
        lab = np.rot90(lab, k=k, axes=(1, 2)).copy()
    return ct, lab


# ============================================================
# Sampling probs for classes 1..3
# ============================================================

def _normalize_probs_1_3(p):
    """
    Нормируем вероятности для классов 1..3 (sacrum/hip/femur).
    Если сумма=0 -> fallback равномерно.
    """
    p = np.asarray(p, dtype=np.float32).reshape(3,)
    s = float(p.sum())
    if s <= 1e-8:
        p = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        s = float(p.sum())
    return (p / s).astype(np.float32)


# ============================================================
# Remap labels -> {0..3}, ignore left/right (Variant A)
# ============================================================

def remap_ignore_lr_to_0_3(lab: np.ndarray, domain: str) -> np.ndarray:
    """
    Вариант A: игнорируем left/right и приводим все в схему 0..3:
      0 bg
      1 sacrum
      2 hip   (L+R)
      3 femur (L+R)

    domain:
      - dataset6: 0 bg, 1 sacrum, 2 hip_L, 3 hip_R, 4 lumbar
                 lumbar -> 0, hip_L/hip_R -> 2, femur GT нет
      - dataset8: 1 femur_L, 2 femur_R, 3 hip_L, 4 hip_R, 5 sacrum
      - smir:     0 bg, 1 sacrum, 2 hip_R, 3 hip_L, 4 femur_R, 5 femur_L
    """
    lab = lab.astype(np.uint8, copy=False)
    out = np.zeros_like(lab, dtype=np.uint8)

    if domain == "dataset6":
        out[lab == 1] = 1
        out[(lab == 2) | (lab == 3)] = 2
        # 4 (lumbar) -> 0
        return out

    if domain == "dataset8":
        out[lab == 5] = 1                 # sacrum -> 1
        out[(lab == 3) | (lab == 4)] = 2  # hips   -> 2
        out[(lab == 1) | (lab == 2)] = 3  # femurs -> 3
        return out

    # SMIR
    out[lab == 1] = 1
    out[(lab == 2) | (lab == 3)] = 2
    out[(lab == 4) | (lab == 5)] = 3
    return out


# ============================================================
# Optional: Femur-only remap (binary) (kept here for future)
# ============================================================

def remap_femur_binary(lab: np.ndarray, domain: str, cfg) -> np.ndarray:
    """
    {0=bg, 1=femur}
    - dataset8: femur ids (1,2)
    - smir: femur ids (4,5)
    - dataset6: femur отсутствует -> будет только bg
    """
    lab = lab.astype(np.uint8, copy=False)
    out = np.zeros_like(lab, dtype=np.uint8)

    if domain == "dataset8":
        ids = set(int(x) for x in getattr(cfg, "FEMUR_IDS_DATASET8", (1, 2)))
        for v in ids:
            out[lab == v] = 1
        return out

    ids = set(int(x) for x in getattr(cfg, "FEMUR_IDS_SMIR", (4, 5)))
    for v in ids:
        out[lab == v] = 1
    return out


# ============================================================
# Dataset
# ============================================================

class PelvisLabelmapDataset(Dataset):
    """
    items: dict:
      - image (path)
      - label (path)
      - case (id)
      - domain ("smir" | "dataset6" | "dataset8")

    Метки внутри датасета (вариант A): {0..3}
      0 bg
      1 sacrum
      2 hip
      3 femur
    """
    def __init__(self, items, cfg, train: bool, seed: int):
        self.items = list(items)
        self.cfg = cfg
        self.train = train
        self.rng = np.random.default_rng(seed)

        self.coord_grid = make_coord_grid_zyx(cfg.PATCH_SIZE) if cfg.USE_COORDS else None

        # probs для классов 1..3
        p_cfg = np.asarray(cfg.FG_CLASS_PROBS_1_3, dtype=np.float32)
        if p_cfg.shape != (3,):
            raise ValueError("FG_CLASS_PROBS_1_3 must have 3 values for classes 1..3 (sacrum, hip, femur).")
        self.fg_probs_1_3 = _normalize_probs_1_3(p_cfg)

        # epoch-aware aug
        self.epoch = 1
        self.aug = AugSchedule()

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        ct = load_ct_zyx(it["image"])      # (Z,Y,X)
        lab0 = load_lab_zyx(it["label"])   # raw labels
        domain = it.get("domain", "smir")
        

        # 1) remap -> {0..3}
        lab = remap_ignore_lr_to_0_3(lab0, domain)

        # 2) sampling policy
        probs_1_3 = self.fg_probs_1_3
        if domain == "dataset6":
            allowed = (1, 2)        # sacrum + hip (femur GT нет)
        else:
            allowed = (1, 2, 3)     # sacrum + hip + femur

        mode = getattr(self.cfg, "MODE", "full").lower()
        
        if mode == "femur":
            # бинарный таргет {0 bg, 1 femur}
            lab = remap_femur_binary(lab0, domain, self.cfg)
            allowed = (1,)              # таргетим только femur
            probs_1_3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # если sampler требует
        else:
            # full: {0..3}
            lab = remap_ignore_lr_to_0_3(lab0, domain)
            if domain == "dataset6":
                allowed = (1, 2)
            else:
                allowed = (1, 2, 3)
            probs_1_3 = self.fg_probs_1_3

        # 3) crop
        if self.train:
            ct_p, lab_p = rand_crop_classbalanced_zyx(
                ct, lab, self.cfg.PATCH_SIZE, self.rng,
                pos_prob=self.cfg.POS_PROB,
                tries=self.cfg.POS_TRIES,
                min_fg_vox=self.cfg.MIN_FG_VOXELS_IN_PATCH,
                min_target_vox=self.cfg.MIN_TARGET_VOXELS_IN_PATCH,
                class_balanced=self.cfg.CLASS_BALANCED_SAMPLING,
                probs_1_3=probs_1_3,
                allowed_classes=allowed,
            )

            # classic aug (label-safe)
            ct_p, lab_p = rand_flip_zyx(ct_p, lab_p, axis=2, prob=self.cfg.FLIP_X_PROB, rng=self.rng)
            ct_p, lab_p = rand_flip_zyx(ct_p, lab_p, axis=1, prob=self.cfg.FLIP_Y_PROB, rng=self.rng)
            ct_p, lab_p = rand_flip_zyx(ct_p, lab_p, axis=0, prob=self.cfg.FLIP_Z_PROB, rng=self.rng)
            ct_p, lab_p = rand_rot90_xy(ct_p, lab_p, prob=self.cfg.ROT90_PROB, rng=self.rng)

        else:
            ct_p, lab_p = rand_crop_classbalanced_zyx(
                ct, lab, self.cfg.PATCH_SIZE, self.rng,
                pos_prob=self.cfg.FAST_VAL_POS_PROB,
                tries=max(10, self.cfg.POS_TRIES // 2),
                min_fg_vox=self.cfg.MIN_FG_VOXELS_IN_PATCH,
                min_target_vox=self.cfg.FAST_VAL_MIN_TARGET_VOX,
                class_balanced=self.cfg.CLASS_BALANCED_SAMPLING,
                probs_1_3=probs_1_3,
                allowed_classes=allowed,
            )

        # 4) intensity scaling
        ct_p = scale_intensity_to_01(ct_p, self.cfg.A_MIN, self.cfg.A_MAX)
        lab_u8 = lab_p.astype(np.uint8, copy=False)

        # 5) epoch-scheduled augs (on ct in [0..1])
        a = _ramp(self.epoch, self.aug.warmup_epochs, self.aug.ramp_end)

        # intensity aug
        ct_p = _rand_intensity_aug(ct_p, self.rng, a, self.aug)

        # safe geometry aug (integer translate)
        if a > 0.0:
            ct_p, lab_u8_tmp = _rand_geo_aug_shift_only(ct_p, lab_u8, self.rng, a, self.aug)
            lab_u8 = lab_u8_tmp

        # 6) input channels
        if self.cfg.USE_COORDS:
            img = np.concatenate([ct_p[None, ...], self.coord_grid], axis=0).astype(np.float32, copy=False)
        else:
            img = ct_p[None, ...].astype(np.float32, copy=False)

        # DiceCELoss expects int64 labels shaped as (B,1,Z,Y,X)
        lab_t = lab_u8.astype(np.int64, copy=False)[None, ...]

        return {
            "image": torch.from_numpy(img),
            "label": torch.from_numpy(lab_t),
            "label_u8": torch.from_numpy(lab_u8),
            "case": it.get("case", f"idx{idx}"),
            "domain": domain,
        }

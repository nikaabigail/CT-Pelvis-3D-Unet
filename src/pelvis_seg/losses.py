# src/pelvis_seg/losses.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .io_nrrd import load_lab_zyx


# ---------------------------
# Remap helpers
# ---------------------------

def remap_full_ignore_lr(lab: np.ndarray, domain: str) -> np.ndarray:
    """
    FULL mode: ignore L/R -> labels {0..3}
      0 bg
      1 sacrum
      2 hip   (L+R)
      3 femur (L+R)

    domain:
      - smir:     0 bg, 1 sacrum, 2 hip_R, 3 hip_L, 4 femur_R, 5 femur_L
      - dataset6: 0 bg, 1 sacrum, 2 hip_L, 3 hip_R, 4 lumbar (ignore->0), femur отсутствует
      - dataset8: 0 bg, 1 femL, 2 femR, 3 hipL, 4 hipR, 5 sacrum
    """
    lab = lab.astype(np.uint8, copy=False)
    out = np.zeros_like(lab, dtype=np.uint8)

    if domain == "dataset6":
        out[lab == 1] = 1
        out[(lab == 2) | (lab == 3)] = 2
        return out

    if domain == "dataset8":
        out[lab == 5] = 1                 # sacrum
        out[(lab == 3) | (lab == 4)] = 2  # hip
        out[(lab == 1) | (lab == 2)] = 3  # femur
        return out

    # smir
    out[lab == 1] = 1
    out[(lab == 2) | (lab == 3)] = 2
    out[(lab == 4) | (lab == 5)] = 3
    return out


def remap_femur_binary(lab: np.ndarray, domain: str, cfg) -> np.ndarray:
    """
    FEMUR mode: binary labels {0..1}
      0 bg
      1 femur
    femur ids come from cfg for flexibility.
    """
    lab = lab.astype(np.uint8, copy=False)
    out = np.zeros_like(lab, dtype=np.uint8)

    if domain == "dataset8":
        ids = getattr(cfg, "FEMUR_IDS_DATASET8", (1, 2))
        ids = set(int(x) for x in ids)
        for v in ids:
            out[lab == v] = 1
        return out

    # smir
    ids = getattr(cfg, "FEMUR_IDS_SMIR", (4, 5))
    ids = set(int(x) for x in ids)
    for v in ids:
        out[lab == v] = 1
    return out


def remap_for_mode(lab: np.ndarray, domain: str, cfg, mode: str) -> np.ndarray:
    mode = str(mode).lower().strip()
    if mode == "femur":
        return remap_femur_binary(lab, domain, cfg)
    return remap_full_ignore_lr(lab, domain)


# ---------------------------
# Voxel counting / CE weights
# ---------------------------

def count_classes_voxels(items, num_classes: int, *, cfg, mode: str) -> np.ndarray:
    """
    Counts voxels per class AFTER remap for current mode.
    Returns shape (num_classes,).
    """
    counts = np.zeros((int(num_classes),), dtype=np.int64)
    for it in items:
        lab = load_lab_zyx(it["label"])
        domain = it.get("domain", "smir")
        lab = remap_for_mode(lab, domain, cfg, mode)
        bc = np.bincount(lab.reshape(-1), minlength=int(num_classes)).astype(np.int64)
        counts += bc
    return counts


def compute_ce_weights_from_train(
    train_items,
    num_classes: int,
    *,
    cfg,
    mode: str,
    clip=(0.2, 5.0),
    bg_weight=0.7,
):
    """
    Computes CE weights based on voxel frequency (inverse-ish),
    then normalizes + applies bg_weight.
    Returns: (weights_np, counts_np)
    """
    counts = count_classes_voxels(train_items, num_classes, cfg=cfg, mode=mode).astype(np.float64)

    # avoid div0
    counts = np.maximum(counts, 1.0)

    # inverse frequency (sqrt to soften)
    inv = 1.0 / np.sqrt(counts)

    # normalize to mean=1
    inv = inv / float(inv.mean())

    # apply bg weight (class 0)
    inv[0] = float(bg_weight)

    lo, hi = float(clip[0]), float(clip[1])
    inv = np.clip(inv, lo, hi).astype(np.float32)

    return inv, counts.astype(np.int64)


# ---------------------------
# Loss
# ---------------------------

class SoftDiceLoss(nn.Module):
    def __init__(self, include_bg: bool = False, smooth: float = 1e-5):
        super().__init__()
        self.include_bg = include_bg
        self.smooth = smooth

    def forward(self, logits, target):
        """
        logits: (B,C,Z,Y,X)
        target: (B,1,Z,Y,X) int64
        """
        B, C = logits.shape[:2]
        probs = torch.softmax(logits, dim=1)

        # one-hot
        y = target[:, 0].long()
        y_oh = F.one_hot(y, num_classes=C).permute(0, 4, 1, 2, 3).float()

        if not self.include_bg and C > 1:
            probs = probs[:, 1:]
            y_oh = y_oh[:, 1:]

        dims = tuple(range(2, probs.ndim))
        inter = (probs * y_oh).sum(dims)
        den = (probs.sum(dims) + y_oh.sum(dims))
        dice = (2.0 * inter + self.smooth) / (den + self.smooth)
        loss = 1.0 - dice.mean()
        return loss


class DiceCELoss(nn.Module):
    def __init__(self, ce_weights: torch.Tensor, include_bg_dice: bool = False, dice_weight: float = 1.0, ce_weight: float = 1.0):
        super().__init__()
        self.register_buffer("ce_weights", ce_weights)
        self.dice = SoftDiceLoss(include_bg=include_bg_dice)
        self.dice_w = float(dice_weight)
        self.ce_w = float(ce_weight)

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target[:, 0].long(), weight=self.ce_weights)
        dice = self.dice(logits, target)
        return self.ce_w * ce + self.dice_w * dice


def make_loss(ce_weights: torch.Tensor, include_bg_dice: bool = False):
    return DiceCELoss(ce_weights, include_bg_dice=include_bg_dice)

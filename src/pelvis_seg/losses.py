import numpy as np
import torch
import inspect
from monai.losses import DiceCELoss
from .io_nrrd import load_lab_zyx

def count_classes_voxels(cases, num_classes: int):
    counts = np.zeros((num_classes,), dtype=np.int64)
    for it in cases:
        lab = load_lab_zyx(it["label"])
        bc = np.bincount(lab.reshape(-1), minlength=num_classes)
        counts += bc.astype(np.int64)
    return counts

def compute_ce_weights_from_train(train_cases, num_classes: int, clip=(0.5, 8.0), bg_weight: float = 0.7):
    counts = count_classes_voxels(train_cases, num_classes)
    total = counts.sum()
    freqs = counts / max(total, 1)

    nonzero = freqs > 0
    if nonzero.sum() <= 1:
        w = np.ones((num_classes,), dtype=np.float32)
        w[0] = float(bg_weight)
        return w, counts

    median_freq = np.median(freqs[nonzero])
    w = np.zeros((num_classes,), dtype=np.float32)
    w[nonzero] = (median_freq / freqs[nonzero]).astype(np.float32)

    w_mean = w[nonzero].mean()
    if w_mean > 0:
        w = w / w_mean

    w[nonzero] = np.clip(w[nonzero], clip[0], clip[1]).astype(np.float32)
    w[~nonzero] = 0.0

    if counts[0] > 0:
        w[0] = float(bg_weight)

    return w, counts

def make_loss(ce_weights: torch.Tensor, include_bg_dice: bool):
    sig = inspect.signature(DiceCELoss.__init__)
    kwargs = dict(
        to_onehot_y=True,
        softmax=True,
        include_background=include_bg_dice,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )
    if "ce_weight" in sig.parameters:
        kwargs["ce_weight"] = ce_weights
    elif "weight" in sig.parameters:
        kwargs["weight"] = ce_weights
    else:
        raise RuntimeError("DiceCELoss has no ce_weight/weight parameter in this MONAI version.")
    return DiceCELoss(**kwargs)

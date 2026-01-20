import numpy as np
import torch
from torch.utils.data import Dataset

from .io_nrrd import load_ct_zyx, load_lab_zyx
from .preprocess import scale_intensity_to_01, make_coord_grid_zyx
from .sampler import rand_crop_classbalanced_zyx

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

class PelvisLabelmapDataset(Dataset):
    def __init__(self, items, cfg, train: bool, seed: int):
        self.items = list(items)
        self.cfg = cfg
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.coord_grid = make_coord_grid_zyx(cfg.PATCH_SIZE) if cfg.USE_COORDS else None

        p = np.asarray(cfg.FG_CLASS_PROBS_1_5, dtype=np.float32)
        if p.shape != (5,):
            raise ValueError("FG_CLASS_PROBS_1_5 must have 5 values for classes 1..5.")
        self.fg_probs_1_5 = (p / max(p.sum(), 1e-8)).astype(np.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        ct = load_ct_zyx(it["image"])
        lab = load_lab_zyx(it["label"])

        if self.train:
            ct_p, lab_p = rand_crop_classbalanced_zyx(
                ct, lab, self.cfg.PATCH_SIZE, self.rng,
                pos_prob=self.cfg.POS_PROB,
                tries=self.cfg.POS_TRIES,
                min_fg_vox=self.cfg.MIN_FG_VOXELS_IN_PATCH,
                min_target_vox=self.cfg.MIN_TARGET_VOXELS_IN_PATCH,
                class_balanced=self.cfg.CLASS_BALANCED_SAMPLING,
                probs_1_5=self.fg_probs_1_5,
            )

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
                probs_1_5=self.fg_probs_1_5,
            )

        ct_p = scale_intensity_to_01(ct_p, self.cfg.A_MIN, self.cfg.A_MAX)
        lab_u8 = lab_p.astype(np.uint8, copy=False)

        if self.cfg.USE_COORDS:
            img = np.concatenate([ct_p[None, ...], self.coord_grid], axis=0).astype(np.float32, copy=False)
        else:
            img = ct_p[None, ...].astype(np.float32, copy=False)

        lab_t = lab_u8.astype(np.int64, copy=False)[None, ...]  # (1,Z,Y,X)

        return {
            "image": torch.from_numpy(img),
            "label": torch.from_numpy(lab_t),
            "label_u8": torch.from_numpy(lab_u8),
            "case": it["case"],
        }

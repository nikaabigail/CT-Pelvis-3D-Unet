# src/pelvis_seg/sampler.py
import numpy as np
from .preprocess import pad_to_roi_zyx, crop_zyx


def _normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    s = float(p.sum())
    if s <= 1e-8:
        return np.ones_like(p, dtype=np.float32) / max(len(p), 1)
    return (p / s).astype(np.float32)


def choose_target_class(rng: np.random.Generator, classes, probs=None) -> int:
    classes = np.asarray(list(classes), dtype=np.int64)
    if classes.size == 0:
        raise ValueError("choose_target_class: empty classes list")

    if probs is None:
        p = np.ones((classes.size,), dtype=np.float32) / float(classes.size)
    else:
        p = _normalize_probs(np.asarray(probs, dtype=np.float32).reshape(-1))
        if p.size != classes.size:
            raise ValueError(f"choose_target_class: probs size {p.size} != classes size {classes.size}")

    return int(rng.choice(classes, p=p))


def rand_crop_classbalanced_zyx(
    ct, lab, roi_zyx, rng,
    pos_prob, tries, min_fg_vox, min_target_vox,
    class_balanced,
    probs_1_3,
    allowed_classes=None,
):
    """
    lab: {0..3}
    probs_1_3: вероятности для классов (1,2,3) в этой же схеме
    allowed_classes:
      - None -> (1,2,3)
      - dataset6 -> (1,2)
    """
    ct, lab = pad_to_roi_zyx(ct, lab, roi_zyx)
    Z, Y, X = ct.shape
    rz, ry, rx = roi_zyx

    want_pos = (rng.random() < pos_prob)

    if want_pos and (lab > 0).any():
        target_c = None
        target_coords = None

        if class_balanced:
            if allowed_classes is None:
                allowed_classes = (1, 2, 3)

            # probs_1_3 для 1..3, индексация: class-1
            probs_sel = []
            classes_sel = []
            p = np.asarray(probs_1_3, dtype=np.float32).reshape(3,)
            for c in allowed_classes:
                c = int(c)
                if 1 <= c <= 3:
                    classes_sel.append(c)
                    probs_sel.append(float(p[c - 1]))

            if len(classes_sel) > 0:
                target_try = choose_target_class(rng, classes_sel, probs_sel)
                if (lab == target_try).any():
                    target_c = target_try
                    target_coords = np.argwhere(lab == target_c)

        if target_coords is None:
            target_coords = np.argwhere(lab > 0)

        for _ in range(int(tries)):
            cz, cy, cx = target_coords[rng.integers(0, len(target_coords))]

            dz = int(rng.integers(-rz // 4, rz // 4 + 1))
            dy = int(rng.integers(-ry // 4, ry // 4 + 1))
            dx = int(rng.integers(-rx // 4, rx // 4 + 1))

            z0 = max(0, min(int(cz - rz // 2 + dz), Z - rz))
            y0 = max(0, min(int(cy - ry // 2 + dy), Y - ry))
            x0 = max(0, min(int(cx - rx // 2 + dx), X - rx))

            ct_c, lab_c = crop_zyx(ct, lab, z0, y0, x0, roi_zyx)

            fg_cnt = int((lab_c > 0).sum())
            if fg_cnt < int(min_fg_vox):
                continue

            if target_c is not None:
                tgt_cnt = int((lab_c == target_c).sum())
                if tgt_cnt < int(min_target_vox):
                    continue

            return ct_c, lab_c

        # fallback: кроп вокруг fg
        cz, cy, cx = target_coords[rng.integers(0, len(target_coords))]
        z0 = max(0, min(int(cz - rz // 2), Z - rz))
        y0 = max(0, min(int(cy - ry // 2), Y - ry))
        x0 = max(0, min(int(cx - rx // 2), X - rx))
        return crop_zyx(ct, lab, z0, y0, x0, roi_zyx)

    # negative/random crop
    z0 = int(rng.integers(0, Z - rz + 1)) if Z > rz else 0
    y0 = int(rng.integers(0, Y - ry + 1)) if Y > ry else 0
    x0 = int(rng.integers(0, X - rx + 1)) if X > rx else 0
    return crop_zyx(ct, lab, z0, y0, x0, roi_zyx)

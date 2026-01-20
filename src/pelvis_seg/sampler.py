import numpy as np
from .preprocess import pad_to_roi_zyx, crop_zyx

def choose_target_class_1_5(rng: np.random.Generator, probs_1_5):
    p = np.asarray(probs_1_5, dtype=np.float32)
    p = p / max(p.sum(), 1e-8)
    return int(rng.choice(np.arange(1, 6), p=p))

def rand_crop_classbalanced_zyx(
    ct, lab, roi_zyx, rng,
    pos_prob, tries, min_fg_vox, min_target_vox,
    class_balanced, probs_1_5,
):
    ct, lab = pad_to_roi_zyx(ct, lab, roi_zyx)
    Z, Y, X = ct.shape
    rz, ry, rx = roi_zyx

    want_pos = (rng.random() < pos_prob)

    if want_pos and (lab > 0).any():
        target_c = None
        target_coords = None

        if class_balanced:
            target_c = choose_target_class_1_5(rng, probs_1_5)
            if (lab == target_c).any():
                target_coords = np.argwhere(lab == target_c)
            else:
                target_c = None

        if target_coords is None:
            target_coords = np.argwhere(lab > 0)

        for _ in range(tries):
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

        cz, cy, cx = target_coords[rng.integers(0, len(target_coords))]
        z0 = max(0, min(int(cz - rz // 2), Z - rz))
        y0 = max(0, min(int(cy - ry // 2), Y - ry))
        x0 = max(0, min(int(cx - rx // 2), X - rx))
        return crop_zyx(ct, lab, z0, y0, x0, roi_zyx)

    z0 = int(rng.integers(0, Z - rz + 1)) if Z > rz else 0
    y0 = int(rng.integers(0, Y - ry + 1)) if Y > ry else 0
    x0 = int(rng.integers(0, X - rx + 1)) if X > rx else 0
    return crop_zyx(ct, lab, z0, y0, x0, roi_zyx)

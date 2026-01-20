import numpy as np

def dice_from_pred_gt_u8(pred_u8: np.ndarray, gt_u8: np.ndarray, num_classes: int, eps: float = 1e-6):
    C = int(num_classes)
    pred = np.ravel(pred_u8).astype(np.int32, copy=False)
    gt = np.ravel(gt_u8).astype(np.int32, copy=False)

    pair = gt * C + pred
    conf = np.bincount(pair, minlength=C * C).reshape(C, C).astype(np.int64)

    tp = np.diag(conf).astype(np.float64)
    pred_cnt = conf.sum(axis=0).astype(np.float64)
    gt_cnt = conf.sum(axis=1).astype(np.float64)

    dice = (2.0 * tp + eps) / (pred_cnt + gt_cnt + eps)
    return dice, gt_cnt.astype(np.int64)

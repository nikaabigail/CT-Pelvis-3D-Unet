# src/pelvis_seg/fast_val.py
import numpy as np
import torch
from torch.amp import autocast
from .dataset import PelvisLabelmapDataset
from .metrics import dice_from_pred_gt_u8

@torch.no_grad()
def fast_val_epoch(model, device, cfg, val_cases, epoch: int):
    model.eval()
    ds = PelvisLabelmapDataset(val_cases, cfg, train=False, seed=cfg.SEED + 10_000 + epoch)

    dice_all = []
    gtcnt_all = []

    for case_i, _ in enumerate(val_cases, start=1):
        for _ in range(int(cfg.FAST_VAL_PATCHES_PER_CASE)):
            batch = ds[case_i - 1]
            x = batch["image"][None, ...].to(device, non_blocking=True)
            gt_u8 = batch["label_u8"].cpu().numpy()

            if device.type == "cuda":
                with autocast("cuda"):
                    out = model(x)
            else:
                out = model(x)

            prob = torch.softmax(out, dim=1)
            pred = torch.argmax(prob, dim=1)[0]
            pred_u8 = pred.detach().cpu().to(torch.uint8).numpy()

            dice_c, gt_cnt = dice_from_pred_gt_u8(pred_u8, gt_u8, cfg.NUM_CLASSES)
            dice_all.append(torch.from_numpy(dice_c))
            gtcnt_all.append(torch.from_numpy(gt_cnt))

            del x, out, prob, pred

    dice_stack = torch.stack(dice_all).float()
    gtcnt_stack = torch.stack(gtcnt_all).long()

    mean_per_class = torch.zeros((cfg.NUM_CLASSES,), dtype=torch.float32)
    for c in range(cfg.NUM_CLASSES):
        if c == 0:
            mean_per_class[c] = dice_stack[:, c].mean()
        else:
            mask = (gtcnt_stack[:, c] > 0)
            mean_per_class[c] = dice_stack[mask, c].mean() if mask.any() else float("nan")

    dice_bg = float(mean_per_class[0].item())

    fg_vals = mean_per_class[1:]
    fg_vals = fg_vals[torch.isfinite(fg_vals)]
    dice_fg = float(fg_vals.mean().item()) if len(fg_vals) > 0 else float("nan")

    # Красивый вывод без None
    per_class_list = []
    for v in mean_per_class.tolist():
        v = float(v)
        per_class_list.append("NA" if not np.isfinite(v) else round(v, 4))

    return dice_bg, dice_fg, per_class_list

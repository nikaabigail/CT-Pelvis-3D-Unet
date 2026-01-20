import os
import time
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from .utils import set_seed, now
from .config import save_config
from .io_nrrd import build_pairs_from_dirs
from .losses import compute_ce_weights_from_train, count_classes_voxels, make_loss
from .dataset import PelvisLabelmapDataset
from .fast_val import fast_val_epoch
from .model import build_model

def pick_split_with_fg(data, val_num, tries=200, seed=42, num_classes=6):
    import numpy as np
    from .losses import count_classes_voxels
    rng = np.random.default_rng(seed)
    data = list(data)

    best = None
    best_fg = -1

    for _ in range(tries):
        idx = np.arange(len(data))
        rng.shuffle(idx)
        val_idx = idx[:val_num]
        train_idx = idx[val_num:]

        train_cases = [data[i] for i in train_idx]
        val_cases = [data[i] for i in val_idx]

        val_counts = count_classes_voxels(val_cases, num_classes)
        fg = int(val_counts[1:].sum())
        if fg > 0:
            return train_cases, val_cases, val_counts

        if fg > best_fg:
            best_fg = fg
            best = (train_cases, val_cases, val_counts)

    return best[0], best[1], best[2]

def save_checkpoint(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))

def run_training(cfg):
    os.makedirs(cfg.ROOT_DIR, exist_ok=True)
    save_config(cfg, cfg.ROOT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = (device.type == "cuda")
    set_seed(cfg.SEED)

    print("Device:", device)

    print("[INFO] Building pairs...")
    pairs, missing_ct, missing_lab = build_pairs_from_dirs(cfg.CT_DIR, cfg.LAB_DIR)
    print(f"[INFO] pairs={len(pairs)}")
    if missing_ct:
        print(f"[WARN] LAB without CT: {len(missing_ct)} (first 5): {missing_ct[:5]}")
    if missing_lab:
        print(f"[WARN] CT without LAB: {len(missing_lab)} (first 5): {missing_lab[:5]}")
    if len(pairs) == 0:
        raise RuntimeError("No pairs found.")
    if len(pairs) <= cfg.VAL_NUM:
        raise RuntimeError("Too few cases for validation split.")

    print("[INFO] Picking split...")
    train_cases, val_cases, _ = pick_split_with_fg(
        pairs, cfg.VAL_NUM, tries=cfg.SPLIT_TRIES, seed=cfg.SEED, num_classes=cfg.NUM_CLASSES
    )
    print(f"Train: {len(train_cases)}, Val: {len(val_cases)}")

    print("[INIT] Counting voxel occurrences per class...")
    train_counts = count_classes_voxels(train_cases, cfg.NUM_CLASSES)
    val_counts = count_classes_voxels(val_cases, cfg.NUM_CLASSES)
    print("[INIT] TRAIN voxel counts:", train_counts.tolist())
    print("[INIT] VAL   voxel counts:", val_counts.tolist())

    w_np, _ = compute_ce_weights_from_train(train_cases, cfg.NUM_CLASSES, clip=cfg.CE_CLIP, bg_weight=cfg.BG_WEIGHT)
    if cfg.BOOST_CLASSES_4_5 and cfg.BOOST_CLASSES_4_5 > 1.0:
        w_np[4] = float(np.clip(w_np[4] * cfg.BOOST_CLASSES_4_5, cfg.CE_CLIP[0], cfg.CE_CLIP[1]))
        w_np[5] = float(np.clip(w_np[5] * cfg.BOOST_CLASSES_4_5, cfg.CE_CLIP[0], cfg.CE_CLIP[1]))
    print("[INIT] CE weights:", [round(float(x), 4) for x in w_np.tolist()])

    in_ch = 1 + (3 if cfg.USE_COORDS else 0)
    model = build_model(in_channels=in_ch, out_channels=cfg.NUM_CLASSES).to(device)

    ce_weights = torch.tensor(w_np, device=device, dtype=torch.float32)
    loss_fn = make_loss(ce_weights, include_bg_dice=cfg.INCLUDE_BACKGROUND_DICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    scaler = GradScaler("cuda") if use_cuda else None

    train_loader = DataLoader(
        PelvisLabelmapDataset(train_cases, cfg, train=True, seed=cfg.SEED),
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=use_cuda,
    )

    ckpt_dir = Path(cfg.ROOT_DIR) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_fg = -1.0
    best_path = None

    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{cfg.MAX_EPOCHS} ===")
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        steps = 0

        for bi, batch in enumerate(train_loader, start=1):
            steps += 1
            case_name = batch.get("case", "?")
            if isinstance(case_name, (list, tuple)):
                case_name = case_name[0]

            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            gt_u8 = batch["label_u8"].cpu().numpy()[0]

            optimizer.zero_grad(set_to_none=True)

            if use_cuda:
                with autocast("cuda"):
                    out = model(x)
                    loss = loss_fn(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()

            epoch_loss += float(loss.item())

            if (bi == 1) or (cfg.PRINT_EVERY > 0 and bi % cfg.PRINT_EVERY == 0):
                with torch.no_grad():
                    prob = torch.softmax(out, dim=1)
                    maxP = float(prob.max().detach().cpu().item())
                    pred = torch.argmax(prob, dim=1)[0]
                    pred_u8 = pred.detach().cpu().to(torch.uint8).numpy()
                    gt_fg = int((gt_u8 > 0).sum())
                    pred_fg = int((pred_u8 > 0).sum())
                print(f"[{now()}] [TRAIN] bi={bi:02d}/{len(train_loader)} "
                      f"case={case_name} loss={float(loss.item()):.4f} "
                      f"GT_fg={gt_fg} PRED_fg={pred_fg} maxP={maxP:.3f}")

            del x, y, out, loss
            if use_cuda and (bi % 10 == 0):
                torch.cuda.empty_cache()

        epoch_loss /= max(steps, 1)
        print(f"Epoch {epoch} average loss: {epoch_loss:.4f} (time {time.time() - t0:.1f}s)")

        do_fast_val = cfg.FAST_VAL_ENABLE and (epoch % max(1, cfg.FAST_VAL_EVERY_EPOCH) == 0)
        if do_fast_val:
            dice_bg, dice_fg, per_class = fast_val_epoch(model, device, cfg, val_cases, epoch)
            print(f"[FAST-VAL] Dice_bg={dice_bg:.4f} Dice_fg={dice_fg:.4f} per_class={per_class}")

            if np.isfinite(dice_fg) and dice_fg > best_fg:
                best_fg = dice_fg
                best_path = ckpt_dir / f"best_epoch_{epoch:03d}_fg_{best_fg:.4f}.pt"
                save_checkpoint(
                    best_path,
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_fg": best_fg,
                        "ce_weights": ce_weights.detach().cpu(),
                        "config": cfg.__dict__,
                        "split": {
                            "train_cases": [x["case"] for x in train_cases],
                            "val_cases": [x["case"] for x in val_cases],
                        },
                        "stats": {"train_counts": train_counts.tolist(), "val_counts": val_counts.tolist()},
                        "fast_val": {"dice_bg": dice_bg, "dice_fg": dice_fg, "per_class": per_class},
                    },
                )
                print(f"[BEST] Saved checkpoint: {best_path}")

    print("\nTraining finished.")
    if best_path:
        print(f"Best FAST-VAL foreground Dice: {best_fg:.4f} @ {best_path}")

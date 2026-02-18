# src/pelvis_seg/train_loop.py
import os
import time
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from .utils import set_seed, now
from .config import save_config
from .io_nrrd import build_pairs_from_dirs, build_pairs_same_filename
from .losses import compute_ce_weights_from_train, count_classes_voxels, make_loss
from .dataset import PelvisLabelmapDataset
from .fast_val import fast_val_epoch
from .model import build_model


def save_checkpoint(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def _pick_split_with_fg(
    data,
    val_num: int,
    tries: int,
    seed: int,
    *,
    cfg,
    mode: str,
):
    """
    Подбираем train/val split так, чтобы в val гарантированно был foreground
    в терминах ТЕКУЩЕГО mode (full: классы 1..3; femur: класс 1).
    Split делаем только по SMIR (как и было).
    """
    rng = np.random.default_rng(seed)
    data = list(data)

    best = None
    best_fg = -1

    for _ in range(int(tries)):
        idx = np.arange(len(data))
        rng.shuffle(idx)
        val_idx = idx[:val_num]
        train_idx = idx[val_num:]

        train_cases = [data[i] for i in train_idx]
        val_cases = [data[i] for i in val_idx]

        val_counts = count_classes_voxels(
            val_cases,
            num_classes=int(cfg.NUM_CLASSES),
            cfg=cfg,
            mode=mode,
        )
        fg = int(val_counts[1:].sum())
        if fg > 0:
            return train_cases, val_cases, val_counts

        if fg > best_fg:
            best_fg = fg
            best = (train_cases, val_cases, val_counts)

    # fallback
    return best[0], best[1], best[2]


def run_training(cfg):
    # ----------------------------
    # 0) MODE + NUM_CLASSES
    # ----------------------------
    mode = getattr(cfg, "MODE", "full")
    mode = str(mode).lower().strip()
    if mode not in ("full", "femur"):
        print(f"[WARN] Unknown MODE='{mode}', fallback to 'full'")
        mode = "full"

    if mode == "femur":
        cfg.NUM_CLASSES = 2  # bg / femur
    else:
        cfg.NUM_CLASSES = 4  # bg / sacrum / hip / femur

    # ----------------------------
    # 1) I/O + device
    # ----------------------------
    os.makedirs(cfg.ROOT_DIR, exist_ok=True)
    save_config(cfg, cfg.ROOT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = (device.type == "cuda")
    set_seed(cfg.SEED)

    print("Device:", device)
    print(f"[INFO] MODE={mode} NUM_CLASSES={cfg.NUM_CLASSES}")

    # ----------------------------
    # 2) Build pairs: SMIR
    # ----------------------------
    print("[INFO] Building pairs (SMIR)...")
    smir_pairs, smir_missing_ct, smir_missing_lab = build_pairs_from_dirs(
        cfg.CT_DIR, cfg.LAB_DIR, domain="smir"
    )
    print(f"[INFO] SMIR pairs={len(smir_pairs)}")
    if smir_missing_ct:
        print(f"[WARN] SMIR LAB without CT: {len(smir_missing_ct)} (first 5): {smir_missing_ct[:5]}")
    if smir_missing_lab:
        print(f"[WARN] SMIR CT without LAB: {len(smir_missing_lab)} (first 5): {smir_missing_lab[:5]}")
    if len(smir_pairs) == 0:
        raise RuntimeError("No SMIR pairs found. Check CT_DIR/LAB_DIR.")

    # ----------------------------
    # 3) dataset6 (optional) - only useful in full mode
    # ----------------------------
    dataset6_pairs = []
    if getattr(cfg, "USE_DATASET6", False):
        print("[INFO] Building pairs (dataset6)...")
        dataset6_pairs, d6_missing_ct, d6_missing_lab = build_pairs_same_filename(
            cfg.DATASET6_CT_DIR, cfg.DATASET6_LAB_DIR, domain="dataset6"
        )
        print(f"[INFO] dataset6 pairs={len(dataset6_pairs)}")
        if d6_missing_ct:
            print(f"[WARN] dataset6 LAB without CT: {len(d6_missing_ct)} (first 5): {d6_missing_ct[:5]}")
        if d6_missing_lab:
            print(f"[WARN] dataset6 CT without LAB: {len(d6_missing_lab)} (first 5): {d6_missing_lab[:5]}")

        if len(dataset6_pairs) == 0:
            print("[WARN] USE_DATASET6=True but no dataset6 pairs found. Continue without dataset6.")
        else:
            # sample ratio / max cases (как у тебя)
            ratio = float(getattr(cfg, "DATASET6_SAMPLE_RATIO", 1.0))
            if ratio < 1.0:
                step = max(int(round(1.0 / max(ratio, 1e-6))), 1)
                dataset6_pairs = dataset6_pairs[::step]

            max_cases = getattr(cfg, "DATASET6_MAX_CASES", None)
            if isinstance(max_cases, int) and max_cases > 0:
                dataset6_pairs = dataset6_pairs[:max_cases]

            print(f"[INFO] dataset6 used in TRAIN (pre-mode-filter): {len(dataset6_pairs)}")

    # femur mode: dataset6 не используем вообще
    if mode == "femur" and len(dataset6_pairs) > 0:
        print("[INFO] MODE=femur -> dataset6 ignored (no femur labels there).")
        dataset6_pairs = []

    # ----------------------------
    # 4) dataset8 (optional) - can be used in both modes
    # ----------------------------
    dataset8_pairs = []
    if getattr(cfg, "USE_DATASET8", False):
        print("[INFO] Building pairs (dataset8)...")
        dataset8_pairs, d8_missing_ct, d8_missing_lab = build_pairs_same_filename(
            cfg.DATASET8_CT_DIR, cfg.DATASET8_LAB_DIR, domain="dataset8"
        )
        print(f"[INFO] dataset8 pairs={len(dataset8_pairs)}")
        if d8_missing_ct:
            print(f"[WARN] dataset8 LAB without CT: {len(d8_missing_ct)} (first 5): {d8_missing_ct[:5]}")
        if d8_missing_lab:
            print(f"[WARN] dataset8 CT without LAB: {len(d8_missing_lab)} (first 5): {d8_missing_lab[:5]}")

        if len(dataset8_pairs) == 0:
            print("[WARN] USE_DATASET8=True but no dataset8 pairs found. Continue without dataset8.")

    # ----------------------------
    # 5) Split: only SMIR for val
    # ----------------------------
    val_domain = getattr(cfg, "VAL_DOMAIN", "smir").lower().strip()
    if val_domain != "smir":
        print(f"[WARN] VAL_DOMAIN='{val_domain}' not supported here. Forcing 'smir'.")
        val_domain = "smir"

    if len(smir_pairs) <= int(cfg.VAL_NUM):
        raise RuntimeError("Too few SMIR cases for validation split.")

    print("[INFO] Picking train/val split on SMIR only...")
    smir_train, smir_val, _ = _pick_split_with_fg(
        smir_pairs,
        val_num=int(cfg.VAL_NUM),
        tries=int(getattr(cfg, "SPLIT_TRIES", 200)),
        seed=int(cfg.SEED),
        cfg=cfg,
        mode=mode,
    )
    print(f"[INFO] SMIR Train: {len(smir_train)}, SMIR Val: {len(smir_val)}")

    # ----------------------------
    # 6) Final train/val cases
    # ----------------------------
    train_cases = list(smir_train) + list(dataset6_pairs) + list(dataset8_pairs)
    val_cases = list(smir_val)

    # ----------------------------
    # 7) Diagnostics: voxel counts (after remap in losses.py)
    # ----------------------------
    print("[INIT] Counting voxel occurrences per class (after remap)...")
    train_counts = count_classes_voxels(train_cases, int(cfg.NUM_CLASSES), cfg=cfg, mode=mode)
    val_counts = count_classes_voxels(val_cases, int(cfg.NUM_CLASSES), cfg=cfg, mode=mode)
    print("[INIT] TRAIN voxel counts:", train_counts.tolist())
    print("[INIT] VAL   voxel counts:", val_counts.tolist())

    if int(val_counts[1:].sum()) == 0:
        print("[WARN] VAL has no foreground; metrics meaningless.")
    else:
        print("[OK] VAL has foreground voxels.")

    # ----------------------------
    # 8) CE weights from TRAIN (same mode)
    # ----------------------------
    w_np, _ = compute_ce_weights_from_train(
        train_cases,
        int(cfg.NUM_CLASSES),
        cfg=cfg,
        mode=mode,
        clip=cfg.CE_CLIP,
        bg_weight=float(cfg.BG_WEIGHT),
    )

    # optional: boost femur in full mode (class index 3)
    boost = float(getattr(cfg, "BOOST_FEMUR", 1.0))
    if mode == "full" and boost > 1.0 and int(cfg.NUM_CLASSES) == 4:
        w_np[3] = float(np.clip(w_np[3] * boost, cfg.CE_CLIP[0], cfg.CE_CLIP[1]))

    print("[INIT] CE weights:", [round(float(x), 4) for x in w_np.tolist()])

    # ----------------------------
    # 9) Model / loss / optim
    # ----------------------------
    in_ch = 1 + (3 if getattr(cfg, "USE_COORDS", False) else 0)
    model = build_model(in_channels=in_ch, out_channels=int(cfg.NUM_CLASSES)).to(device)

    ce_weights = torch.tensor(w_np, device=device, dtype=torch.float32)
    loss_fn = make_loss(ce_weights, include_bg_dice=bool(getattr(cfg, "INCLUDE_BACKGROUND_DICE", False)))

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.LR))
    scaler = GradScaler("cuda") if use_cuda else None

    # ----------------------------
    # 10) DataLoader
    # ----------------------------
    train_ds = PelvisLabelmapDataset(train_cases, cfg, train=True, seed=int(cfg.SEED))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.BATCH_SIZE),
        shuffle=True,
        num_workers=int(cfg.NUM_WORKERS),
        pin_memory=use_cuda,
    )

    ckpt_dir = Path(cfg.ROOT_DIR) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_fg = -1.0
    best_path = None

    # ----------------------------
    # 11) Train loop
    # ----------------------------
    for epoch in range(1, int(cfg.MAX_EPOCHS) + 1):
        train_ds.set_epoch(epoch)

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

            if (bi == 1) or (int(getattr(cfg, "PRINT_EVERY", 0)) > 0 and bi % int(cfg.PRINT_EVERY) == 0):
                with torch.no_grad():
                    prob = torch.softmax(out, dim=1)
                    maxP = float(prob.max().detach().cpu().item())
                    pred = torch.argmax(prob, dim=1)[0]
                    pred_u8 = pred.detach().cpu().to(torch.uint8).numpy()
                    gt_fg = int((gt_u8 > 0).sum())
                    pred_fg = int((pred_u8 > 0).sum())

                print(
                    f"[{now()}] [TRAIN] bi={bi:02d}/{len(train_loader)} "
                    f"case={case_name} loss={float(loss.item()):.4f} "
                    f"GT_fg={gt_fg} PRED_fg={pred_fg} maxP={maxP:.3f}"
                )

            del x, y, out, loss
            if use_cuda and (bi % 10 == 0):
                torch.cuda.empty_cache()

        epoch_loss /= max(steps, 1)
        print(f"Epoch {epoch} average loss: {epoch_loss:.4f} (time {time.time() - t0:.1f}s)")

        # ----------------------------
        # 12) FAST-VAL (still on SMIR only)
        # ----------------------------
        do_fast_val = bool(getattr(cfg, "FAST_VAL_ENABLE", True)) and (
            epoch % max(1, int(getattr(cfg, "FAST_VAL_EVERY_EPOCH", 1))) == 0
        )
        if do_fast_val:
            dice_bg, dice_fg, per_class = fast_val_epoch(model, device, cfg, val_cases, epoch)
            print(f"[FAST-VAL] Dice_bg={dice_bg:.4f} Dice_fg={dice_fg:.4f} per_class={per_class}")

            if np.isfinite(dice_fg) and dice_fg > best_fg:
                best_fg = float(dice_fg)
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
                        "mode": mode,
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

    return float(best_fg)

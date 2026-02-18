# src/pelvis_seg/infer_best.py
# ------------------------------------------------------------
# GUI просмотрщик: CT + overlay GT/PRED по слайсам (3 оси)
# + sliding-window inference из чекпоинта MONAI UNet
# + легенда "цвет -> класс/кость"
#
# PowerShell (из D:\Val_test_sliding):
#   python -m src.pelvis_seg.infer_best --ckpt "D:\Dic\MONAI_dataset\checkpoints\best_epoch_078_fg_0.7499.pt" --ct "D:\Dic\CT_non_format\SMIR.Body.025Y.M.CT.57697-Pelvis-Thighs.nrrd"
#
# С GT:
#   python -m src.pelvis_seg.infer_best --ckpt "...\best_epoch_078_fg_0.7499.pt" --ct "...\ct.nrrd" --lab "...\lab.nrrd"
#
# Требования:
#   pip install numpy torch monai simpleitk matplotlib
# ------------------------------------------------------------

import os
import argparse
import numpy as np

import torch
from torch.amp import autocast

import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

import tkinter as tk
from tkinter import ttk

from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet


# -----------------------------
# IO (NRRD via SimpleITK)
# -----------------------------
def load_nrrd_zyx(path: str) -> np.ndarray:
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # (Z,Y,X)
    return arr


def save_nrrd_zyx(path: str, arr_zyx: np.ndarray):
    img = sitk.GetImageFromArray(arr_zyx)
    sitk.WriteImage(img, path)


def scale_intensity_to_01(ct_zyx: np.ndarray, a_min: int, a_max: int) -> np.ndarray:
    x = ct_zyx.astype(np.float32, copy=False)
    x = (x - float(a_min)) / max(float(a_max - a_min), 1.0)
    x = np.clip(x, 0.0, 1.0)
    return x


def make_coord_grid_zyx(roi_zyx):
    rz, ry, rx = roi_zyx
    zz = np.linspace(-1.0, 1.0, rz, dtype=np.float32)[:, None, None]
    yy = np.linspace(-1.0, 1.0, ry, dtype=np.float32)[None, :, None]
    xx = np.linspace(-1.0, 1.0, rx, dtype=np.float32)[None, None, :]
    Z = np.broadcast_to(zz, (rz, ry, rx))
    Y = np.broadcast_to(yy, (rz, ry, rx))
    X = np.broadcast_to(xx, (rz, ry, rx))
    return np.stack([Z, Y, X], axis=0)  # (3,Z,Y,X)


# -----------------------------
# Model (match твоему build_model)
# -----------------------------
def build_model(in_channels: int, out_channels: int):
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


# -----------------------------
# CKPT config parsing
# -----------------------------
def parse_ckpt_config(ckpt: dict):
    cfg = ckpt.get("config", {}) or {}
    num_classes = int(cfg.get("NUM_CLASSES", ckpt.get("num_classes", 6)))

    patch = cfg.get("PATCH_SIZE", (128, 128, 128))
    if isinstance(patch, list):
        patch = tuple(int(x) for x in patch)
    else:
        patch = tuple(int(x) for x in patch)

    use_coords = bool(cfg.get("USE_COORDS", True))
    a_min = int(cfg.get("A_MIN", -1000))
    a_max = int(cfg.get("A_MAX", 3000))
    return cfg, num_classes, patch, use_coords, a_min, a_max


def get_class_names(num_classes: int):
    # твой вариант A (ignore L/R): 4 класса
    if num_classes == 4:
        return ["BG", "Sacrum", "Hip", "Femur"]
    if num_classes == 6:
        # если вдруг вернешься к 6
        return ["BG", "Sacrum", "Hip_L", "Hip_R", "Class4", "Class5"]
    return [f"Class_{i}" for i in range(num_classes)]


def make_fixed_cmap(num_classes: int):
    # 0 — прозрачный BG
    base = [
        (0, 0, 0, 0.0),       # 0 BG transparent
        (1.0, 0.2, 0.2, 1.0), # 1 red
        (0.2, 1.0, 0.2, 1.0), # 2 green
        (0.2, 0.5, 1.0, 1.0), # 3 blue
        (1.0, 1.0, 0.2, 1.0), # 4 yellow
        (1.0, 0.2, 1.0, 1.0), # 5 magenta
        (0.2, 1.0, 1.0, 1.0), # 6 cyan
        (1.0, 0.6, 0.2, 1.0), # 7 orange
    ]
    colors = base[:num_classes]
    while len(colors) < num_classes:
        colors.append((1, 1, 1, 1))
    return ListedColormap(colors)


# -----------------------------
# Slice helpers
# -----------------------------
def get_slice(vol_zyx: np.ndarray, axis: str, idx: int) -> np.ndarray:
    if axis == "axial":      # Z fixed -> (Y,X)
        idx = int(np.clip(idx, 0, vol_zyx.shape[0] - 1))
        return vol_zyx[idx, :, :]
    if axis == "coronal":    # Y fixed -> (Z,X)
        idx = int(np.clip(idx, 0, vol_zyx.shape[1] - 1))
        return vol_zyx[:, idx, :]
    if axis == "sagittal":   # X fixed -> (Z,Y)
        idx = int(np.clip(idx, 0, vol_zyx.shape[2] - 1))
        return vol_zyx[:, :, idx]
    raise ValueError("axis must be axial/coronal/sagittal")


def axis_len(vol_zyx: np.ndarray, axis: str) -> int:
    if axis == "axial":
        return vol_zyx.shape[0]
    if axis == "coronal":
        return vol_zyx.shape[1]
    if axis == "sagittal":
        return vol_zyx.shape[2]
    raise ValueError("axis must be axial/coronal/sagittal")


# -----------------------------
# Inference
# -----------------------------
@torch.no_grad()
def infer_volume(
    ct_zyx: np.ndarray,
    ckpt_path: str,
    roi_zyx=(128, 128, 128),
    overlap=0.5,
    force_cpu=False,
):
    device = torch.device("cpu" if force_cpu or (not torch.cuda.is_available()) else "cuda")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg, num_classes, patch, use_coords, a_min, a_max = parse_ckpt_config(ckpt)

    # если ROI не задан явно — берем из ckpt
    if roi_zyx is None:
        roi_zyx = patch

    print(f"[INFO] Inference (device={device}, roi={tuple(roi_zyx)}, overlap={overlap}, classes={num_classes}, coords={use_coords}) ...")

    in_ch = 1 + (3 if use_coords else 0)
    model = build_model(in_channels=in_ch, out_channels=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    ct01 = scale_intensity_to_01(ct_zyx, a_min, a_max)  # (Z,Y,X)
    x = torch.from_numpy(ct01[None, None, ...]).float().to(device)  # (1,1,Z,Y,X)

    if use_coords:
        coord = make_coord_grid_zyx(roi_zyx)  # (3,rz,ry,rx)
        coord = torch.from_numpy(coord[None, ...]).float().to(device)  # (1,3,rz,ry,rx)

        def predictor(patch):
            B = patch.shape[0]
            c = coord.expand(B, -1, -1, -1, -1)
            inp = torch.cat([patch, c], dim=1)
            if device.type == "cuda":
                with autocast("cuda"):
                    return model(inp)
            return model(inp)
    else:
        def predictor(patch):
            if device.type == "cuda":
                with autocast("cuda"):
                    return model(patch)
            return model(patch)

    out = sliding_window_inference(
        inputs=x,
        roi_size=tuple(roi_zyx),
        sw_batch_size=1,
        predictor=predictor,
        overlap=float(overlap),
        mode="gaussian",
    )

    prob = torch.softmax(out, dim=1)
    pred = torch.argmax(prob, dim=1)[0].detach().cpu().numpy().astype(np.uint8)  # (Z,Y,X)
    maxp = float(prob.max().detach().cpu().item())

    return pred, ct01, cfg, num_classes, maxp


# -----------------------------
# GUI
# -----------------------------
def run_gui(ct01, pred, gt, class_names):
    num_classes = len(class_names)
    cmap = make_fixed_cmap(num_classes)

    root = tk.Tk()
    root.title("PelvisSeg Viewer (CT + Pred/GT)")

    ax_var = tk.StringVar(value="axial")
    idx_var = tk.IntVar(value=ct01.shape[0] // 2)
    alpha_var = tk.IntVar(value=45)
    show_pred = tk.IntVar(value=1)
    show_gt = tk.IntVar(value=1 if gt is not None else 0)

    info = ttk.Label(root, text="Axis/Slider: view slices | Overlay: Pred/GT | Legend: color -> class")
    info.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

    ctrl = ttk.Frame(root)
    ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

    ttk.Label(ctrl, text="Axis").grid(row=0, column=0, sticky="w", padx=4)
    axis_box = ttk.Combobox(ctrl, textvariable=ax_var, values=["axial", "coronal", "sagittal"], width=10, state="readonly")
    axis_box.grid(row=0, column=1, sticky="w", padx=4)

    ttk.Label(ctrl, text="Slice").grid(row=0, column=2, sticky="w", padx=4)
    slice_scale = ttk.Scale(ctrl, from_=0, to=ct01.shape[0] - 1, orient=tk.HORIZONTAL)
    slice_scale.grid(row=0, column=3, sticky="we", padx=4)
    ctrl.columnconfigure(3, weight=1)

    idx_entry = ttk.Entry(ctrl, width=6, textvariable=idx_var)
    idx_entry.grid(row=0, column=4, sticky="w", padx=4)

    ttk.Label(ctrl, text="Alpha").grid(row=0, column=5, sticky="w", padx=4)
    alpha_scale = ttk.Scale(ctrl, from_=0, to=100, orient=tk.HORIZONTAL)
    alpha_scale.grid(row=0, column=6, sticky="w", padx=4)
    alpha_scale.set(alpha_var.get())

    chk_pred = ttk.Checkbutton(ctrl, text="Pred", variable=show_pred)
    chk_pred.grid(row=0, column=7, sticky="w", padx=4)

    chk_gt = ttk.Checkbutton(ctrl, text="GT", variable=show_gt)
    chk_gt.grid(row=0, column=8, sticky="w", padx=4)

    fig = plt.Figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(111)
    ax.axis("off")
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def update_slider_range():
        axis = ax_var.get()
        n = axis_len(ct01, axis)
        slice_scale.configure(to=max(0, n - 1))
        i = int(np.clip(idx_var.get(), 0, n - 1))
        idx_var.set(i)
        slice_scale.set(i)

    def redraw(*_):
        axis = ax_var.get()
        n = axis_len(ct01, axis)
        i = int(np.clip(int(float(slice_scale.get())), 0, n - 1))
        idx_var.set(i)

        ct2d = get_slice(ct01, axis, i)
        ax.clear()
        ax.axis("off")
        ax.imshow(ct2d, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")

        alpha = float(alpha_scale.get()) / 100.0

        if show_pred.get() == 1:
            p2d = get_slice(pred, axis, i).astype(np.int32, copy=False)
            pm = np.ma.masked_where(p2d == 0, p2d)
            ax.imshow(pm, cmap=cmap, vmin=0, vmax=num_classes - 1, alpha=alpha, interpolation="nearest")

        if gt is not None and show_gt.get() == 1:
            g2d = get_slice(gt, axis, i).astype(np.int32, copy=False)
            gm = np.ma.masked_where(g2d == 0, g2d)
            ax.imshow(gm, cmap=cmap, vmin=0, vmax=num_classes - 1, alpha=max(0.15, alpha * 0.6), interpolation="nearest")

        ax.set_title(f"{axis} slice {i+1}/{n} | alpha={alpha:.2f}", fontsize=10)

        handles = []
        for c in range(1, num_classes):
            rgba = cmap(c)
            handles.append(mpatches.Patch(color=rgba, label=f"{c}: {class_names[c]}"))
        if handles:
            ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.7, borderpad=0.4, handlelength=1.2)

        canvas.draw_idle()

    def on_axis_change(*_):
        update_slider_range()
        redraw()

    axis_box.bind("<<ComboboxSelected>>", on_axis_change)
    slice_scale.configure(command=lambda v: redraw())
    alpha_scale.configure(command=lambda v: redraw())
    chk_pred.configure(command=redraw)
    chk_gt.configure(command=redraw)

    def on_entry_return(_evt):
        axis = ax_var.get()
        n = axis_len(ct01, axis)
        try:
            i = int(idx_var.get())
        except Exception:
            i = 0
        i = int(np.clip(i, 0, n - 1))
        idx_var.set(i)
        slice_scale.set(i)
        redraw()

    idx_entry.bind("<Return>", on_entry_return)

    update_slider_range()
    redraw()
    root.mainloop()


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to best_epoch_XXX.pt")
    ap.add_argument("--ct", required=True, help="path to CT .nrrd")
    ap.add_argument("--lab", default=None, help="optional GT labelmap .nrrd")
    ap.add_argument("--roi", default="128,128,128", help="roi Z,Y,X for sliding window, e.g. 128,128,128")
    ap.add_argument("--overlap", type=float, default=0.5, help="sliding window overlap [0..1]")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument("--outdir", default=None, help="optional output dir to save pred.nrrd")
    args = ap.parse_args()

    roi_zyx = tuple(int(x) for x in args.roi.split(","))

    print(f"[INFO] Loading CT: {args.ct}")
    ct = load_nrrd_zyx(args.ct)

    gt = None
    if args.lab:
        print(f"[INFO] Loading GT: {args.lab}")
        gt = load_nrrd_zyx(args.lab).astype(np.uint8)

    pred, ct01, cfg, num_classes, maxp = infer_volume(
        ct_zyx=ct,
        ckpt_path=args.ckpt,
        roi_zyx=roi_zyx,
        overlap=args.overlap,
        force_cpu=args.cpu,
    )

    print(f"[OK] pred shape: {pred.shape} | maxP={maxp:.3f}")

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        out_pred = os.path.join(args.outdir, "pred.nrrd")
        save_nrrd_zyx(out_pred, pred)
        print(f"[OK] saved pred -> {out_pred}")

    class_names = get_class_names(num_classes)
    run_gui(ct01, pred, gt, class_names)


if __name__ == "__main__":
    main()

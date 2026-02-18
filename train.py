import argparse
from src.pelvis_seg.config import CFG
from src.pelvis_seg.train_loop import run_training

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "femur"], default="full",
                    help="full=0..3 (sacrum/hip/femur), femur=0..1 (bg/femur)")
    args = ap.parse_args()

    cfg = CFG()
    cfg.MODE = args.mode
    run_training(cfg)

if __name__ == "__main__":
    main()
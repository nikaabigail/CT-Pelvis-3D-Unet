import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pelvis_seg.config import CFG
from pelvis_seg.train_loop import run_training

if __name__ == "__main__":
    cfg = CFG()
    run_training(cfg)

from dataclasses import dataclass, asdict
import json
from pathlib import Path

@dataclass
class CFG:
    ROOT_DIR: str = r"D:\Dic\MONAI_dataset"
    CT_DIR: str   = r"D:\Dic\CT_non_format"
    LAB_DIR: str  = r"D:\Dic\Label_map_FIXED"

    NUM_CLASSES: int = 6
    EXPECTED_LABELS: tuple = (0, 1, 2, 3, 4, 5)

    VAL_NUM: int = 6
    SEED: int = 42
    SPLIT_TRIES: int = 200

    MAX_EPOCHS: int = 50
    LR: float = 1e-4

    BATCH_SIZE: int = 1
    NUM_WORKERS: int = 0
    PATCH_SIZE: tuple = (128, 128, 128)  # (Z,Y,X)

    A_MIN: int = -1000
    A_MAX: int = 3000

    POS_PROB: float = 0.90
    POS_TRIES: int = 30
    MIN_FG_VOXELS_IN_PATCH: int = 200
    MIN_TARGET_VOXELS_IN_PATCH: int = 2000

    CLASS_BALANCED_SAMPLING: bool = True
    FG_CLASS_PROBS_1_5: tuple = (0.20, 0.15, 0.15, 0.25, 0.25)

    ROT90_PROB: float = 0.0
    FLIP_X_PROB: float = 0.0
    FLIP_Y_PROB: float = 0.0
    FLIP_Z_PROB: float = 0.0

    USE_COORDS: bool = True

    INCLUDE_BACKGROUND_DICE: bool = False
    CE_CLIP: tuple = (0.5, 8.0)
    BG_WEIGHT: float = 0.7
    BOOST_CLASSES_4_5: float = 2.0

    FAST_VAL_ENABLE: bool = True
    FAST_VAL_PATCHES_PER_CASE: int = 2
    FAST_VAL_POS_PROB: float = 0.95
    FAST_VAL_MIN_TARGET_VOX: int = 1000
    FAST_VAL_EVERY_EPOCH: int = 1

    PRINT_EVERY: int = 5

def save_config(cfg: CFG, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

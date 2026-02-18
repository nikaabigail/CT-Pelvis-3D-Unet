# src/pelvis_seg/config.py
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class CFG:
    # ----------------------------
    # Пути и домены данных
    # ----------------------------

    # Основной датасет (SMIR / Pelvis-Thighs)
    ROOT_DIR: str = r"D:\Dic\MONAI_dataset"
    CT_DIR: str   = r"D:\Dic\CT_non_format"
    LAB_DIR: str  = r"D:\Dic\Label_map_FIXED"

    # Дополнительный домен (CTPelvic1K dataset6)
    USE_DATASET6: bool = True
    DATASET6_CT_DIR: str  = r"D:\Dic\CTPelvic1K_dataset6_nrrd_raw\ct_nrrd"
    DATASET6_LAB_DIR: str = r"D:\Dic\CTPelvic1K_dataset6_nrrd_raw\seg_nrrd"

    # Доп. домен (dataset8)
    USE_DATASET8: bool = True
    DATASET8_CT_DIR: str  = r"D:\Dic\dataset8_nrrd\ct"
    DATASET8_LAB_DIR: str = r"D:\Dic\dataset8_nrrd\label"

    DATASET8_SAMPLE_RATIO: float = 1.0
    DATASET8_MAX_CASES: int | None = None

    # Валидация только на SMIR
    VAL_DOMAIN: str = "smir"

    # ----------------------------
    # Контроль баланса доменов
    # ----------------------------
    DATASET6_SAMPLE_RATIO: float = 0.5
    DATASET6_MAX_CASES: int | None = None

    # ----------------------------
    # Семантика классов (ВАРИАНТ A: ignore L/R)
    # ----------------------------
    # Итоговая схема разметки внутри пайплайна:
    # 0 = BG
    # 1 = Sacrum
    # 2 = Hip (L+R)
    # 3 = Femur (L+R)
    #
    # dataset6: femur виден на CT, но НЕ размечен -> в GT это BG (нормально)
    IGNORE_LR: bool = True
    NUM_CLASSES: int = 4
    EXPECTED_LABELS: tuple = (0, 1, 2, 3)

    # ----------------------------
    # Train / Val split
    # ----------------------------
    VAL_NUM: int = 6
    SEED: int = 42
    SPLIT_TRIES: int = 200

    # ----------------------------
    # Оптимизация
    # ----------------------------
    MAX_EPOCHS: int = 120
    LR: float = 1e-4

    BATCH_SIZE: int = 1
    NUM_WORKERS: int = 0
    PATCH_SIZE: tuple = (128, 128, 128)  # (Z,Y,X)

    # ----------------------------
    # CT preprocessing
    # ----------------------------
    A_MIN: int = -1000
    A_MAX: int = 3000

    # ----------------------------
    # Sampling стратегии
    # ----------------------------
    POS_PROB: float = 0.90
    POS_TRIES: int = 30
    MIN_FG_VOXELS_IN_PATCH: int = 200
    MIN_TARGET_VOXELS_IN_PATCH: int = 2000

    # Баланс классов для FG (1..3) (BG исключен)
    # порядок: (sacrum, hip, femur)
    CLASS_BALANCED_SAMPLING: bool = True
    FG_CLASS_PROBS_1_3: tuple = (0.25, 0.35, 0.40)

    # ----------------------------
    # Аугментации
    # ----------------------------
    ROT90_PROB: float = 0.0
    FLIP_X_PROB: float = 0.0
    FLIP_Y_PROB: float = 0.0
    FLIP_Z_PROB: float = 0.0

    # ----------------------------
    # CoordConv
    # ----------------------------
    USE_COORDS: bool = True

    # ----------------------------
    # Loss
    # ----------------------------
    INCLUDE_BACKGROUND_DICE: bool = False
    CE_CLIP: tuple = (0.5, 8.0)
    BG_WEIGHT: float = 0.7

    # Усиление класса femur (класс 3) — опционально
    BOOST_FEMUR: float = 1.0

    # ----------------------------
    # FAST VALIDATION
    # ----------------------------
    FAST_VAL_ENABLE: bool = True
    FAST_VAL_PATCHES_PER_CASE: int = 2
    FAST_VAL_POS_PROB: float = 0.95
    FAST_VAL_MIN_TARGET_VOX: int = 1000
    FAST_VAL_EVERY_EPOCH: int = 1

    # ----------------------------
    # Логи
    # ----------------------------
    PRINT_EVERY: int = 5
    # ----------------------------
    # Training mode
    # ----------------------------
    TRAIN_MODE: str = "full"  # "full" | "femur"

    # Для femur-only: какие исходные id считаем femur в каждом домене
    # smir: 4=femur_R, 5=femur_L
    FEMUR_IDS_SMIR: tuple = (4, 5)

    # dataset8: Segment_1 femur_L, Segment_2 femur_R, Segment_3 hip_L, Segment_4 hip_R, Segment_5 sacrum
    FEMUR_IDS_DATASET8: tuple = (1, 2)

    # dataset6 femur НЕ размечен -> игнорируем домен в femur-only
    MODE: str = "full"  # full|femur

    # raw ids femur в исходных датасетах (до remap)
    FEMUR_IDS_SMIR: tuple = (4, 5)       # SMIR: 4=femR,5=femL
    FEMUR_IDS_DATASET8: tuple = (1, 2)   # dataset8: 1=femL,2=femR

def save_config(cfg: CFG, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

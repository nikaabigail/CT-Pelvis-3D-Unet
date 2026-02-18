# optuna_tune.py
import os
from pathlib import Path
import optuna

from src.pelvis_seg.config import CFG
from src.pelvis_seg.train_loop import run_training


def make_cfg_for_trial(trial: optuna.Trial) -> CFG:
    cfg = CFG()

    # ---- ВАЖНО: уменьшай epochs для тюнинга, потом доучишь ----
    cfg.MAX_EPOCHS = trial.suggest_int("MAX_EPOCHS", 8, 20)  # для быстрых прогонов
    cfg.LR = trial.suggest_float("LR", 3e-5, 3e-4, log=True)

    # sampling pressure
    cfg.POS_PROB = trial.suggest_float("POS_PROB", 0.75, 0.98)
    cfg.POS_TRIES = trial.suggest_int("POS_TRIES", 10, 50)
    cfg.MIN_FG_VOXELS_IN_PATCH = trial.suggest_int("MIN_FG_VOXELS", 50, 800)
    cfg.MIN_TARGET_VOXELS_IN_PATCH = trial.suggest_int("MIN_TARGET_VOXELS", 500, 6000)

    # loss weights
    cfg.BG_WEIGHT = trial.suggest_float("BG_WEIGHT", 0.3, 1.2)
    cfg.BOOST_CLASSES_4_5 = trial.suggest_float("BOOST_4_5", 1.0, 4.0)

    # domain mixing (анатомический regularizer)
    cfg.DATASET6_SAMPLE_RATIO = trial.suggest_float("DATASET6_SAMPLE_RATIO", 0.0, 1.0)

    # class probs (если ты сейчас в режиме ignore_lr и NUM_CLASSES=4,
    # то FG_CLASS_PROBS_1_5 может быть уже не актуален.
    # Но если sampler все равно использует этот prior через "мост" — оставляем.
    p_sac = trial.suggest_float("p_sac", 0.1, 0.5)
    p_hipR = trial.suggest_float("p_hipR", 0.05, 0.35)
    p_hipL = trial.suggest_float("p_hipL", 0.05, 0.35)
    p_femR = trial.suggest_float("p_femR", 0.05, 0.5)
    p_femL = trial.suggest_float("p_femL", 0.05, 0.5)
    cfg.FG_CLASS_PROBS_1_5 = (p_sac, p_hipR, p_hipL, p_femR, p_femL)

    # FAST-VAL (метрика быстрее и стабильнее если брать 2-4 патча)
    cfg.FAST_VAL_ENABLE = True
    cfg.FAST_VAL_EVERY_EPOCH = 1
    cfg.FAST_VAL_PATCHES_PER_CASE = trial.suggest_int("FAST_VAL_PATCHES_PER_CASE", 1, 4)

    # фиксируем вал домен
    cfg.VAL_DOMAIN = "smir"

    # уникальная папка эксперимента под trial
    base = Path(cfg.ROOT_DIR)
    cfg.ROOT_DIR = str(base / "optuna_runs" / f"trial_{trial.number:04d}")

    # чтобы сплит был разный — можно менять SEED
    cfg.SEED = 42 + trial.number

    return cfg


def objective(trial: optuna.Trial) -> float:
    cfg = make_cfg_for_trial(trial)

    # если хочешь логировать параметры в trial.user_attrs — можно так
    trial.set_user_attr("root_dir", cfg.ROOT_DIR)

    best_fg = run_training(cfg)

    # Optuna хочет float, и лучше чтобы nan не ломал
    if best_fg != best_fg:  # nan check
        return 0.0
    return float(best_fg)


def main():
    storage = "sqlite:///optuna_pelvis.db"  # чтобы можно было прерывать/продолжать
    study = optuna.create_study(
        study_name="pelvis_seg",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )

    # число прогонов
    study.optimize(objective, n_trials=30)

    print("\n=== BEST TRIAL ===")
    print("value:", study.best_value)
    print("params:", study.best_params)


if __name__ == "__main__":
    main()

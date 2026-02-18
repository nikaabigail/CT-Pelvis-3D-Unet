# CT-Pelvis-3D-Unet

Проект для обучения 3D U-Net (MONAI) на CT в формате NRRD.
Поддерживаются два сценария обучения:
- **общая сегментация таза** (`full`) — несколько анатомических классов,
- **только femur** (`femur`) — бинарная сегментация (фон / бедренные кости).

## Что делает проект
- Загружает пары `CT + labelmap` из нескольких доменов.
- Выполняет препроцессинг, патч-сэмплинг и обучение 3D U-Net.
- Поддерживает быстрый валидационный проход (`fast val`) во время тренировки.
- Позволяет тюнить гиперпараметры через Optuna.

## Структура репозитория
- `train.py` — основная точка запуска обучения (режимы `full` / `femur`).
- `infer.py` — точка запуска инференса/валидации (сейчас заготовка).
- `optuna_tune.py` — подбор гиперпараметров через Optuna.
- `configs/default.yaml` — YAML-конфиг (можно расширять под свои эксперименты).
- `src/pelvis_seg/` — основная логика:
  - `dataset.py`, `preprocess.py` — подготовка данных,
  - `model.py` — модель,
  - `losses.py`, `metrics.py` — функции потерь и метрики,
  - `train_loop.py`, `fast_val.py` — обучение и быстрая валидация,
  - `io_nrrd.py`, `utils.py` — I/O и утилиты.

## Датасеты
Ниже перечислены домены, которые используются в коде (пути задаются в `src/pelvis_seg/config.py`):

1. **SMIR / Pelvis-Thighs** (основной датасет)
   - CT: `CT_DIR`
   - Label: `LAB_DIR`
   - Ссылка: **TODO: добавить ссылку**

2. **CTPelvic1K dataset6** (дополнительный домен)
   - CT: `DATASET6_CT_DIR`
   - Label: `DATASET6_LAB_DIR`
   - Ссылка: **TODO: добавить ссылку**

3. **dataset8** (дополнительный домен)
   - CT: `DATASET8_CT_DIR`
   - Label: `DATASET8_LAB_DIR`
   - Ссылка: **TODO: добавить ссылку**

> Примечание: в `femur`-режиме домен dataset6 обычно не используется как источник positive femur-разметки (это отражено в комментариях/логике конфигурации).

## Требования
- Python 3.10+
- CUDA-совместимый GPU (желательно для практического времени обучения)

## Установка
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Настройка перед запуском
1. Откройте `src/pelvis_seg/config.py`.
2. Проверьте и измените пути к данным:
   - `ROOT_DIR`
   - `CT_DIR`, `LAB_DIR`
   - `DATASET6_CT_DIR`, `DATASET6_LAB_DIR`
   - `DATASET8_CT_DIR`, `DATASET8_LAB_DIR`
3. При необходимости скорректируйте:
   - `VAL_NUM`, `MAX_EPOCHS`, `PATCH_SIZE`, `LR`
   - баланс доменов (`DATASET6_SAMPLE_RATIO`, `DATASET8_SAMPLE_RATIO`)

## Запуск обучения
### 1) Общий режим (multi-class pelvis)
```bash
python train.py --mode full
```

### 2) Режим только для femur
```bash
python train.py --mode femur
```

## Запуск Optuna-тюнинга
```bash
python optuna_tune.py
```

## Логи и артефакты
- Рекомендуется хранить артефакты обучения в отдельных директориях (`runs/`, `logs/`, `checkpoints/`).
- Большие данные и веса моделей не должны попадать в git.

## Планы по README
- Добавить рабочие ссылки на датасеты в секцию **Датасеты**.
- Добавить пример инференса после финализации `infer.py`.

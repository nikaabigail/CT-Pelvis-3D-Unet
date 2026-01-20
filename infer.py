# infer.py
# Единая точка запуска инференса/валидации (пока заглушка).
# Аналогично train.py добавляем src/ в sys.path.

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main():
    print("[infer.py] Пока заглушка. Подключим сюда sliding-window infer/val.")
    print("Подумай: нужно ли сохранять pred .nrrd с метаданными CT (origin/spacing/direction) под Slicer.")


if __name__ == "__main__":
    main()

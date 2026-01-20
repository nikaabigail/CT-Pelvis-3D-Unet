# Val_test_sliding

Проект для обучения 3D U-Net (MONAI) на CT NRRD + фиксированных labelmap {0..5}.
Код логически разнесен по модулям в `src/pelvis_seg/`, запуск — через `train.py`.

## Структура
- `train.py` — единая точка запуска обучения
- `infer.py` — (опционально) единая точка инференса/валидации
- `src/pelvis_seg/` — пакет с логикой (датасет, сэмплинг, модель, лосс, fast-val и т.д.)

## Установка
```bash
pip install -r requirements.txt

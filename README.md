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

## Датасет
CT_non_format — https://drive.google.com/drive/folders/1r11n_45PQvrHqbGHlfpvQ6e9ySL3mpmC?usp=sharing
Label_map_FIXED — https://drive.google.com/drive/folders/1VjPFBKW4_yBc-hxCu77LdHIYp_4fdwqx?usp=sharing

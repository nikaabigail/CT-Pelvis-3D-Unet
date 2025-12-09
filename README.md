CT Pelvis Segmentation — 3D UNet (MONAI)



Этот проект содержит полный рабочий пайплайн для подготовки датасета и обучения 3D-сегментационной модели (UNet 3D) на базе MONAI для разметки органов таза по компьютерной томографии (КТ).



Структура проекта:



CT\_monai/

│── preprocess/

│     ├── convert\_dataset.py      # Конвертация исходных данных в формат MONAI

│     ├── fix\_masks.py            # Исправление некорректных масок (4D → 3D)

│     ├── mapping.csv             # Таблица соответствия CT ↔ Mask

│

│── train/

│     ├── CT\_Pelvis\_3DUnet.py     # Скрипт обучения 3D UNet

│

│── requirements.txt              # Список зависимостей



Установка зависимостей

pip install -r requirements.txt



1\. Подготовка данных

1.1. Конвертация датасета



Скрипт создает стандартную структуру:



MONAI\_dataset/

&nbsp;   images/

&nbsp;   labels/





Запуск:



python preprocess/convert\_dataset.py





Входные данные берутся из mapping.csv, где указано:



CT File	Segmentation File

path/to/ct.nii.gz	path/to/mask.nii.gz

1.2. Исправление масок



Некоторые маски оказываются не в 3D, а в 4D формате (например (X, Y, Z, 1, 2)), что ломает обучение.

Этот скрипт приводит маски к нормальному 3D виду:



python preprocess/fix\_masks.py





После запуска все маски гарантированно продаются в формате:



(depth, height, width)



2\. Обучение 3D UNet



Основной обучающий скрипт:



python train/CT\_Pelvis\_3DUnet.py





Функциональность:



читает подготовленный датасет



автоматически делит набор на train/val



обучает 3D-UNet (MONAI)



считает Dice-метрику



применяет sliding window inference



сохраняет лучшую модель:



best\_unet.pth



Требования к GPU



Обучение 3D UNet требует много видеопамяти.



Рекомендации:



GPU	Настройки

8 GB	уменьшить patch\_size и channels

12 GB	работает со средними настройками

16–24 GB	рекомендуется для стабильного обучения



Если появляется ошибка Out Of Memory, можно снизить:



patch\_size (например 96×96×96)



roi\_size в sliding window



количество фильтров channels=(16,32,64,...)



Пример структуры выходных данных



После подготовки данных:



MONAI\_dataset/

│── images/

│     ├── case\_001\_image.nii.gz

│     ├── case\_002\_image.nii.gz

│     └── ...

│

│── labels/

&nbsp;     ├── case\_001\_label.nii.gz

&nbsp;     ├── case\_002\_label.nii.gz

&nbsp;     └── ...



Автор



Владимир Барышев

Проект создан для задач медицинской сегментации и исследования моделей 3D-нейросетей.


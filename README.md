# Gesture Recognition Web App

Веб-приложение для распознавания жестов русского жестового языка с камеры пользователя. Система извлекает ключевые точки MediaPipe, классифицирует жесты нейросетевой моделью и преобразует поток распознанных слов в связный русский текст с помощью отдельной seq2seq-модели.

## Возможности

- распознавание жестов слов по holistic-точкам;
- распознавание дактильных букв по точкам рук;
- вывод двух текстовых состояний: готовый нормализованный текст и сырые распознанные жесты;
- сегментация потока жестов на короткие фрагменты без отправки всей истории в NLP-модель;
- обучение модели нормализации gloss-to-text на JSONL-датасете;
- отдельные модули препроцессинга и обучения моделей жестов.

## Структура

```text
Web_app/
  backend/              FastAPI backend
  frontend/             HTML, CSS, JS клиент
  imgs/                 изображения интерфейса
  model/                место для весов и label-файлов
  tools/                генерация локального HTTPS-сертификата
Gloss_to_Text/
  Dataset/data.jsonl    датасет для обучения NLP-модели
  Train/train.py        обучение ruT5/mT5 gloss-to-text
Models/
  preprocess/           общий препроцессинг SLOVO в keypoints_out
  preprocess_slovo.py   общий запуск препроцессинга для words/hands
  Model_holistic/       обучение модели слов EfficientNetV2-S 384 из Neuro_Kall
  Model_hands/          обучение модели дактиля
requirements.txt
```

## Датасеты

Большие датасеты жестов не добавлены в репозиторий, чтобы не загружать GitHub тяжёлыми видео и NPZ-файлами.

- SLOVO, датасет слов русского жестового языка: https://github.com/hukenovs/slovo
- Bukva, датасет дактильной азбуки РЖЯ: https://github.com/ai-forever/bukva

После загрузки датасеты должны быть размещены в корне проекта в папке `Dataset/slovo/` или адаптированы под пути в конфигурациях препроцессинга.
В текущей GitHub-версии этой папки с видео нет: препроцессинг ожидает, что пользователь скачает датасет отдельно и положит его в `Dataset/slovo/`.

NLP-датасет для преобразования последовательностей жестовых слов в русский текст включён в проект:

```text
Gloss_to_Text/Dataset/data.jsonl
```

## Модели

Веса рабочих моделей включены в структуру проекта:

```text
Web_app/model/holistic/holistic_model.pt
Web_app/model/hands/hands_model.pt
Web_app/model/rut5-small-gloss/model.safetensors
```

Также включён checkpoint модели слов EfficientNetV2-S:

```text
Models/Model_holistic/checkpoints_effnetv2_s_384/effnetv2_s_384_best_val_top1.pt
```

Также включён checkpoint модели дактиля:

```text
Models/Model_hands/checkpoints/effnetb0_best_val_top1_hands_letters_03.pt
```

Сохранённая NLP-модель для gloss-to-text лежит отдельно от веб-приложения:

```text
Gloss_to_Text/Model/rut5-small-gloss/model.safetensors
```

Для повторного обучения модели слов также включены локальные ImageNet-веса EfficientNetV2-S:

```text
Models/Model_holistic/pretrained/efficientnet_v2_s-dd5fe13b.pth
```

Файлы `labels.json` для gesture-моделей и tokenizer/config-файлы для NLP-модели находятся рядом с весами.

## Установка

Создайте виртуальное окружение в корне проекта:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Файл зависимостей в проекте один и находится в корне репозитория.

## Запуск веб-приложения

Сначала создайте локальный HTTPS-сертификат. Вместо `<IP_ПК>` укажите IP компьютера в локальной сети:

```powershell
cd Web_app
python tools\generate_dev_cert.py --hosts <IP_ПК>,localhost,127.0.0.1
```

Запуск:

```powershell
cd ..
powershell -ExecutionPolicy Bypass -File .\Web_app\run_https.ps1
```

Обычно приложение будет доступно по адресу:

```text
https://localhost:8443
```

## Обучение NLP-модели

```powershell
python Gloss_to_Text\Train\train.py
```

После обучения сохранённую модель можно перенести в:

```text
Web_app/model/rut5-small-gloss/
```

## Обучение моделей жестов

Общий препроцессинг лежит в `Models/preprocess` и используется для подготовки данных обеих моделей.

Подготовка ключевых точек для модели слов:

```powershell
python Models\preprocess_slovo.py --mode holistic
```

Результат сохраняется в:

```text
Models/Model_holistic/keypoints_out/
```

Ожидаемая структура:

```text
keypoints_out/
  labels.json
  splits.csv
  samples/*.npz
```

Подготовка ключевых точек для модели дактиля:

```powershell
python Models\preprocess_slovo.py --mode hands
```

Результат сохраняется в:

```text
Models/Model_hands/keypoints_out/
```

Запуск обучения модели слов:

```powershell
python Models\Model_holistic\scripts\train_effnetv2_s_384.py
```

Обучение модели дактиля:

```powershell
python Models\Model_hands\scripts\train_effnet.py
```

После обучения новые веса нужно положить в папки `Web_app/model/holistic/` и `Web_app/model/hands/` с именами, указанными выше.

## Как работает преобразование слов в предложения

Backend хранит поток распознанных жестов в буфере. Повторяющиеся жесты фильтруются по cooldown, дактильные буквы не отправляются в NLP-модель и добавляются в итоговый текст отдельно. Обычные слова накапливаются в короткий сегмент. Сегмент закрывается по маркеру новой мысли, новому субъекту, финальной фразе, максимальной длине или явному flush при завершении распознавания.

Каждый завершённый сегмент передаётся в seq2seq-модель `rut5-small-gloss`, которая преобразует последовательность классов жестов в нормальное русское предложение. В интерфейсе отдельно показываются сырые жесты и готовый текст.

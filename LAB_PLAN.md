# Docker + GOT-OCR-2.0 + Streamlit

## Информация о модели

**Модель:** [stepfun-ai/GOT-OCR-2.0-hf](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)

| Параметр | Значение |
|----------|----------|
| Размер | ~1 GB |
| Параметров | 0.6B |
| Тип | Image-Text-to-Text |
| Библиотека | transformers (HuggingFace) |
| Точность | BF16 |
| CPU поддержка | Да (медленно) |

### Возможности модели
- Plain text OCR (документы, скриншоты)
- Scene text OCR
- Formatted document OCR (markdown, LaTeX)
- OCR регионов по координатам или цвету
- Multi-page processing
- Sheet music, таблицы, формулы

## Архитектура решения

```
got-ocr-lab/
├── app.py              # Streamlit приложение
├── requirements.txt    # dependencies
├── Dockerfile          # образ с моделью
├── docker-compose.yml  # запуск
├── test_data/
│   └── sample.json     # ожидаемый вывод для теста
└── sample_images/
    └── test_image.jpg  # тестовое изображение
```

## Шаги реализации

### 1. Streamlit приложение (app.py)

Два режима:
1. **Demo mode** — загружена тестовая картинка + ожидаемый вывод из JSON
2. **Custom mode** — загрузка своей картинки

Минимальный UI:
- Заголовок
- Две вкладки: "Demo" и "Upload"
- Вывод распознанного текста

### 2. Зависимости (requirements.txt)

```
streamlit
torch
transformers
Pillow
```

### 3. Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Модель скачивается при первом запуске
ENV MODEL_ID=stepfun-ai/GOT-OCR-2.0-hf

COPY app.py .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

### 4. docker-compose.yml

```yaml
services:
  app:
    build: .
    ports:
      - "8501:8501"
    # CPU-only запуск
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

### 5. Тестовые данные

**sample.json:**
```json
{
  "image": "test_image.jpg",
  "expected_output": "R&D QUALITY IMPROVEMENT\nSUGGESTION/SOLUTION FORM"
}
```

**test_image.jpg** — изображение с текстом для валидации

## Запуск

```bash
docker compose up --build
# открыть http://localhost:8501
```

## Критерии сдачи

| # | Критерий |
|---|----------|
| 1 | Docker образ собирается без ошибок |
| 2 | `docker compose up` запускает приложение |
| 3 | Streamlit доступен на localhost:8501 |
| 4 | Demo mode показывает тестовое изображение и вывод |
| 5 | Upload mode позволяет загрузить своё изображение |
| 6 | Модель корректно распознаёт текст (проверка на test_image) |


Модель будет скачиваться при **первом** запуске контейнера с HuggingFace Hub. Это позволяет:
- Уменьшить размер образа
- Не хранить 1GB в образе
- Использовать latest версию модели

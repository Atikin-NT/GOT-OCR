FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY sample_images/ ./sample_images/

ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_TOKEN=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

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port=8501"]
# Usamos Python 3.11 que tiene mejor soporte para TensorFlow en ARM
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="-1" \
    TF_CPP_MIN_LOG_LEVEL="3"

WORKDIR /app

# Instalamos libgl1 (el fix de antes) y hdf5 (necesario a veces para TF en ARM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Aumentamos el timeout porque compilar en ARM puede ser lento
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
# Usamos una imagen base ligera de Python 3.10
FROM python:3.10-slim

# Evita que Python genere archivos .pyc y fuerza logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Desactiva GPU explícitamente para ahorrar memoria/drivers
    CUDA_VISIBLE_DEVICES="-1" \
    # Limpia logs basura de TF
    TF_CPP_MIN_LOG_LEVEL="3"

WORKDIR /app

# Instalar dependencias de sistema mínimas (para Pillow/OpenCV si fuera necesario)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python (Cacheada)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la app
COPY app.py .

# Exponer el puerto (Coolify detectará esto)
EXPOSE 8000

# Comando de arranque profesional
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
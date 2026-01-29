import os
import sys
import time
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
from tensorflow import keras
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Configuración del Modelo
# OJO: En Coolify montaremos el volumen aquí
MODEL_FOLDER = "/app/models"
MODEL_FILENAME = "model_production.keras"
MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_FILENAME)

IMG_SIZE = 224
CLASS_NAMES = ['Dog', 'Automobile', 'Bird']

app = FastAPI(title="SOTA Production API", version="2.0.0")
model = None

@app.on_event("startup")
async def startup_event():
    global model
    print(f"🚀 INICIANDO API EN ENTORNO COOLIFY")
    print(f"📂 Buscando modelo en: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: No encuentro el archivo {MODEL_FILENAME} en {MODEL_FOLDER}")
        print("💡 TIP: Asegúrate de montar el volumen en Coolify en '/app/models'")
        # No matamos la app para que Coolify no entre en bucle de reinicios, 
        # pero la API responderá errores 503.
        return

    try:
        print("⏳ Cargando modelo en RAM (CPU)...")
        model = keras.models.load_model(MODEL_PATH)
        
        # Warmup
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3))
        model.predict(dummy, verbose=0)
        print("✅ ¡SISTEMA ONLINE Y LISTO!")
    except Exception as e:
        print(f"❌ ERROR CRÍTICO CARGANDO MODELO: {e}")

def process_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    
    # 1. Convertir a float
    img_array = np.array(img).astype(np.float32)
    # 2. Normalizar de 0-255 a 0-1
    img_array = img_array / 255.0  
    
    return np.expand_dims(img_array, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado o no se encontró el archivo.")
    
    try:
        contents = await file.read()
        processed = process_image(contents)
        
        start = time.time()
        preds = model.predict(processed, verbose=0)[0]
        inference_time = (time.time() - start) * 1000
        
        results = {name: float(prob) for name, prob in zip(CLASS_NAMES, preds)}
        best_idx = np.argmax(preds)
        
        return {
            "prediction": CLASS_NAMES[best_idx],
            "confidence": float(preds[best_idx]),
            "time_ms": round(inference_time, 2),
            "probabilities": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# SERVIR FRONTEND
@app.get("/")
async def read_index():
    return FileResponse('index.html')
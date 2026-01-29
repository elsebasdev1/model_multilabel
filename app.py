import os
import sys
import time
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import tensorflow as tf
from tensorflow import keras

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
MODEL_FOLDER = "/app/models"
MODEL_FILENAME = "model_production.keras"
MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_FILENAME)

IMG_SIZE = 224

# ORDEN DE CLASES CORRECTO (Basado en Notebook 01: Dog=0, Auto=1, Bird=2)
CLASS_NAMES = ['Dog', 'Automobile', 'Bird']

app = FastAPI(title="SOTA Production API", version="Fixed.Final")
model = None

# ============================================================================
# 1. CARGA DEL MODELO (STARTUP)
# ============================================================================
@app.on_event("startup")
async def startup_event():
    global model
    print(f"🚀 INICIANDO API (SOTA FIX)")
    print(f"📂 Buscando modelo en: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: No encuentro {MODEL_FILENAME} en {MODEL_FOLDER}")
        return

    try:
        print("⏳ Cargando modelo en RAM...")
        model = keras.models.load_model(MODEL_PATH)
        
        # Warmup
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3))
        model.predict(dummy, verbose=0)
        print("✅ ¡MODELO CARGADO Y LISTO! (Esperando inputs 0-255)")
    except Exception as e:
        print(f"❌ ERROR CRÍTICO: {e}")

# ============================================================================
# 2. PROCESAMIENTO (LA CORRECCIÓN ESTÁ AQUÍ) 🛠️
# ============================================================================
def process_image(image_bytes):
    # 1. Abrir como RGB
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 2. Resize Bicúbico (Alta calidad)
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    
    # 3. Convertir a Array Float32
    img_array = np.array(img).astype(np.float32)
    
    # 🛑 IMPORTANTE: NO NORMALIZAMOS (NO / 255.0) 🛑
    # El modelo fue entrenado con 'include_preprocessing=True', 
    # por lo que espera valores RAW de 0 a 255.
    
    # 4. Expandir dimensiones (Batch size = 1)
    return np.expand_dims(img_array, axis=0)

# ============================================================================
# 3. ENDPOINT DE PREDICCIÓN
# ============================================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        contents = await file.read()
        processed = process_image(contents)
        
        start = time.time()
        # Inferencia
        preds = model.predict(processed, verbose=0)[0]
        inference_time = (time.time() - start) * 1000
        
        # Mapeo de probabilidades
        results = {name: float(prob) for name, prob in zip(CLASS_NAMES, preds)}
        
        # Obtener mejor clase
        best_idx = np.argmax(preds)
        
        return {
            "prediction": CLASS_NAMES[best_idx],
            "confidence": float(preds[best_idx]),
            "time_ms": round(inference_time, 2),
            "probabilities": results
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Error interno")

# ============================================================================
# 4. FRONTEND
# ============================================================================
@app.get("/")
async def read_index():
    return FileResponse('index.html')
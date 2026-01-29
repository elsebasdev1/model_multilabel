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
from tensorflow.keras.applications.convnext import preprocess_input # Preprocesador OFICIAL

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
MODEL_FOLDER = "/app/models"
MODEL_FILENAME = "model_production.keras"
MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_FILENAME)

IMG_SIZE = 224
CLASS_NAMES = ['Dog', 'Automobile', 'Bird']
THRESHOLD = 0.50  # Si supera el 50%, se considera presente

app = FastAPI(title="SOTA Multi-Label API", version="3.0.0")
model = None

# ============================================================================
# STARTUP
# ============================================================================
@app.on_event("startup")
async def startup_event():
    global model
    print(f"🚀 INICIANDO API MULTI-LABEL REAL")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: No encuentro {MODEL_FILENAME}")
        return

    try:
        print("⏳ Cargando modelo...")
        model = keras.models.load_model(MODEL_PATH)
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3))
        model.predict(dummy, verbose=0)
        print("✅ SISTEMA LISTO (Sigmoid Activation Active)")
    except Exception as e:
        print(f"❌ ERROR CRÍTICO: {e}")

# ============================================================================
# PROCESAMIENTO (RAW INPUT 0-255)
# ============================================================================
def process_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    img_array = np.array(img).astype(np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    # Usamos el preprocesador de Keras para asegurar que coincida con el training
    return preprocess_input(img_batch)

# ============================================================================
# ENDPOINT (LÓGICA MULTI-LABEL)
# ============================================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        contents = await file.read()
        processed = process_image(contents)
        
        start = time.time()
        # Predicción (Array de 3 floats independientes, ej: [0.98, 0.85, 0.01])
        preds = model.predict(processed, verbose=0)[0]
        inference_time = (time.time() - start) * 1000
        
        # 1. Mapear resultados
        results = {name: float(prob) for name, prob in zip(CLASS_NAMES, preds)}
        
        # 2. LÓGICA MULTI-LABEL REAL
        # Filtramos TODO lo que supere el umbral
        detected_objects = [name for name, prob in results.items() if prob >= THRESHOLD]
        
        # Caso borde: Si nada supera el 50%, devolvemos la clase con mayor probabilidad
        # pero indicamos que es "Incierto"
        primary_prediction = detected_objects if detected_objects else [CLASS_NAMES[np.argmax(preds)]]
        
        return {
            "prediction": primary_prediction, # Ahora es una LISTA (ej: ["Dog", "Automobile"])
            "is_multilabel": len(detected_objects) > 1,
            "time_ms": round(inference_time, 2),
            "probabilities": results # Aquí van los porcentajes crudos
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Error interno")

@app.get("/")
async def read_index():
    return FileResponse('index.html')
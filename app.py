import os
import sys
import time
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.convnext import preprocess_input

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
MODEL_FOLDER = "/app/models"
# Definimos los dos archivos
MODEL_FILES = {
    "standard": "model_production.keras",      # El original (99% en CIFAR)
    "hd": "model_production_hd.keras"          # El adaptado (Mejor en fotos reales)
}
IMG_SIZE = 224
CLASS_NAMES = ['Dog', 'Automobile', 'Bird']

app = FastAPI(title="SOTA Dual API", version="5.0.0")

# Diccionario para mantener los modelos en RAM
loaded_models = {}

@app.on_event("startup")
async def startup_event():
    print(f"🚀 INICIANDO API DUAL (STANDARD + HD)")
    
    for key, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_FOLDER, filename)
        if os.path.exists(path):
            try:
                print(f"⏳ Cargando modelo {key.upper()} desde {filename}...")
                loaded_models[key] = keras.models.load_model(path)
                # Warmup
                loaded_models[key].predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 3)), verbose=0)
                print(f"✅ Modelo {key.upper()} LISTO.")
            except Exception as e:
                print(f"❌ Error cargando {key}: {e}")
        else:
            print(f"⚠️ Aviso: No se encontró {filename}, ese modo no funcionará.")

def process_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    img_array = np.array(img).astype(np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_batch)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Form("standard") # Recibimos la elección del usuario
):
    # 1. Selección del Modelo
    if model_type not in loaded_models:
        raise HTTPException(status_code=503, detail=f"El modelo '{model_type}' no está cargado o no existe.")
    
    selected_model = loaded_models[model_type]
    
    # 2. Configuración de Umbral Dinámico
    # Standard: Es muy seguro, usamos 0.50
    # HD: Es más sensible a contextos, usamos 0.30 para recuperar casos difíciles (Recall)
    current_threshold = 0.30 if model_type == "hd" else 0.50

    try:
        contents = await file.read()
        processed = process_image(contents)
        
        start = time.time()
        
        # 3. Inferencia
        preds = selected_model.predict(processed, verbose=0)[0]
        inference_time = (time.time() - start) * 1000
        
        results = {name: float(prob) for name, prob in zip(CLASS_NAMES, preds)}
        
        # 4. Filtrado Inteligente (Backend)
        detected_objects = []
        for name, prob in results.items():
            if prob >= current_threshold:
                detected_objects.append(name)
        
        # Fallback para no devolver vacío si hay algo "casi" seguro
        if not detected_objects:
            best_idx = np.argmax(preds)
            # Solo si supera un mínimo de seguridad (20%) para evitar ruido total
            if preds[best_idx] > 0.20:
                detected_objects = [CLASS_NAMES[best_idx]]
            else:
                detected_objects = ["Unknown"]

        return {
            "prediction": detected_objects,
            "is_multilabel": len(detected_objects) > 1,
            "time_ms": round(inference_time, 2),
            "probabilities": results,
            "model_used": model_type,
            "threshold_applied": current_threshold
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Error interno")

@app.get("/")
async def read_index():
    return FileResponse('index.html')
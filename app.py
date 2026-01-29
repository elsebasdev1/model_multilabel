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
from tensorflow.keras.applications.convnext import preprocess_input

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
MODEL_FOLDER = "/app/models"
MODEL_FILENAME = "model_production.keras"
MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_FILENAME)

IMG_SIZE = 224
CLASS_NAMES = ['Dog', 'Automobile', 'Bird']
THRESHOLD = 0.50 

app = FastAPI(title="SOTA Tiling API", version="4.0.0")
model = None

@app.on_event("startup")
async def startup_event():
    global model
    print(f"🚀 INICIANDO API CON INFERENCE TILING")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: No encuentro {MODEL_FILENAME}")
        return
    try:
        print("⏳ Cargando modelo...")
        model = keras.models.load_model(MODEL_PATH)
        # Warmup con batch de 6 (Simulando los tiles)
        dummy = np.zeros((6, IMG_SIZE, IMG_SIZE, 3))
        model.predict(dummy, verbose=0)
        print("✅ SISTEMA LISTO (Tiling Active)")
    except Exception as e:
        print(f"❌ ERROR CRÍTICO: {e}")

# ============================================================================
# LÓGICA DE TROCEADO (TILING) 🍰
# ============================================================================
def create_tiles(image):
    """Genera 6 vistas de la imagen: Original, Centro y 4 Esquinas"""
    tiles = []
    w, h = image.size
    
    # 1. Imagen Completa (Resized)
    tiles.append(image.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC))
    
    # Definir coordenadas para recortes (Superposición del 20%)
    # Usamos mitades con un poco de holgura
    half_w, half_h = int(w * 0.6), int(h * 0.6)
    
    # 2. Top-Left
    tiles.append(image.crop((0, 0, half_w, half_h)))
    # 3. Top-Right
    tiles.append(image.crop((w - half_w, 0, w, half_h)))
    # 4. Bottom-Left
    tiles.append(image.crop((0, h - half_h, half_w, h)))
    # 5. Bottom-Right
    tiles.append(image.crop((w - half_w, h - half_h, w, h)))
    
    # 6. Center Crop (Enfocado al medio)
    left = (w - half_w) // 2
    top = (h - half_h) // 2
    tiles.append(image.crop((left, top, left + half_w, top + half_h)))
    
    return tiles

def process_batch(image_bytes):
    # Abrir imagen original
    img_original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Generar los 6 tiles
    tiles = create_tiles(img_original)
    
    # Preprocesar cada tile
    batch = []
    for tile in tiles:
        # Resize final a 224x224
        tile_resized = tile.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        tile_arr = np.array(tile_resized).astype(np.float32)
        batch.append(tile_arr)
        
    # Convertir a tensor batch (6, 224, 224, 3)
    batch_np = np.array(batch)
    return preprocess_input(batch_np)

# ============================================================================
# ENDPOINT INTELIGENTE
# ============================================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        contents = await file.read()
        
        # 1. Procesar Batch de Tiles
        processed_batch = process_batch(contents)
        
        start = time.time()
        
        # 2. Inferencia en Paralelo (El modelo predice las 6 vistas a la vez)
        # shape output: (6, 3) -> 6 imágenes, 3 clases
        preds_batch = model.predict(processed_batch, verbose=0)
        
        # 3. AGREGACIÓN INTELIGENTE (MAX POOLING)
        # Tomamos la confianza máxima de cada clase a través de todas las vistas.
        # Si el perro sale en la esquina (tile 2), cogemos ese 99%.
        # axis=0 colapsa las 6 filas en 1 sola tomando el valor máximo.
        final_probs = np.max(preds_batch, axis=0)
        
        inference_time = (time.time() - start) * 1000
        
        # 4. Resultados
        results = {name: float(prob) for name, prob in zip(CLASS_NAMES, final_probs)}
        detected_objects = [name for name, prob in results.items() if prob >= THRESHOLD]
        
        # Fallback si nada supera el umbral
        if not detected_objects:
            best_idx = np.argmax(final_probs)
            detected_objects = [CLASS_NAMES[best_idx]]
        
        return {
            "prediction": detected_objects,
            "is_multilabel": len(detected_objects) > 1,
            "time_ms": round(inference_time, 2),
            "probabilities": results,
            "debug_mode": "Tiling Activated (6 views)"
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Error interno")

@app.get("/")
async def read_index():
    return FileResponse('index.html')
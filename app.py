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
MODEL_FILES = {
    "standard": "model_production.keras",      # Modelo Notebook 03
    "hd": "model_production_hd.keras"          # Modelo Notebook 04
}
IMG_SIZE = 224
CLASS_NAMES = ['Dog', 'Automobile', 'Bird']

app = FastAPI(title="SOTA Tiling Dual API", version="6.0.0")

# Cache de modelos en RAM
loaded_models = {}

@app.on_event("startup")
async def startup_event():
    print(f"🚀 INICIANDO API CON TILING STRATEGY (6-VIEWS)")
    
    # Cargar ambos modelos al inicio
    for key, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_FOLDER, filename)
        if os.path.exists(path):
            try:
                print(f"⏳ Cargando {key.upper()} desde {filename}...")
                loaded_models[key] = keras.models.load_model(path)
                
                # Warmup con un batch de 6 (Simulando los tiles)
                dummy_batch = np.zeros((6, IMG_SIZE, IMG_SIZE, 3))
                loaded_models[key].predict(dummy_batch, verbose=0)
                print(f"✅ Modelo {key.upper()} LISTO.")
            except Exception as e:
                print(f"❌ Error cargando {key}: {e}")
        else:
            print(f"⚠️ FALTANTE: {filename}")

# ============================================================================
# LÓGICA DE VISIÓN ARTIFICIAL (TILING / RECORTES)
# ============================================================================
def create_tiles(image):
    """
    Genera 6 vistas estratégicas de la imagen para detectar objetos pequeños.
    1. Original (Redimensionada)
    2. Centro
    3. Esquina Superior Izquierda
    4. Esquina Superior Derecha
    5. Esquina Inferior Izquierda
    6. Esquina Inferior Derecha
    """
    tiles = []
    w, h = image.size
    
    # 1. Vista Global
    tiles.append(image.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC))
    
    # Coordenadas para recortes (60% del tamaño para solapamiento)
    half_w, half_h = int(w * 0.60), int(h * 0.60)
    
    # 2. Esquinas
    tiles.append(image.crop((0, 0, half_w, half_h)))             # Top-Left
    tiles.append(image.crop((w - half_w, 0, w, half_h)))         # Top-Right
    tiles.append(image.crop((0, h - half_h, half_w, h)))         # Bottom-Left
    tiles.append(image.crop((w - half_w, h - half_h, w, h)))     # Bottom-Right
    
    # 3. Centro (Focus)
    left = (w - half_w) // 2
    top = (h - half_h) // 2
    tiles.append(image.crop((left, top, left + half_w, top + half_h)))
    
    return tiles

def process_batch(image_bytes):
    # Abrir imagen
    img_original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Generar los 6 tiles
    tiles = create_tiles(img_original)
    
    # Preprocesar cada tile para el modelo
    batch = []
    for tile in tiles:
        tile_resized = tile.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        tile_arr = np.array(tile_resized).astype(np.float32)
        batch.append(tile_arr)
        
    # Crear tensor batch (6, 224, 224, 3)
    batch_np = np.array(batch)
    
    # Normalización oficial de ConvNeXt
    return preprocess_input(batch_np)

# ============================================================================
# ENDPOINT DE PREDICCIÓN
# ============================================================================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Form("standard")
):
    # 1. Validación de Modelo
    if model_type not in loaded_models:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")
    
    selected_model = loaded_models[model_type]
    
    # 2. Ajuste de Umbral (Sensitivity Tuning)
    # HD necesita un umbral más bajo (30%) para detectar contextos difíciles
    current_threshold = 0.30 if model_type == "hd" else 0.50

    try:
        contents = await file.read()
        
        # 3. Procesamiento Tiling (Genera 6 imágenes)
        batch_processed = process_batch(contents)
        
        start = time.time()
        
        # 4. Inferencia en Batch
        # El modelo predice las 6 vistas simultáneamente
        # preds_batch shape: (6, 3)
        preds_batch = selected_model.predict(batch_processed, verbose=0)
        
        # 5. MAX POOLING (La clave del éxito)
        # Tomamos la confianza máxima encontrada en CUALQUIER vista.
        # Si el loro sale en la esquina (tile 2), esa confianza gana.
        final_probs = np.max(preds_batch, axis=0)
        
        inference_time = (time.time() - start) * 1000
        
        results = {name: float(prob) for name, prob in zip(CLASS_NAMES, final_probs)}
        
        # 6. Filtrado de Clases (Solo lo que supera el umbral)
        detected_objects = []
        for name, prob in results.items():
            if prob >= current_threshold:
                detected_objects.append(name)
        
        # Fallback: Si nada supera el umbral, devolvemos la mejor opción si es > 20%
        if not detected_objects:
            best_idx = np.argmax(final_probs)
            if final_probs[best_idx] > 0.20:
                detected_objects = [CLASS_NAMES[best_idx]]
            else:
                detected_objects = ["Unknown"]

        return {
            "prediction": detected_objects,
            "is_multilabel": len(detected_objects) > 1,
            "time_ms": round(inference_time, 2),
            "probabilities": results,
            "model_used": model_type,
            "method": "Tiling (6-Views)"
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Error interno")

@app.get("/manifest.json")
async def get_manifest():
    # El navegador necesita saber que es un JSON para activarlo
    return FileResponse("manifest.json", media_type="application/json")

@app.get("/icon.png")
async def get_icon():
    # Sirve el icono para que se vea en el celular
    return FileResponse("icon.png", media_type="image/png")

@app.get("/")
async def read_index():
    return FileResponse('index.html')
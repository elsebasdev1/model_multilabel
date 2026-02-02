# 👁️ SOTA Multi-Label Visual Analysis System

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange?style=for-the-badge&logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)
![Status](https://img.shields.io/badge/Status-Production_Ready-success?style=for-the-badge)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue?style=for-the-badge&logo=mlflow)


> **Un sistema de visión artificial de alto rendimiento capaz de detectar múltiples objetos simultáneamente en entornos no controlados, utilizando arquitecturas ConvNeXt Base y estrategias de Adaptación de Dominio.**

---

## 📸 Demo & Interfaz

El sistema cuenta con una interfaz minimalista y profesional desarrollada con **TailwindCSS**, diseñada para la inferencia en tiempo real.

<img width="2511" height="1343" alt="Screenshot 2026-02-01 122139" src="https://github.com/user-attachments/assets/6f982791-6b76-439f-9851-f5aa0ee95448" />

### 🔍 Detección Inteligente (Tiling Strategy)
El sistema no solo mira la imagen completa. Aplica una estrategia de **"Smart Tiling"** (6 vistas simultáneas) para detectar objetos pequeños u ocultos.

<img width="1815" height="1167" alt="Screenshot 2026-02-01 122407" src="https://github.com/user-attachments/assets/890b12f8-31c3-4074-a66d-98769df496d6" />

---

## 🚀 Características Clave

* **🧠 Arquitectura SOTA:** Basado en **ConvNeXt Base** (88M parámetros), pre-entrenado en ImageNet y ajustado específicamente para nuestro dominio.
* **🔄 Motor Dual (Dual-Engine):**
    * **Modo Standard:** Entrenado en CIFAR-10 (Accuracy 99.8%) para benchmarks académicos.
    * **Modo HD (Real World):** Ajustado mediante *Fine-Tuning* para fotografías de alta resolución, superando el problema del "Domain Gap".
* **🍰 Inference Tiling:** Procesa 6 recortes estratégicos (Centro + 4 Esquinas + Original) en paralelo para maximizar el Recall.
* **🎚️ Umbral Dinámico:** Ajuste automático de sensibilidad (30% vs 50%) dependiendo del modelo seleccionado para reducir Falsos Negativos en contextos complejos.
* **🐳 Dockerized:** Despliegue inmediato con un solo comando.

---

## MLOps & Experiment Tracking (MLflow)

Para garantizar la reproducibilidad científica y el monitoreo en tiempo real, se integró el ciclo de entrenamiento con MLflow. Esto permitió auditar la evolución de los gradientes y detectar convergencia temprana.

### Tablero de métricas en tiempo real
<img width="1866" height="696" alt="Screenshot_20260202_141826" src="https://github.com/user-attachments/assets/764fb2f4-c5c1-4570-9431-1bd69ccdcb02" />

## Análisis de las Métricas

### Convergencia Robusta
El val_loss cae rápidamente y se estabiliza cerca de 0.01, confirmando que no hay overfitting degradante.

### Efecto MixUp
Se observa que el train_accuracy es inferior al val_accuracy. Esto es un comportamiento esperado y deseable cuando se utiliza MixUp Augmentation: el modelo entrena con imágenes mezcladas para forzar una generalización perfecta en los datos de validación.

### AUC SOTA
La métrica val_auc se mantiene constante cerca de 1.0, lo que valida la capacidad del modelo para separar las clases con un umbral de decisión limpio.

---

## 🛠️ Arquitectura del Proyecto

El proyecto sigue una metodología rigurosa de Data Science dividida en 4 fases (Cuadernos):

### 1. Análisis & Estrategia
Definición del problema Multi-Label. Selección de **CIFAR-10** como dataset base y **Sigmoid** como función de activación para permitir probabilidades independientes (e.g., 99% Perro, 99% Auto).

### 2. Ingeniería de Datos (ETL)
Pipeline de extracción y transformación.
* Filtrado de clases irrelevantes.
* Upscaling bicúbico a **224x224**.
* Persistencia en formato binario `.npy` para optimizar I/O.

### 3. Entrenamiento (Transfer Learning)
Entrenamiento del modelo base utilizando técnicas de regularización avanzadas:
* **MixUp Augmentation:** Para suavizar la frontera de decisión.
* **Mixed Precision (FP16):** Para optimizar el uso de VRAM.
* **Resultado:** 99.87% Accuracy en Test Set.

### 4. Adaptación de Dominio (The "Real World" Fix)
Resolución del problema de **"Catastrophic Forgetting"** en imágenes HD.
* Ingesta de dataset curado HD.
* Corrección automática de alineación de etiquetas (Label Re-ordering).
* Fine-Tuning con Learning Rate reducido (`1e-5`).
* **Mejora:** Del 83% al **94.4%** en imágenes reales.
<img width="1438" height="553" alt="Screenshot_20260202_141033" src="https://github.com/user-attachments/assets/c1d9c860-1873-4a8a-b6f7-d88b09a84eab" />

## Resultados Comparativos (Dataset Real-World)

| Métrica           | Standard (CIFAR-10) | HD (Fine-Tuned) | Diferencia |
|-------------------|--------------------|----------------|------------|
| Accuracy Global   | 33.33%             | 100.00%        | +66.67%    |
| F1 dog            | 50.00%             | 100.00%        | +50.00%    |
| F1 automobile     | 0.00%              | 0.00%          | +0.00%     |
| F1 bird           | 0.00%              | 0.00%          | +0.00%     |

---

## 💻 Instalación y Uso

### Prerrequisitos
* Docker & Docker Compose
* NVIDIA GPU (Opcional, el sistema tiene modo CPU-Safe)

### Despliegue Rápido
Clona el repositorio y levanta el contenedor:

```bash
git clone [https://github.com/elsebasdev1/model_multilabel.git](https://github.com/elsebasdev1/model_multilabel.git)
cd model_multilabel

# Construir y levantar
docker-compose up --build

Accede a la interfaz web en: http://localhost:8000
```
## 📂 Estructura del Repositorio
```
├── app.py                 # Backend FastAPI (Lógica Dual + Tiling)
├── Dockerfile             # Configuración de entorno Python 3.11 Slim
├── requirements.txt       # Dependencias (TensorFlow, Pillow, FastAPI)
├── index.html             # Frontend (HTML5 + TailwindCSS)
├── notebooks/             # Jupyter Notebooks (El cerebro del proyecto)
│   ├── 01_Analysis.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Training_SOTA.ipynb
│   └── 04_Domain_Adaptation.ipynb
└── models/                # Pesos de los modelos (.keras)
```
## 📊 Métricas de Rendimiento
| Modelo         | Dataset         | Accuracy | Inferencia (Avg)      |
|:--------------|:----------------|:--------:|----------------------:|
| Standard      | CIFAR-10 (Test) | 99.87%   | ~5000 ms               |
| HD Fine-Tuned | Real World HD   | 94.44%   | ~5500 ms              |


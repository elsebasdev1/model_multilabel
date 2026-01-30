# 👁️ SOTA Multi-Label Visual Analysis System

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange?style=for-the-badge&logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)
![Status](https://img.shields.io/badge/Status-Production_Ready-success?style=for-the-badge)

> **Un sistema de visión artificial de alto rendimiento capaz de detectar múltiples objetos simultáneamente en entornos no controlados, utilizando arquitecturas ConvNeXt Base y estrategias de Adaptación de Dominio.**

---

## 📸 Demo & Interfaz

El sistema cuenta con una interfaz minimalista y profesional desarrollada con **TailwindCSS**, diseñada para la inferencia en tiempo real.

![Dashboard Principal](assets/demo_dashboard.png)
*(Aquí va captura del Front completo mostrando el input y los resultados vacíos)*

### 🔍 Detección Inteligente (Tiling Strategy)
El sistema no solo mira la imagen completa. Aplica una estrategia de **"Smart Tiling"** (6 vistas simultáneas) para detectar objetos pequeños u ocultos, como un loro en la ventana de un auto.

![Resultado Multi-Label](assets/result_multilabel.png)
*(Aquí va captura del resultado "Automobile + Bird" mostrando las barras de progreso)*

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

![Curvas de Entrenamiento](assets/training_curves.png)
*(Aquí va captura de las gráficas de Loss/Accuracy del cuaderno 03)*

### 4. Adaptación de Dominio (The "Real World" Fix)
Resolución del problema de **"Catastrophic Forgetting"** en imágenes HD.
* Ingesta de dataset curado HD.
* Corrección automática de alineación de etiquetas (Label Re-ordering).
* Fine-Tuning con Learning Rate reducido (`1e-5`).
* **Mejora:** Del 83% al **94.4%** en imágenes reales.

---

## 💻 Instalación y Uso

### Prerrequisitos
* Docker & Docker Compose
* NVIDIA GPU (Opcional, el sistema tiene modo CPU-Safe)

### Despliegue Rápido
Clona el repositorio y levanta el contenedor:

```bash
git clone [https://github.com/tu-usuario/multilabel-vision-system.git](https://github.com/tu-usuario/multilabel-vision-system.git)
cd multilabel-vision-system

# Construir y levantar
docker-compose up --build

Accede a la interfaz web en: http://localhost:8000
```
## 📂 Estructura del Repositorio

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

## 📊 Métricas de Rendimiento
Modelo,Dataset,Accuracy,Inferencia (Avg)
Standard,CIFAR-10 (Test),99.87%,~150ms
HD Fine-Tuned,Real World HD,94.44%,~3000ms (con Tiling)

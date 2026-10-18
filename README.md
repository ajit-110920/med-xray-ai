# MedXRay-AI: Chest X-ray Multi-Label Disease Classification System

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg?style=flat&logo=streamlit)](https://streamlit.io)
[![Keras 3](https://img.shields.io/badge/Deep%20Learning-Keras%203%20%2F%20TensorFlow-D00000.svg?style=flat&logo=keras)](https://keras.io)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python)](https://python.org)
[![Tests Passing](https://img.shields.io/badge/Tests-16%20Passed%20(100%25)-brightgreen.svg?style=flat&logo=pytest)](tests/)
[![Repository](https://img.shields.io/badge/GitHub-ajit--110920%2Fmed--xray--ai-blue.svg?style=flat&logo=github)](https://github.com/ajit-110920/med-xray-ai.git)

An end-to-end, clinical-research deep learning application built around pre-trained **DenseNet121** convolutional neural networks fine-tuned on the **NIH ChestX-ray14** dataset to detect and score 14 distinct thoracic pathologies from frontal chest radiographs.

The solution features a high-throughput **FastAPI** asynchronous REST backend, an interactive medical-research workspace built with **Streamlit** (featuring a refined soft-dark radiology aesthetic), and an automated **pytest** suite validating unit, integration, and on-disk model artifacts with 100% test coverage.

> **⚠️ RESEARCH & EDUCATIONAL PROTOTYPE ONLY**  
> This software is strictly an **academic and research demonstration**. It has **not** been cleared, certified, or approved by the US FDA, EMA, or any healthcare regulatory body. Model scores are raw, uncalibrated activation values and **must never be used** for clinical diagnosis, patient triage, prognosis, or medical treatment decisions.

---

## Table of Contents

1. [Executive Summary & Clinical Context](#1-executive-summary--clinical-context)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Deep Learning Model & Preprocessing Pipeline](#3-deep-learning-model--preprocessing-pipeline)
4. [Dataset & Target Pathologies](#4-dataset--target-pathologies)
5. [Model Checkpoints & Loss Function Analysis](#5-model-checkpoints--loss-function-analysis)
6. [Repository Structure](#6-repository-structure)
7. [REST API Documentation](#7-rest-api-documentation)
8. [Frontend Radiology Research Workspace](#8-frontend-radiology-research-workspace)
9. [Installation & Setup Guide](#9-installation--setup-guide)
10. [Running the Application](#10-running-the-application)
11. [Model Selection & Configuration](#11-model-selection--configuration)
12. [Automated Test Suite](#12-automated-test-suite)
13. [Limitations & Responsible AI Disclaimer](#13-limitations--responsible-ai-disclaimer)

---

## 1. Executive Summary & Clinical Context

Chest radiography (CXR) is the most frequently performed diagnostic imaging examination worldwide. Interpreting chest radiographs requires distinguishing subtle visual cues across overlapping anatomical structures (lungs, cardiac silhouette, ribs, spine, and mediastinum).

This project demonstrates how deep transfer learning can be operationalized into a maintainable, full-stack software application that:
* Ingests frontal chest radiographs (PA/AP views) in PNG or JPEG format.
* Preprocesses and normalizes image matrices to match DenseNet121 ImageNet statistics.
* Evaluates 14 thoracic disease labels simultaneously using independent sigmoid activation units.
* Delivers real-time predictions via an asynchronous REST API.
* Presents interactive visual rankings and complete tabular breakdowns through a soft-dark radiology workstation dashboard.

---

## 2. End-to-End System Architecture

The application is structured into decoupled, loosely-coupled layers adhering to clean software engineering practices:

```
                            +-------------------------------------------+
                            |       User / Clinical Researcher          |
                            +---------------------+---------------------+
                                                  |
                                   Multipart Image Upload (HTTP)
                                                  |
                                                  v
+-------------------------------+      +-----------------------------------------+
|     Streamlit Frontend        | ---> |            FastAPI Backend              |
|      (frontend/app.py)        | HTTP |           (backend/main.py)             |
|                               | <--- |                                         |
| - Header & System Badge       |      | - GET  /         (Service status)       |
| - Drag-and-drop X-ray Upload  |      | - GET  /health   (Model readiness)      |
| - Radiograph Lightbox Canvas  |      | - GET  /labels   (14 disease labels)    |
| - Top 3 Activation Cards      |      | - POST /predict  (Preprocessing & Run)  |
| - Ranked Altair Bar Chart     |      +--------------------+--------------------+
| - Complete Dataframe Table    |                           |
+-------------------------------+             Image Bytes Preprocessing Pipeline
                                                            |
                                                            v
                                       +-----------------------------------------+
                                       |          Keras Model Manager            |
                                       |           (backend/model.py)            |
                                       |                                         |
                                       | Architecture: DenseNet121               |
                                       | Loaded via: keras.models.load_model     |
                                       |             (compile=False)             |
                                       | Checkpoints:                            |
                                       | - weighted_best.keras (Default)         |
                                       | - finetuned_best.keras                  |
                                       +-----------------------------------------+
```

### Key Architectural Strengths
* **Zero Model Duplication:** Streamlit functions strictly as a client. Heavy TensorFlow/Keras weights (30–36 MB) are loaded once in memory by the FastAPI backend during the `lifespan` startup phase.
* **Separation of Concerns:** Business logic, input validation (Pydantic schemas), preprocessing, and inference are isolated into dedicated modules (`backend/config.py`, `backend/schemas.py`, `backend/model.py`, and `backend/main.py`).
* **Error Resilience:** Corrupted image streams, unsupported file types, and uninitialized models are handled with structured HTTP error codes (`400 Bad Request`, `422 Unprocessable Entity`, `503 Service Unavailable`).

---

## 3. Deep Learning Model & Preprocessing Pipeline

### Model Architecture
* **Backbone:** `DenseNet121` (Densely Connected Convolutional Networks). Dense connectivity facilitates feature reuse across layers, alleviates vanishing gradients, and substantially reduces parameters compared to classic ResNet backbones.
* **Feature Extractor:** Convolutional dense blocks pre-trained on ImageNet.
* **Classification Head:** Global Average Pooling (GAP) $\rightarrow$ Dropout layer $\rightarrow$ Fully-connected Dense layer (`disease_scores`) with 14 output units and **sigmoid** activation.

### Preprocessing Protocol
Preprocessing strictly reproduces the model training conditions:
1. **Stream Validation & Decoding:** Image bytes are validated using Pillow (`PIL.Image.open`) to confirm file integrity.
2. **Color Space Normalization:** Grayscale radiographs (mode `L`) or 4-channel scans (`RGBA`) are converted to standard 3-channel `RGB`.
3. **Spatial Resizing:** Rescaled to $224 \times 224$ pixels using bilinear interpolation.
4. **Data Type Conversion:** Converted to 32-bit floating-point NumPy tensor (`float32`).
5. **DenseNet Normalization:** Standard ImageNet channel-wise normalization is applied:
   $$\text{Input} = \frac{\frac{\text{Pixel}}{255.0} - \mu}{\sigma}$$
   Where $\mu = [0.485, 0.456, 0.406]$ and $\sigma = [0.229, 0.224, 0.225]$.
6. **Batch Expansion:** Expanded to tensor shape `(1, 224, 224, 3)`.
7. **Direct Forward Pass:** Computed using `model(img_batch, training=False)` to minimize inference latency.

---

## 4. Dataset & Target Pathologies

The model targets 14 binary thoracic pathology classifications from the **NIH ChestX-ray14** dataset:

| # | Pathology Label | Clinical Overview |
|:---:|---|---|
| **1** | `Atelectasis` | Partial or complete collapse of the lung or lobe |
| **2** | `Cardiomegaly` | Enlarged heart silhouette (cardiothoracic ratio > 50%) |
| **3** | `Consolidation` | Alveolar air spaces filled with fluid/pus/exudate |
| **4** | `Edema` | Accumulation of fluid in extravascular spaces of the lung |
| **5** | `Effusion` | Abnormal collection of fluid in the pleural space |
| **6** | `Emphysema` | Permanent enlargement of airspace distal to terminal bronchioles |
| **7** | `Fibrosis` | Scarring and thickening of lung interstitium |
| **8** | `Hernia` | Diaphragmatic herniation into thoracic cavity |
| **9** | `Infiltration` | Radiodense substance in the pulmonary parenchyma |
| **10** | `Mass` | Pulmonary lesion greater than 3 cm in diameter |
| **11** | `Nodule` | Small, well-defined pulmonary lesion $\le 3\text{ cm}$ |
| **12** | `Pleural_Thickening`| Fibrous thickening or calcification of the pleura |
| **13** | `Pneumonia` | Inflammatory condition affecting alveoli |
| **14** | `Pneumothorax` | Presence of air in the pleural cavity causing lung collapse |

---

## 5. Model Checkpoints & Loss Function Analysis

The repository includes two trained checkpoints saved in Keras 3 (`v3.13+`) format:

| Checkpoint File | Size | Training Strategy | Characteristics |
|---|:---:|---|---|
| **`weighted_best.keras`** *(Default)* | 29.8 MB | Class-Weighted Binary Cross-Entropy | Addresses severe label imbalance; achieves best overall validation ROC-AUC ($\sim 0.7047$) |
| **`finetuned_best.keras`** | 36.4 MB | Standard Binary Cross-Entropy | Unweighted fine-tuning checkpoint |

### The Class Imbalance Problem
In medical imaging datasets such as NIH ChestX-ray14, positive disease cases are rare compared to negative controls (often $<5\%$ prevalence for conditions like `Hernia` or `Fibrosis`). 

Under standard Binary Cross-Entropy (BCE), a model optimizing for accuracy easily gets biased toward negative predictions. To counteract this:
$$L_{\text{weighted}}(y, \hat{y}) = - \left( w_p \cdot y \log(\hat{y}) + w_n \cdot (1 - y) \log(1 - \hat{y}) \right)$$
where $w_p = \frac{N_{\text{neg}}}{N_{\text{total}}}$ and $w_n = \frac{N_{\text{pos}}}{N_{\text{total}}}$.

> **Important Note on Inference Loading:** `weighted_best.keras` was compiled with a custom loss function (`weighted_binary_crossentropy`). When initializing the model for production inference, our model manager uses `keras.models.load_model(..., compile=False)`. This ensures seamless deserialization without requiring loss function recompilation.

---

## 6. Repository Structure

```
chest_xray_model_package/
├── backend/
│   ├── __init__.py
│   ├── config.py              # Environment variables, file paths, preprocessing settings
│   ├── model.py               # ModelManager, Pillow image decoder, DenseNet normalization, inference
│   ├── schemas.py             # Pydantic v2 validation and response models
│   └── main.py                # FastAPI app with lifespan management, CORS, and REST endpoints
├── frontend/
│   ├── __init__.py
│   └── app.py                 # Streamlit medical-research dashboard (soft-dark theme, tabs, Altair chart)
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # Fixtures for synthetic radiograph images and mock inference
│   ├── test_api.py            # Unit tests for /, /health, /labels, /predict, and error handling
│   ├── test_smoke_model.py    # Smoke tests against real on-disk .keras model files
│   └── test_integration.py    # End-to-end integration and dynamic checkpoint switching tests
├── .streamlit/
│   └── config.toml            # Enforces soft-dark theme (#141B26 background, #38BDF8 primary)
├── .env.example               # Template for environment configuration
├── .gitignore                 # Excludes .venv/, caches, temporary uploads, and logs
├── pytest.ini                 # Pytest configuration and marker registration
├── requirements.txt           # Unified dependency specifications
├── requirements-backend.txt   # Backend-specific dependencies
├── requirements-frontend.txt  # Frontend-specific dependencies
├── labels.json                # 14 disease labels in model output order
├── training_config.json       # Training metadata (DenseNet121, 224x224, 8k train / 2k val)
├── weighted_best.keras        # Class-weighted loss model checkpoint (29.8 MB)
├── finetuned_best.keras       # Standard fine-tuned model checkpoint (36.4 MB)
└── README.md                  # Comprehensive project documentation
```

---

## 7. REST API Documentation

### Endpoints Overview

| Method | Endpoint | Description | Status Codes |
|:---:|---|---|:---:|
| `GET` | `/` | Returns service status, version, and model name | `200` |
| `GET` | `/health` | Verifies service readiness and model loading status | `200`, `503` |
| `GET` | `/labels` | Returns all 14 disease labels in model output order | `200`, `500` |
| `POST` | `/predict` | Ingests chest X-ray image; returns 14 activation scores | `200`, `400`, `500`, `503` |

### Prediction Request Example (`cURL`)
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_scan.png"
```

### Prediction Response Schema (`JSON`)
```json
{
  "labels": [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
  ],
  "scores": [
    0.3028, 0.2552, 0.6318, 0.3704, 0.3603,
    0.0227, 0.2460, 0.0503, 0.3385, 0.1570,
    0.3179, 0.0931, 0.1802, 0.2928
  ],
  "predictions": [
    {"label": "Atelectasis", "score": 0.3028},
    {"label": "Cardiomegaly", "score": 0.2552},
    {"label": "Consolidation", "score": 0.6318}
  ],
  "ranked_predictions": [
    {"label": "Consolidation", "score": 0.6318},
    {"label": "Edema", "score": 0.3704},
    {"label": "Effusion", "score": 0.3603}
  ],
  "model_identifier": "weighted_best.keras",
  "disclaimer": "RESEARCH AND EDUCATIONAL PROTOTYPE ONLY: The model outputs provided by this system are raw, uncalibrated activation scores..."
}
```

Interactive OpenAPI documentation is accessible at `http://127.0.0.1:8000/docs`.

---

## 8. Frontend Radiology Research Workspace

The Streamlit interface (`frontend/app.py`) is styled as a professional, soft-dark clinical AI workspace:

* **Color Palette:**
  * Background Canvas: `#141B26` (deep slate navy)
  * Sidebar & Panels: `#1A2332`
  * Elevated Cards: `#222D3E` with `#38455A` borders
  * Primary Accent: `#38BDF8` (radiant medical blue)
  * Secondary Accent: `#2DD4BF` (mint teal)
  * Headings: `#FFFFFF` (brilliant white for high contrast)
  * Body Text: `#CBD5E1`
* **Zero Horizontal Scrollbar:** Strict responsive box-sizing prevents any unwanted horizontal overflow.
* **Two-Column Workspace:**
  * **Left:** Drag-and-drop uploader with file metadata summary, primary *"Analyze image"* button, and *"Clear image"* reset.
  * **Right:** High-contrast neutral radiograph lightbox preserving scan aspect ratio.
* **Results Display:**
  * **Top 3 Highlights:** White/slate metric cards highlighting highest relative activations with 4-decimal precision.
  * **Interactive Tabs:**
    * *Ranked Activation Chart:* Full-width horizontal Altair bar chart with hover tooltips.
    * *Complete Pathology Table:* Full data table with rank, disease label, raw score, and relative indicator progress bar.

---

## 9. Installation & Setup Guide

### Prerequisites
* **Operating System:** Windows, macOS, or Linux.
* **Python Version:** Python 3.10, 3.11, or 3.12 (Python 3.11 recommended).
* **Package Manager:** `pip` or [Astral uv](https://github.com/astral-sh/uv).

### Installation (Windows PowerShell)
```powershell
# 1. Clone repository
git clone https://github.com/ajit-110920/med-xray-ai.git
cd med-xray-ai

# 2. Create and activate a virtual environment
uv venv --python 3.11 .venv
.venv\Scripts\activate

# 3. Install dependencies
uv pip install -r requirements.txt
```

### Installation (Linux / macOS)
```bash
# 1. Clone repository
git clone https://github.com/ajit-110920/med-xray-ai.git
cd med-xray-ai

# 2. Create and activate virtual environment
uv venv --python 3.11 .venv
source .venv/bin/activate

# 3. Install dependencies
uv pip install -r requirements.txt
```

---

## 10. Running the Application

### Step 1: Start the FastAPI Backend
Launch the inference service on port 8000:
```powershell
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Verify the backend is live at [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health).

### Step 2: Start the Streamlit Frontend
In a second terminal window (with `.venv` activated):
```powershell
streamlit run frontend/app.py --server.port 8501
```
Open your browser to [http://localhost:8501](http://localhost:8501).

---

## 11. Model Selection & Configuration

You can seamlessly switch model checkpoints using environment variables:

### Option A: Using `.env` file
Copy `.env.example` to `.env`:
```powershell
cp .env.example .env
```
Set `MODEL_PATH`:
```ini
MODEL_PATH=finetuned_best.keras
```

### Option B: Terminal Environment Variable
**Windows (PowerShell):**
```powershell
$env:MODEL_PATH = "finetuned_best.keras"
uvicorn backend.main:app --port 8000
```

**Linux / macOS:**
```bash
export MODEL_PATH="finetuned_best.keras"
uvicorn backend.main:app --port 8000
```

---

## 12. Automated Test Suite

The test suite validates API routes, input sanitization, error responses, end-to-end integration, and actual `.keras` model checkpoints:

```powershell
# Run the complete test suite
pytest -v

# Run fast API unit tests (mocked model)
pytest tests/test_api.py -v

# Run smoke tests on actual saved .keras models
pytest tests/test_smoke_model.py -v

# Run end-to-end integration and checkpoint switching tests
pytest tests/test_integration.py -v
```

### Test Coverage Summary (16 / 16 Passed)
* ✅ Root status and version verification (`GET /`)
* ✅ Model health reporting when loaded vs uninitialized (`GET /health`)
* ✅ Exact 14 disease label verification (`GET /labels`)
* ✅ PNG, JPEG, and Grayscale image inference (`POST /predict`)
* ✅ Rejection of unsupported extensions, empty files, and corrupt images (HTTP 400)
* ✅ NaN and Inf output safety guards
* ✅ End-to-end integration with real checkpoint inference
* ✅ Dynamic checkpoint switching (`weighted_best.keras` $\leftrightarrow$ `finetuned_best.keras`)
* ✅ Smoke tests verifying forward passes directly on real on-disk `.keras` files

---

## 13. Limitations & Responsible AI Disclaimer

1. **Uncalibrated Output Scores:** The values generated by the model are sigmoid activation scores in $[0.0000, 1.0000]$. They reflect the relative excitation of network nodes given input features, **not calibrated Bayesian probabilities** of disease presence.
2. **Decision Thresholds:** Due to class imbalance in training data, standard decision thresholds ($0.5$) yield sub-optimal sensitivity on rare conditions. Any deployment must calibrate class-specific operational thresholds against validated clinical validation splits.
3. **No Clinical Diagnosis:** This application does not constitute a medical device, diagnostic tool, or radiologist substitute.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

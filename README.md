# RoadGuardAI: Intelligent Road Safety and Pothole Detection Using Artificial Intelligence

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%2BCUDA124-orange.svg)](https://pytorch.org/)
[![Ultralytics YOLO11n](https://img.shields.io/badge/YOLO-v11n-green.svg)](https://docs.ultralytics.com/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 1. Introduction
**RoadGuardAI** is an advanced, production-ready computer vision platform designed to autonomously detect road hazards, structural asphalt degradation, and potholes in real time. Built specifically as a BTech Capstone Project, RoadGuardAI bridges state-of-the-art Deep Learning object detection (Ultralytics YOLO11n) with a high-throughput asynchronous FastAPI backend and a responsive glassmorphic web dashboard.

---

## 2. Problem Statement
Road potholes and surface fractures are primary contributors to vehicle wear, traffic congestion, and fatal vehicular collisions globally. Traditional road inspection relies on manual visual audits or expensive specialized sensor vehicles, leading to:
- Delayed reporting and hazard identification.
- Inconsistent and subjective hazard assessment.
- High operational inspection costs.
- Inability to perform continuous real-time roadway monitoring.

---

## 3. Project Objectives
1. **Automated Vision Pipeline**: Build a real-time deep learning pipeline capable of detecting and localizing potholes across still images, high-speed dashcam videos, and live webcam feeds.
2. **Leakage-Free Dataset Architecture**: Implement strict scene-level grouping to prevent Roboflow-augmented variations from cross-contaminating training and evaluation sets.
3. **Rigorous Academic Benchmarking**: Evaluate model accuracy using genuine object detection metrics ($mAP@50$, $mAP@50\text{-}95$, Precision, Recall, F1) on an independent test split.
4. **Interactive AI SaaS Web Interface**: Provide a modern glassmorphic dashboard with live video/webcam processing, HUD telemetry, and persistent SQLite history tracking.

---

## 4. Existing System vs. Proposed System

| Dimension | Existing System | Proposed RoadGuardAI System |
| :--- | :--- | :--- |
| **Inspection Method** | Manual survey / Accelerometers | Autonomous Computer Vision (YOLO) |
| **Processing Latency** | Days to weeks | Real-time (< 20 ms per frame on GPU) |
| **Localization** | Coarse GPS point | Exact pixel bounding box coordinates |
| **Video Support** | Difficult manual review | Automated frame-by-frame pipeline |
| **Data Leakage** | Often overlooked in academic projects | Scene-level hash grouping (0% leakage) |
| **Cost** | Extremely High | Standard dashboard camera & laptop GPU |

---

## 5. System Architecture

```
[ Camera / Dashcam Video / Images ]
                 │
                 ▼
[ FastAPI High-Performance Web Server (Port 8000) ]
                 │
  ┌──────────────┼───────────────────────────┐
  │              │                           │
  ▼              ▼                           ▼
[OpenCV Decoder] [Ultralytics YOLO11 Engine] [SQLite Audit Database]
(Frame Pipeline) (CUDA GPU Accelerated)      (Persistent History)
  │              │                           │
  └──────────────┴─────────────┬─────────────┘
                               ▼
               [Glassmorphic SaaS Frontend]
                 - Real-time HUD Counter
                 - Chart.js Performance Curves
                 - Interactive Video Player
```

---

## 6. Technologies & Libraries
- **Deep Learning Framework**: PyTorch 2.6.0 with CUDA 12.4 acceleration.
- **Object Detection Architecture**: Ultralytics YOLO11n (2.59M parameters, 6.5 GFLOPs).
- **Computer Vision**: OpenCV (cv2) 5.0, Pillow 13.0, NumPy.
- **Backend Server**: FastAPI, Uvicorn (ASGI), Python-Multipart, Aiofiles.
- **Database**: SQLite3 with indexing on timestamps and detection types.
- **Frontend Architecture**: Modern HTML5, Glassmorphism CSS, JavaScript (Vanilla ES6), Chart.js 4.4, FontAwesome 6.5.

---

## 7. Dataset Architecture & Verification

The verified dataset comprises **2,083 genuine road images** and **4,246 annotated bounding boxes**:

- **Annotation Format**: YOLO normalized format (`class_id x_center y_center width height`).
- **Classes**: 1 class (`0: pothole`).
- **Data Quality**: 0 corrupted images, 0 missing labels, 0 out-of-bound annotations.
- **Leakage Prevention**: All images were partitioned across 671 unique base scene identities to ensure augmented variations remained strictly inside their respective split.

### Split Distribution (70 / 20 / 10)
- **Train Set**: 1,445 images (69.4%) — 2,897 annotations
- **Validation Set**: 420 images (20.2%) — 890 annotations
- **Test Set**: 218 images (10.5%) — 459 annotations

---

## 8. Machine Learning Model & Training

### Model Selection: YOLO11n
YOLO11n is the newest lightweight generation of the You Only Look Once architecture featuring:
- C3k2 cross-stage partial blocks.
- Spatial Pyramid Pooling Fast (SPPF).
- Dynamic Anchor-free Head with decoupled classification and box regression branches.

### Training Configuration
- **Epochs**: 25
- **Batch Size**: 16
- **Image Size**: 640 × 640
- **Optimizer**: AdamW ($lr=0.002$, momentum $= 0.9$, weight decay $= 0.0005$)
- **Mixed Precision**: Automatic Mixed Precision (AMP) enabled on NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM).

---

## 9. Evaluation Metrics: Object Detection vs. Classification

In standard classification tasks, accuracy merely evaluates whether an image as a whole was categorized correctly. In **Object Detection**, accuracy requires dual verification:
1. **Localization**: Does the predicted bounding box intersect the ground truth above a defined Intersection-over-Union ($IoU$) threshold?
2. **Classification**: Is the identified category correct?

Therefore, RoadGuardAI benchmarks performance using:
- **Precision ($P$)**: $\frac{TP}{TP + FP}$
- **Recall ($R$)**: $\frac{TP}{TP + FN}$
- **F1-Score**: $2 \times \frac{P \times R}{P + R}$
- **mAP@50**: Mean Average Precision at $IoU = 0.50$.
- **mAP@50-95**: Mean Average Precision averaged across IoU thresholds from 0.50 to 0.95 in steps of 0.05.

---

## 10. Installation & Setup

### Prerequisites
- Windows 10/11 (or Linux/macOS)
- Python 3.10 to 3.12 installed
- NVIDIA GPU recommended for CUDA acceleration (CPU fallback fully supported)

### Step 1: Clone or Navigate to Directory
```powershell
cd c:\Users\vikram\OneDrive\Desktop\RoadGuardAI
```

### Step 2: Set up Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
# For NVIDIA GPU with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r backend/requirements.txt
```

---

## 11. How to Run the Application

### Option A: Using the 1-Click Launcher (Windows)
Double-click `start_app.bat` or run:
```powershell
.\run_app.ps1
```

### Option B: Manual Terminal Execution
```powershell
# 1. Activate environment
.\.venv\Scripts\Activate.ps1

# 2. Start FastAPI Server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your web browser at: **`http://127.0.0.1:8000`**

---

## 12. API Documentation

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Health check & model status |
| `GET` | `/api/dashboard` | Aggregated detection statistics |
| `POST` | `/api/detect/image` | Run YOLO inference on an uploaded image |
| `POST` | `/api/detect/video` | Process dashcam video frame by frame |
| `POST` | `/api/detect/frame` | Live webcam frame inference |
| `GET` | `/api/model/status` | Model loading & hardware metadata |
| `GET` | `/api/model/metrics` | Genuine test evaluation metrics & curves |
| `GET` | `/api/dataset/summary` | Dataset split counts & validation report |
| `GET` | `/api/history` | Paginated detection history |
| `DELETE`| `/api/history/{id}` | Delete detection record |
| `GET` | `/api/results/{file}` | Secure file serving for annotated results |

---

## 13. System Features & Views
1. **Welcome / Hero View**: Dynamic road safety overview with project motivation and call-to-actions.
2. **Dashboard View**: Real-time KPI summary (images scanned, videos processed, potholes detected, average confidence) and distribution charts.
3. **Image Detection**: Drag-and-drop image upload with adjustable confidence threshold, before/after visualizer, and instant download.
4. **Video Analysis**: Automated dashcam video pipeline displaying cumulative detections, peak detections per frame, and annotated video playback.
5. **Real-time Live Webcam**: Low-latency camera feed streaming directly to the YOLO engine with live FPS and bounding box HUD.
6. **Model Accuracy & Benchmarking**: Genuine confusion matrix, PR curves, and mAP metrics computed strictly on the test set.
7. **Dataset Architecture**: Visual breakdown of the 70/20/10 split and sample ground-truth annotated road images.
8. **Audit History**: Filterable, searchable SQLite detection log with direct result downloads and record deletion.
9. **About & Research**: Full technical documentation, system architecture, and academic references.

---

## 14. Project Directory Structure
```
RoadGuardAI/
├── backend/
│   ├── main.py              # FastAPI application & REST endpoints
│   ├── config.py            # Environment & path configuration
│   ├── database.py          # SQLite persistence & analytics
│   ├── detection.py         # YOLO inference & image annotation engine
│   ├── video_processing.py  # OpenCV video frame-by-frame pipeline
│   ├── requirements.txt     # Python package requirements
│   └── roadguard.db         # SQLite persistent database
│
├── frontend/
│   ├── index.html           # Master Single Page Application (10 views)
│   ├── css/
│   │   └── style.css        # Glassmorphism dark navy theme
│   ├── js/
│   │   ├── api.js           # Client-side API connector
│   │   └── app.js           # Navigation, Chart.js & Webcam stream
│   └── assets/              # Evaluation curves & dataset visualizations
│
├── datasets/
│   └── pothole/
│       ├── images/          # train (1,445), val (420), test (218)
│       ├── labels/          # train (1,445), val (420), test (218)
│       ├── visualizations/  # Ground-truth sample visual outputs
│       ├── data.yaml        # Ultralytics dataset definition
│       └── dataset_summary.json
│
├── models/
│   ├── best.pt              # Best trained YOLO model checkpoint
│   ├── last.pt              # Last training checkpoint
│   └── evaluation/          # Genuine test metrics & confusion matrix
│
├── training/
│   ├── prepare_dataset.py   # Extraction, verification & leakage-free split
│   ├── train.py             # CUDA-accelerated YOLO11n training engine
│   └── evaluate.py          # Independent test split benchmark script
│
├── uploads/                 # Temporary user upload staging
├── outputs/                 # Annotated images & processed videos
├── start_app.bat            # Windows 1-click startup script
├── run_app.ps1              # PowerShell startup script
└── README.md                # Complete academic documentation
```

---

## 15. Limitations & Future Scope
- **Current Limitations**: Performance under severe nighttime or heavy rainfall conditions depends on dashcam lighting and visibility.
- **Future Scope**:
  - Integration with vehicle CAN bus telemetry to cross-reference physical suspension accelerometer spikes with visual detections.
  - Geo-tagging via GPS coordinates for automated GIS road maintenance heatmaps.
  - Multi-class expansion to classify severe road cracks, debris, and missing manhole covers.

"""
RoadGuardAI - Backend Configuration
"""

import os
from pathlib import Path

# Disable Ultralytics online checks and telemetry for fast offline inference
os.environ["YOLO_VERBOSE"] = "False"
os.environ["ULTRALYTICS_OFFLINE"] = "True"

BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = BASE_DIR / "datasets" / "pothole"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BACKEND_DIR / "data"
DATABASE_PATH = BACKEND_DIR / "roadguard.db"

BEST_MODEL_PATH = MODELS_DIR / "best.pt"
DATA_YAML_PATH = DATASETS_DIR / "data.yaml"

# Ensure all critical runtime directories exist
for directory in [UPLOADS_DIR, OUTPUTS_DIR, MODELS_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Inference defaults
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45

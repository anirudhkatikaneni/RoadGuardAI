"""
RoadGuardAI - YOLO Model Training Engine
Phase 3 Implementation
"""

import os
import sys
import argparse
import shutil
import json
import time
from pathlib import Path
import torch
from ultralytics import YOLO

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = WORKSPACE_DIR / "datasets" / "pothole" / "data.yaml"
MODELS_DIR = WORKSPACE_DIR / "models"
RUNS_DIR = WORKSPACE_DIR / "runs" / "detect" / "roadguard_train"

def get_optimal_device():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[+] CUDA GPU detected: {gpu_name}")
        return "0"
    print("[!] No CUDA GPU detected; falling back to CPU.")
    return "cpu"

def train(
    data_yaml=str(DATA_YAML),
    model_name="yolo11n.pt",
    epochs=25,
    imgsz=640,
    batch=16,
    lr0=0.01,
    device=None,
    workers=2,
    name="roadguard_model"
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    if device is None:
        device = get_optimal_device()
        
    print("\n" + "="*60)
    print("ROADGUARDAI MODEL TRAINING PIPELINE")
    print("="*60)
    print(f"Dataset config:     {data_yaml}")
    print(f"Base Model:         {model_name}")
    print(f"Target Device:      {device} ({torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'})")
    print(f"Epochs:             {epochs}")
    print(f"Image Size:         {imgsz}")
    print(f"Batch Size:         {batch}")
    print(f"Initial LR:         {lr0}")
    print("="*60 + "\n")
    
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}. Run prepare_dataset.py first!")
        
    start_time = time.time()
    
    # Initialize YOLO with pretrained lightweight weights
    print(f"[*] Loading pretrained weights: {model_name}...")
    model = YOLO(model_name)
    
    # Train the model
    print("[*] Starting model training...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        device=device,
        workers=workers,
        project=str(WORKSPACE_DIR / "runs" / "detect"),
        name=name,
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True
    )
    
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"\n[+] Training completed in {minutes}m {seconds}s!")
    
    # Locate saved weights
    save_dir = Path(model.trainer.save_dir)
    best_weight = save_dir / "weights" / "best.pt"
    last_weight = save_dir / "weights" / "last.pt"
    
    target_best = MODELS_DIR / "best.pt"
    target_last = MODELS_DIR / "last.pt"
    
    if best_weight.exists():
        shutil.copy2(best_weight, target_best)
        print(f"[+] Saved best weights to: {target_best}")
    if last_weight.exists():
        shutil.copy2(last_weight, target_last)
        print(f"[+] Saved last weights to: {target_last}")
        
    # Copy training graphs to models and frontend
    frontend_viz = WORKSPACE_DIR / "frontend" / "assets" / "training"
    frontend_viz.mkdir(parents=True, exist_ok=True)
    
    for graph in ["results.png", "confusion_matrix.png", "PR_curve.png", "F1_curve.png"]:
        src = save_dir / graph
        if src.exists():
            shutil.copy2(src, MODELS_DIR / graph)
            shutil.copy2(src, frontend_viz / graph)
            
    # Record training metadata
    training_summary = {
        "status": "trained",
        "model_variant": model_name,
        "epochs": epochs,
        "batch_size": batch,
        "imgsz": imgsz,
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device != "cpu" else "CPU",
        "training_duration_seconds": round(elapsed_time, 2),
        "training_duration_human": f"{minutes}m {seconds}s",
        "best_model_path": str(target_best),
        "last_model_path": str(target_last),
        "save_dir": str(save_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    summary_path = MODELS_DIR / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=4)
        
    (WORKSPACE_DIR / "backend" / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy2(summary_path, WORKSPACE_DIR / "backend" / "data" / "training_summary.json")
    
    print("\n" + "="*60)
    print("TRAINING SUCCESSFUL!")
    print(f"Model checkpoint: {target_best}")
    print("="*60 + "\n")
    return training_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RoadGuardAI Pothole Detection Model")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs (default: 25)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Pretrained model")
    parser.add_argument("--device", type=str, default=None, help="Device (0 for GPU, cpu for CPU)")
    
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        model_name=args.model,
        device=args.device
    )

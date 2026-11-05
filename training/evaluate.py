"""
RoadGuardAI - YOLO Model Evaluation Engine
Phase 4 Implementation
"""

import os
import sys
import json
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = WORKSPACE_DIR / "datasets" / "pothole" / "data.yaml"
MODELS_DIR = WORKSPACE_DIR / "models"
BEST_MODEL = MODELS_DIR / "best.pt"
EVAL_DIR = MODELS_DIR / "evaluation"
FRONTEND_EVAL_DIR = WORKSPACE_DIR / "frontend" / "assets" / "evaluation"
BACKEND_DATA_DIR = WORKSPACE_DIR / "backend" / "data"

def evaluate_model(weights_path=str(BEST_MODEL), data_yaml=str(DATA_YAML), device=None):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}. Train the model first!")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}!")
        
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    FRONTEND_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    BACKEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
        
    print("\n" + "="*60)
    print("ROADGUARDAI MODEL TEST EVALUATION")
    print("="*60)
    print(f"Model checkpoint: {weights_path}")
    print(f"Dataset YAML:     {data_yaml}")
    print(f"Evaluation split: test (Separate unseen test set)")
    print(f"Device:           {device}")
    print("="*60 + "\n")
    
    model = YOLO(weights_path)
    
    # Run genuine validation on the dedicated test split
    metrics = model.val(
        data=data_yaml,
        split="test",
        device=device,
        project=str(WORKSPACE_DIR / "runs" / "detect"),
        name="test_eval",
        exist_ok=True,
        save_json=True,
        plots=True,
        verbose=True
    )
    
    # Extract genuine detection metrics
    # box.mp: mean precision
    # box.mr: mean recall
    # box.map50: mAP@0.50
    # box.map: mAP@0.50:0.95
    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    f1 = float(2 * (mp * mr) / (mp + mr + 1e-8))
    
    eval_dir_path = Path(metrics.save_dir)
    
    # Per-class metrics
    class_metrics = {}
    class_names = model.names
    for i, name in class_names.items():
        class_p = float(metrics.box.p[i]) if hasattr(metrics.box, 'p') and len(metrics.box.p) > i else mp
        class_r = float(metrics.box.r[i]) if hasattr(metrics.box, 'r') and len(metrics.box.r) > i else mr
        class_map50 = float(metrics.box.all_ap[i, 0]) if hasattr(metrics.box, 'all_ap') and metrics.box.all_ap.shape[0] > i else map50
        class_metrics[name] = {
            "precision": round(class_p, 4),
            "recall": round(class_r, 4),
            "f1_score": round(2 * (class_p * class_r) / (class_p + class_r + 1e-8), 4),
            "mAP50": round(class_map50, 4)
        }
        
    evaluation_results = {
        "evaluation_split": "test",
        "dataset_name": "RoadGuardAI Pothole Test Set",
        "total_test_images": metrics.nt_per_image.shape[0] if hasattr(metrics, 'nt_per_image') else 218,
        "metrics": {
            "precision": round(mp, 4),
            "recall": round(mr, 4),
            "f1_score": round(f1, 4),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4)
        },
        "per_class": class_metrics,
        "metrics_explanation": {
            "precision": "Proportion of detected potholes that were actually potholes (True Positives / (True Positives + False Positives)).",
            "recall": "Proportion of all true road potholes that were successfully detected (True Positives / (True Positives + False Negatives)).",
            "f1_score": "Harmonic mean of precision and recall balancing detection accuracy and coverage.",
            "mAP50": "Mean Average Precision at IoU threshold 0.50 - standard object detection accuracy benchmark.",
            "mAP50_95": "Mean Average Precision averaged over IoU thresholds from 0.50 to 0.95 in steps of 0.05 - rigorous localization benchmark.",
            "classification_vs_detection_note": "In classification, accuracy simply measures whether an entire image was categorized correctly. In object detection, accuracy depends on localization (predicting precise bounding box coordinates with high Intersection-over-Union) as well as correct class identification. mAP is the gold-standard academic metric."
        },
        "artifacts_generated": []
    }
    
    # Copy evaluation artifacts (plots, curves, confusion matrix)
    eval_artifacts = [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
        "results.csv",
        "val_batch0_labels.jpg",
        "val_batch0_pred.jpg"
    ]
    
    for art in eval_artifacts:
        src = eval_dir_path / art
        if src.exists():
            dest1 = EVAL_DIR / art
            dest2 = FRONTEND_EVAL_DIR / art
            shutil.copy2(src, dest1)
            shutil.copy2(src, dest2)
            evaluation_results["artifacts_generated"].append(art)
            
    # Save metrics JSON
    json_path = EVAL_DIR / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(evaluation_results, jf, indent=4)
        
    shutil.copy2(json_path, FRONTEND_EVAL_DIR / "metrics.json")
    shutil.copy2(json_path, BACKEND_DATA_DIR / "metrics.json")
    
    print("\n" + "="*60)
    print("TEST EVALUATION RESULTS (GENUINE METRICS)")
    print("="*60)
    print(f"Precision:   {evaluation_results['metrics']['precision'] * 100:.2f}%")
    print(f"Recall:      {evaluation_results['metrics']['recall'] * 100:.2f}%")
    print(f"F1-Score:    {evaluation_results['metrics']['f1_score'] * 100:.2f}%")
    print(f"mAP@50:      {evaluation_results['metrics']['mAP50'] * 100:.2f}%")
    print(f"mAP@50-95:   {evaluation_results['metrics']['mAP50_95'] * 100:.2f}%")
    print("="*60)
    print(f"Evaluation artifacts saved to: {EVAL_DIR}")
    print("="*60 + "\n")
    
    return evaluation_results

if __name__ == "__main__":
    evaluate_model()

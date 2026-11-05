"""
RoadGuardAI - Dataset Ingestion, Inspection, Splitting, and Visualization
Phase 2 Implementation
"""

import os
import sys
import glob
import json
import shutil
import hashlib
import zipfile
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
ARCHIVE_ZIP = WORKSPACE_DIR / "archive.zip"
DATASETS_DIR = WORKSPACE_DIR / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
POTHOLE_DIR = DATASETS_DIR / "pothole"
VIZ_DIR = POTHOLE_DIR / "visualizations"
FRONTEND_VIZ_DIR = WORKSPACE_DIR / "frontend" / "assets" / "dataset_viz"

def compute_file_md5(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def extract_archive():
    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        print(f"[*] Archive already extracted at {RAW_DIR}. Skipping re-extraction.")
        return
    print(f"[*] Extracting {ARCHIVE_ZIP} into {RAW_DIR}...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE_ZIP, "r") as z:
        z.extractall(RAW_DIR)
    print(f"[+] Extracted successfully to {RAW_DIR}")

def inspect_and_prepare():
    print("\n" + "="*60)
    print("ROADGUARDAI DATASET INSPECTION & PREPARATION")
    print("="*60)
    
    # 1. Discover all raw images and labels (deduplicate paths for Windows case-insensitivity)
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    raw_images_set = set()
    for ext in img_extensions:
        for p in RAW_DIR.rglob(f"*{ext}"):
            raw_images_set.add(p.resolve())
    raw_images = sorted(list(raw_images_set), key=lambda x: str(x))
    
    raw_labels_set = set()
    for p in RAW_DIR.rglob("*.txt"):
        if not p.name.startswith("README") and not p.name.startswith("LICENSE"):
            raw_labels_set.add(p.resolve())
    raw_labels = sorted(list(raw_labels_set), key=lambda x: str(x))
    
    print(f"Total raw image files found: {len(raw_images)}")
    print(f"Total raw label files found: {len(raw_labels)}")
    
    # Map label files by stem
    label_map = {}
    for lbl in raw_labels:
        label_map[lbl.stem] = lbl
        
    # Check images
    image_formats = {}
    image_dimensions = []
    corrupted_images = []
    duplicate_hashes = {}
    duplicates = []
    missing_labels = []
    empty_annotations = 0
    total_bounding_boxes = 0
    invalid_annotations = 0
    class_ids_found = set()
    
    valid_pairs = []
    
    for img_path in raw_images:
        fmt = img_path.suffix.lower()
        image_formats[fmt] = image_formats.get(fmt, 0) + 1
        
        # Check corruption and dimensions
        try:
            with Image.open(img_path) as im:
                w, h = im.size
                image_dimensions.append((w, h))
        except Exception as e:
            corrupted_images.append(str(img_path))
            continue
            
        # Check duplicate
        file_hash = compute_file_md5(img_path)
        if file_hash in duplicate_hashes:
            duplicates.append((str(img_path), duplicate_hashes[file_hash]))
        else:
            duplicate_hashes[file_hash] = str(img_path)
            
        # Check label match
        lbl_path = label_map.get(img_path.stem)
        if not lbl_path or not lbl_path.exists():
            missing_labels.append(str(img_path))
            continue
            
        # Check annotation format and bounding boxes
        with open(lbl_path, "r", encoding="utf-8") as lf:
            lines = [line.strip() for line in lf.readlines() if line.strip()]
            
        if len(lines) == 0:
            empty_annotations += 1
            
        bboxes = []
        file_has_invalid = False
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                invalid_annotations += 1
                file_has_invalid = True
                continue
            try:
                cls_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Check normalized boundaries
                if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                    invalid_annotations += 1
                    file_has_invalid = True
                else:
                    class_ids_found.add(cls_id)
                    total_bounding_boxes += 1
                    bboxes.append((cls_id, x_center, y_center, width, height))
            except ValueError:
                invalid_annotations += 1
                file_has_invalid = True
                
        valid_pairs.append({
            "img_path": img_path,
            "lbl_path": lbl_path,
            "stem": img_path.stem,
            "bboxes": bboxes,
            "size": (w, h)
        })

    # Group by base identity to prevent data leakage from augmented variations
    groups = {}
    for item in valid_pairs:
        # Roboflow exports usually use filename_ext.rf.<hash> or original filename
        stem = item["stem"]
        if ".rf." in stem:
            base_id = stem.split(".rf.")[0]
        elif "_jpg.rf." in stem:
            base_id = stem.split("_jpg.rf.")[0]
        else:
            base_id = stem.split("_")[0] if "_" in stem else stem
            
        groups.setdefault(base_id, []).append(item)
        
    print(f"\nUnique base scenes/groups identified (for leakage prevention): {len(groups)}")
    
    # Deterministic split: 70% train, 20% val, 10% test
    # Use deterministic shuffle with fixed seed 42
    import random
    rng = random.Random(42)
    group_keys = sorted(list(groups.keys()))
    rng.shuffle(group_keys)
    
    n_groups = len(group_keys)
    n_train = int(0.70 * n_groups)
    n_val = int(0.20 * n_groups)
    n_test = n_groups - n_train - n_val
    
    train_groups = set(group_keys[:n_train])
    val_groups = set(group_keys[n_train:n_train + n_val])
    test_groups = set(group_keys[n_train + n_val:])
    
    splits = {"train": [], "val": [], "test": []}
    for base_id, items in groups.items():
        if base_id in train_groups:
            splits["train"].extend(items)
        elif base_id in val_groups:
            splits["val"].extend(items)
        else:
            splits["test"].extend(items)
            
    # Setup target directories (clean up old split files if any)
    for folder in ["images", "labels"]:
        target_sub = POTHOLE_DIR / folder
        if target_sub.exists():
            shutil.rmtree(target_sub)
    for split_name in ["train", "val", "test"]:
        (POTHOLE_DIR / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (POTHOLE_DIR / "labels" / split_name).mkdir(parents=True, exist_ok=True)
        
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    FRONTEND_VIZ_DIR.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    print("\n[*] Organizing dataset into train/val/test splits...")
    split_counts = {}
    split_bbox_counts = {}
    for split_name, items in splits.items():
        split_counts[split_name] = len(items)
        split_bbox_counts[split_name] = sum(len(it["bboxes"]) for it in items)
        
        for it in items:
            dest_img = POTHOLE_DIR / "images" / split_name / it["img_path"].name
            dest_lbl = POTHOLE_DIR / "labels" / split_name / it["lbl_path"].name
            shutil.copy2(it["img_path"], dest_img)
            shutil.copy2(it["lbl_path"], dest_lbl)
            
    # Create data.yaml
    data_yaml_path = POTHOLE_DIR / "data.yaml"
    # YOLO ultralytics requires paths relative or absolute
    # Using forward slashes for cross-platform compatibility
    yaml_content = f"""# RoadGuardAI Pothole Dataset Configuration
path: {POTHOLE_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['pothole']
"""
    with open(data_yaml_path, "w", encoding="utf-8") as yf:
        yf.write(yaml_content)
        
    # Also save in training/
    shutil.copy2(data_yaml_path, WORKSPACE_DIR / "training" / "dataset.yaml")
    
    # 2.4 Dataset Visualization
    print("\n[*] Generating sample ground-truth visualizations...")
    sample_items = splits["val"][:6] if len(splits["val"]) >= 6 else valid_pairs[:6]
    viz_images_saved = []
    
    for idx, it in enumerate(sample_items):
        img_bgr = cv2.imread(str(it["img_path"]))
        if img_bgr is None:
            continue
        h, w, _ = img_bgr.shape
        annotated = img_bgr.copy()
        
        for cls_id, xc, yc, bw, bh in it["bboxes"]:
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            
            # Pothole bounding box: Vibrant Orange-Red / Amber
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 140, 255), 2)
            label_text = f"Pothole"
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), (0, 140, 255), -1)
            cv2.putText(annotated, label_text, (x1 + 2, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
        viz_name = f"sample_groundtruth_{idx+1}.jpg"
        pothole_viz_path = VIZ_DIR / viz_name
        frontend_viz_path = FRONTEND_VIZ_DIR / viz_name
        cv2.imwrite(str(pothole_viz_path), annotated)
        cv2.imwrite(str(frontend_viz_path), annotated)
        viz_images_saved.append(viz_name)

    # Calculate summary metrics
    widths = [d[0] for d in image_dimensions] if image_dimensions else [0]
    heights = [d[1] for d in image_dimensions] if image_dimensions else [0]
    
    summary = {
        "dataset_name": "RoadGuardAI Pothole Detection Dataset",
        "total_images": len(valid_pairs),
        "total_annotations": total_bounding_boxes,
        "image_formats": image_formats,
        "dimension_stats": {
            "min_width": int(min(widths)),
            "max_width": int(max(widths)),
            "min_height": int(min(heights)),
            "max_height": int(max(heights)),
            "avg_width": int(sum(widths) / len(widths)) if widths else 0,
            "avg_height": int(sum(heights) / len(heights)) if heights else 0
        },
        "annotation_format": "YOLO normalized format (class x_center y_center width height)",
        "num_classes": len(class_ids_found) if class_ids_found else 1,
        "class_names": ["pothole"],
        "splits": {
            "train": {
                "images": split_counts.get("train", 0),
                "annotations": split_bbox_counts.get("train", 0),
                "percentage": round(split_counts.get("train", 0) / len(valid_pairs) * 100, 1) if valid_pairs else 0
            },
            "val": {
                "images": split_counts.get("val", 0),
                "annotations": split_bbox_counts.get("val", 0),
                "percentage": round(split_counts.get("val", 0) / len(valid_pairs) * 100, 1) if valid_pairs else 0
            },
            "test": {
                "images": split_counts.get("test", 0),
                "annotations": split_bbox_counts.get("test", 0),
                "percentage": round(split_counts.get("test", 0) / len(valid_pairs) * 100, 1) if valid_pairs else 0
            }
        },
        "data_quality_report": {
            "missing_labels_count": len(missing_labels),
            "empty_annotation_files_count": empty_annotations,
            "corrupted_images_count": len(corrupted_images),
            "duplicate_images_count": len(duplicates),
            "invalid_annotations_count": invalid_annotations,
            "data_leakage_prevented": True,
            "unique_scenes": len(groups)
        },
        "sample_visualizations": viz_images_saved,
        "data_yaml_path": str(data_yaml_path)
    }
    
    # Save summary JSON
    summary_path = POTHOLE_DIR / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=4)
        
    # Also copy to frontend assets and backend directory for the API
    shutil.copy2(summary_path, FRONTEND_VIZ_DIR / "dataset_summary.json")
    (WORKSPACE_DIR / "backend" / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy2(summary_path, WORKSPACE_DIR / "backend" / "data" / "dataset_summary.json")
    
    print("\n" + "="*60)
    print("DATASET PREPARATION COMPLETE!")
    print("="*60)
    print(f"Total verified images:      {summary['total_images']}")
    print(f"Total verified annotations: {summary['total_annotations']}")
    print(f"Train split:                {summary['splits']['train']['images']} images ({summary['splits']['train']['percentage']}%)")
    print(f"Validation split:           {summary['splits']['val']['images']} images ({summary['splits']['val']['percentage']}%)")
    print(f"Test split:                 {summary['splits']['test']['images']} images ({summary['splits']['test']['percentage']}%)")
    print(f"data.yaml generated at:     {data_yaml_path}")
    print(f"Visualizations saved to:    {VIZ_DIR}")
    print("="*60 + "\n")
    
    return summary

if __name__ == "__main__":
    extract_archive()
    inspect_and_prepare()

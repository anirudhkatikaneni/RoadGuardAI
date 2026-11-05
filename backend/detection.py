"""
RoadGuardAI - Detection & Inference Engine
Phase 5 Implementation
"""

import os
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO, settings

# Ensure offline settings
try:
    settings.update({'sync': False, 'auto_update': False})
except Exception:
    pass

from backend.config import (
    BEST_MODEL_PATH,
    OUTPUTS_DIR,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD
)

class PotholeDetector:
    _instance: Optional['PotholeDetector'] = None
    _model: Optional[YOLO] = None
    _model_loaded: bool = False
    _model_path: Optional[str] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PotholeDetector, cls).__new__(cls)
        return cls._instance

    def load_model(self, model_path: Optional[str] = None) -> bool:
        path_to_use = Path(model_path) if model_path else BEST_MODEL_PATH
        if not path_to_use.exists():
            self._model_loaded = False
            self._model = None
            return False
            
        try:
            print(f"[*] Loading detection model from {path_to_use}...")
            self._model = YOLO(str(path_to_use))
            self._model_path = str(path_to_use)
            self._model_loaded = True
            print("[+] Model loaded successfully!")
            return True
        except Exception as e:
            print(f"[!] Error loading YOLO model: {e}")
            self._model_loaded = False
            self._model = None
            return False

    def is_ready(self) -> bool:
        if self._model is not None and self._model_loaded:
            return True
        if BEST_MODEL_PATH.exists():
            return self.load_model()
        return False

    def get_model_info(self) -> Dict[str, Any]:
        ready = self.is_ready()
        return {
            "model_ready": ready,
            "weights_path": str(BEST_MODEL_PATH) if ready else None,
            "weights_exist": BEST_MODEL_PATH.exists(),
            "device": "CUDA" if (self._model and next(self._model.model.parameters()).is_cuda) else "CPU" if self._model else "None"
        }

    def detect_image(
        self,
        image_input, # can be file path (str/Path) or numpy BGR array
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        save_result: bool = True
    ) -> Dict[str, Any]:
        if not self.is_ready():
            return {
                "status": "error",
                "message": "Model weights best.pt not found. Please train the model first.",
                "pothole_count": 0,
                "boxes": [],
                "avg_confidence": 0.0,
                "processing_time_ms": 0.0
            }
            
        # Read image
        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input))
            if img_bgr is None:
                return {"status": "error", "message": "Failed to decode image file."}
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input.copy()
        else:
            return {"status": "error", "message": "Unsupported image format."}
            
        h, w = img_bgr.shape[:2]
        start_time = time.perf_counter()
        
        # Real YOLO Inference
        results = self._model.predict(
            source=img_bgr,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0
        
        boxes_data = []
        annotated_img = img_bgr.copy()
        confidences = []
        
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                coords = box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                x1, y1, x2, y2 = map(int, coords)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                
                confidences.append(conf)
                boxes_data.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": round(conf, 4),
                    "class_name": "pothole"
                })
                
                # Visual styling: Premium glowing Amber/Orange bounding box
                box_color = (0, 140, 255) # BGR
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 2)
                
                # Label badge
                badge_text = f"Pothole {int(conf * 100)}%"
                font_scale = 0.5
                thickness = 1
                (tw, th), baseline = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Draw badge background
                badge_y1 = max(0, y1 - th - baseline - 6)
                badge_y2 = y1
                cv2.rectangle(annotated_img, (x1, badge_y1), (x1 + tw + 8, badge_y2), box_color, -1)
                cv2.putText(
                    annotated_img,
                    badge_text,
                    (x1 + 4, y1 - baseline - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA
                )
                
        pothole_count = len(boxes_data)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        
        # Overlay HUD counter banner on top-left of the image
        hud_bg = (15, 23, 42) # Dark navy
        cv2.rectangle(annotated_img, (15, 15), (270, 65), hud_bg, -1)
        cv2.rectangle(annotated_img, (15, 15), (270, 65), (0, 215, 255), 1)
        hud_text = f"Potholes Detected: {pothole_count}"
        cv2.putText(annotated_img, hud_text, (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated_img, f"Inference: {inference_time_ms:.1f}ms", (25, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        output_filename = None
        output_url = None
        
        if save_result:
            output_filename = f"detect_{uuid.uuid4().hex[:10]}.jpg"
            save_path = OUTPUTS_DIR / output_filename
            cv2.imwrite(str(save_path), annotated_img)
            output_url = f"/api/results/{output_filename}"
            
        return {
            "status": "success",
            "pothole_count": pothole_count,
            "boxes": boxes_data,
            "avg_confidence": round(avg_conf, 4),
            "processing_time_ms": round(inference_time_ms, 2),
            "output_filename": output_filename,
            "output_url": output_url,
            "annotated_image": annotated_img
        }

# Global singleton detector
detector = PotholeDetector()

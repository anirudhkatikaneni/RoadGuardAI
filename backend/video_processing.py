"""
RoadGuardAI - Video Processing Pipeline
Phase 8 Implementation
"""

import os
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import cv2
import numpy as np

from backend.config import OUTPUTS_DIR, DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD
from backend.detection import detector

def process_video_file(
    input_video_path: str,
    output_filename: Optional[str] = None,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    max_fps_limit: Optional[int] = 30,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, Any]:
    if not os.path.exists(input_video_path):
        return {"status": "error", "message": f"Video file not found: {input_video_path}"}
        
    if not detector.is_ready():
        return {"status": "error", "message": "Model weights best.pt not found. Please train model first."}
        
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return {"status": "error", "message": "Failed to open video file with OpenCV."}
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    
    if not output_filename:
        output_filename = f"video_detect_{uuid.uuid4().hex[:10]}.mp4"
    output_path = OUTPUTS_DIR / output_filename
    thumbnail_filename = f"thumb_{output_filename}.jpg"
    thumbnail_path = OUTPUTS_DIR / thumbnail_filename
    
    # Try H264 or MP4V codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_idx = 0
    total_detections_sum = 0
    max_potholes_in_frame = 0
    confidences = []
    start_time = time.perf_counter()
    best_thumbnail_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        
        # Run detection without saving each frame as an individual image
        res = detector.detect_image(
            frame,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            save_result=False
        )
        
        count = res.get("pothole_count", 0)
        total_detections_sum += count
        if res.get("avg_confidence", 0) > 0:
            confidences.append(res["avg_confidence"])
            
        annotated_frame = res.get("annotated_image", frame)
        
        # Add frame index & video info banner
        info_text = f"Frame: {frame_idx}/{total_frames} | FPS: {fps:.1f}"
        cv2.putText(annotated_frame, info_text, (25, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        out.write(annotated_frame)
        
        if count >= max_potholes_in_frame:
            max_potholes_in_frame = count
            best_thumbnail_frame = annotated_frame.copy()
            
        if progress_callback and frame_idx % 10 == 0:
            progress_callback(frame_idx, total_frames)
            
    cap.release()
    out.release()
    
    total_time_s = time.perf_counter() - start_time
    avg_fps = (frame_idx / total_time_s) if total_time_s > 0 else 0.0
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    
    # Save thumbnail
    if best_thumbnail_frame is not None:
        cv2.imwrite(str(thumbnail_path), best_thumbnail_frame)
    else:
        # Save a black frame or placeholder
        placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.imwrite(str(thumbnail_path), placeholder)
        
    return {
        "status": "success",
        "output_filename": output_filename,
        "output_url": f"/api/results/{output_filename}",
        "thumbnail_filename": thumbnail_filename,
        "thumbnail_url": f"/api/results/{thumbnail_filename}",
        "total_frames": frame_idx,
        "total_potholes_detected": total_detections_sum,
        "max_potholes_in_single_frame": max_potholes_in_frame,
        "average_confidence": round(avg_conf, 4),
        "processing_time_seconds": round(total_time_s, 2),
        "average_fps": round(avg_fps, 2),
        "resolution": f"{width}x{height}"
    }

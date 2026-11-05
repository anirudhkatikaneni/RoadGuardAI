"""
RoadGuardAI - FastAPI Web Application
Phase 6 Implementation
"""

import os
import shutil
import base64
import uuid
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import (
    UPLOADS_DIR,
    OUTPUTS_DIR,
    FRONTEND_DIR,
    MODELS_DIR,
    DATA_DIR,
    BEST_MODEL_PATH,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD
)
from backend.database import (
    add_detection_record,
    get_detection_history,
    delete_detection_record,
    get_dashboard_statistics
)
from backend.detection import detector
from backend.video_processing import process_video_file

app = FastAPI(
    title="RoadGuardAI API",
    description="Intelligent Road Safety and Pothole Detection Platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 0. Root Web Interface
@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file), media_type="text/html")
    raise HTTPException(status_code=404, detail="index.html not found.")

# 1. Health check
@app.get("/api/health")
def health_check():
    model_info = detector.get_model_info()
    return {
        "status": "healthy",
        "service": "RoadGuardAI Detection Service",
        "model_loaded": model_info["model_ready"],
        "device": model_info["device"],
        "database": "connected"
    }

# 2. Dashboard metrics
@app.get("/api/dashboard")
def get_dashboard():
    stats = get_dashboard_statistics()
    model_info = detector.get_model_info()
    stats["model_info"] = model_info
    return stats

# 3. Model status
@app.get("/api/model/status")
def model_status():
    info = detector.get_model_info()
    training_summary_path = DATA_DIR / "training_summary.json"
    summary = {}
    if training_summary_path.exists():
        import json
        with open(training_summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    return {
        "model_ready": info["model_ready"],
        "device": info["device"],
        "weights_path": info["weights_path"],
        "training_metadata": summary
    }

# 4. Model metrics
@app.get("/api/model/metrics")
def model_metrics():
    metrics_path = DATA_DIR / "metrics.json"
    if not metrics_path.exists():
        metrics_path = MODELS_DIR / "evaluation" / "metrics.json"
        
    if not metrics_path.exists():
        return {
            "status": "not_evaluated",
            "message": "Model evaluation has not been completed yet."
        }
        
    import json
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# 5. Dataset summary
@app.get("/api/dataset/summary")
def dataset_summary():
    summary_path = DATA_DIR / "dataset_summary.json"
    if not summary_path.exists():
        summary_path = Path(__file__).resolve().parent.parent / "datasets" / "pothole" / "dataset_summary.json"
        
    if not summary_path.exists():
        return {"status": "not_ready", "message": "Dataset summary not found."}
        
    import json
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# 6. Image detection
@app.post("/api/detect/image")
async def detect_image_endpoint(
    file: UploadFile = File(...),
    conf_threshold: float = Form(DEFAULT_CONF_THRESHOLD),
    iou_threshold: float = Form(DEFAULT_IOU_THRESHOLD)
):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported image format: {ext}")
        
    # Save uploaded file
    upload_filename = f"upload_{uuid.uuid4().hex[:8]}_{file.filename}"
    upload_path = UPLOADS_DIR / upload_filename
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Run YOLO detection
    result = detector.detect_image(
        image_input=upload_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        save_result=True
    )
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message", "Inference error"))
        
    # Save to history database
    record_id = add_detection_record(
        file_name=file.filename,
        detection_type="image",
        pothole_count=result["pothole_count"],
        avg_confidence=result["avg_confidence"],
        processing_time_ms=result["processing_time_ms"],
        result_path=result["output_filename"],
        thumbnail_path=result["output_filename"],
        details={"boxes": result["boxes"], "conf_threshold": conf_threshold}
    )
    result["record_id"] = record_id
    
    # Remove binary image array before returning JSON
    if "annotated_image" in result:
        del result["annotated_image"]
        
    return result

# 7. Video detection
@app.post("/api/detect/video")
async def detect_video_endpoint(
    file: UploadFile = File(...),
    conf_threshold: float = Form(DEFAULT_CONF_THRESHOLD),
    iou_threshold: float = Form(DEFAULT_IOU_THRESHOLD)
):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported video format: {ext}")
        
    upload_filename = f"upload_video_{uuid.uuid4().hex[:8]}_{file.filename}"
    upload_path = UPLOADS_DIR / upload_filename
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Process video
    result = process_video_file(
        input_video_path=str(upload_path),
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message", "Video processing error"))
        
    # Record to history
    record_id = add_detection_record(
        file_name=file.filename,
        detection_type="video",
        pothole_count=result["total_potholes_detected"],
        avg_confidence=result["average_confidence"],
        processing_time_ms=result["processing_time_seconds"] * 1000.0,
        result_path=result["output_filename"],
        thumbnail_path=result["thumbnail_filename"],
        details=result
    )
    result["record_id"] = record_id
    return result

# 8. Real-time webcam frame detection
@app.post("/api/detect/frame")
async def detect_frame_endpoint(
    file: Optional[UploadFile] = File(None),
    frame_base64: Optional[str] = Form(None),
    conf_threshold: float = Form(DEFAULT_CONF_THRESHOLD)
):
    img_bgr = None
    if file:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif frame_base64:
        if "," in frame_base64:
            frame_base64 = frame_base64.split(",")[1]
        decoded = base64.b64decode(frame_base64)
        nparr = np.frombuffer(decoded, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        raise HTTPException(status_code=400, detail="Either file or frame_base64 is required.")
        
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Failed to decode webcam frame.")
        
    result = detector.detect_image(
        image_input=img_bgr,
        conf_threshold=conf_threshold,
        save_result=False
    )
    
    # Encode annotated frame back to JPEG base64 for fast display
    annotated = result.get("annotated_image")
    annotated_b64 = None
    if annotated is not None:
        _, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
        del result["annotated_image"]
        
    result["annotated_frame_base64"] = annotated_b64
    return result

# 9. Detection history
@app.get("/api/history")
def get_history(limit: int = 50, offset: int = 0, type: Optional[str] = None):
    records = get_detection_history(limit=limit, offset=offset, detection_type=type)
    return {"records": records, "count": len(records)}

# 10. Delete history item
@app.delete("/api/history/{record_id}")
def delete_history_item(record_id: int):
    success = delete_detection_record(record_id)
    if not success:
        raise HTTPException(status_code=404, detail="Record not found.")
    return {"status": "success", "deleted_id": record_id}

# 11. Results files serving
@app.get("/api/results/{filename}")
def serve_result_file(filename: str):
    # Sanitize filename to prevent directory traversal
    safe_filename = Path(filename).name
    file_path = OUTPUTS_DIR / safe_filename
    if not file_path.exists():
        # Check in evaluation or visualizations
        file_path = MODELS_DIR / "evaluation" / safe_filename
        if not file_path.exists():
            file_path = Path(__file__).resolve().parent.parent / "datasets" / "pothole" / "visualizations" / safe_filename
            if not file_path.exists():
                raise HTTPException(status_code=404, detail="File not found.")
                
    return FileResponse(path=str(file_path))

# Mount frontend asset subdirectories
if (FRONTEND_DIR / "css").exists():
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
if (FRONTEND_DIR / "js").exists():
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")
if (FRONTEND_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8080, reload=True)

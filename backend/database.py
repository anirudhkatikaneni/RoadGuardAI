"""
RoadGuardAI - SQLite Database Layer
Phase 11 Implementation
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from backend.config import DATABASE_PATH

def get_connection():
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            detection_type TEXT NOT NULL,
            pothole_count INTEGER NOT NULL DEFAULT 0,
            avg_confidence REAL NOT NULL DEFAULT 0.0,
            processing_time_ms REAL NOT NULL DEFAULT 0.0,
            created_at TEXT NOT NULL,
            result_path TEXT NOT NULL,
            thumbnail_path TEXT,
            details_json TEXT
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_detection_type ON detection_history(detection_type)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_created_at ON detection_history(created_at)
    """)
    conn.commit()
    conn.close()

def add_detection_record(
    file_name: str,
    detection_type: str,
    pothole_count: int,
    avg_confidence: float,
    processing_time_ms: float,
    result_path: str,
    thumbnail_path: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    details_str = json.dumps(details or {})
    
    cursor.execute("""
        INSERT INTO detection_history 
        (file_name, detection_type, pothole_count, avg_confidence, processing_time_ms, created_at, result_path, thumbnail_path, details_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        file_name,
        detection_type,
        pothole_count,
        round(avg_confidence, 4),
        round(processing_time_ms, 2),
        now_str,
        result_path,
        thumbnail_path or result_path,
        details_str
    ))
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id

def get_detection_history(limit: int = 50, offset: int = 0, detection_type: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM detection_history"
    params = []
    
    if detection_type:
        query += " WHERE detection_type = ?"
        params.append(detection_type)
        
    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "file_name": r["file_name"],
            "detection_type": r["detection_type"],
            "pothole_count": r["pothole_count"],
            "avg_confidence": r["avg_confidence"],
            "processing_time_ms": r["processing_time_ms"],
            "created_at": r["created_at"],
            "result_path": r["result_path"],
            "thumbnail_path": r["thumbnail_path"],
            "details": json.loads(r["details_json"]) if r["details_json"] else {}
        })
    return results

def delete_detection_record(record_id: int) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM detection_history WHERE id = ?", (record_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def get_dashboard_statistics() -> Dict[str, Any]:
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM detection_history WHERE detection_type = 'image'")
    total_images = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM detection_history WHERE detection_type = 'video'")
    total_videos = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM detection_history WHERE detection_type = 'webcam'")
    total_webcam = cursor.fetchone()[0]
    
    cursor.execute("SELECT SUM(pothole_count) FROM detection_history")
    total_potholes = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT AVG(avg_confidence) FROM detection_history WHERE pothole_count > 0")
    avg_conf = cursor.fetchone()[0] or 0.0
    
    cursor.execute("SELECT AVG(processing_time_ms) FROM detection_history")
    avg_time = cursor.fetchone()[0] or 0.0
    
    cursor.execute("SELECT * FROM detection_history ORDER BY id DESC LIMIT 5")
    recent_rows = cursor.fetchall()
    conn.close()
    
    recent_activity = []
    for r in recent_rows:
        recent_activity.append({
            "id": r["id"],
            "file_name": r["file_name"],
            "detection_type": r["detection_type"],
            "pothole_count": r["pothole_count"],
            "avg_confidence": r["avg_confidence"],
            "created_at": r["created_at"],
            "result_path": r["result_path"]
        })
        
    return {
        "total_images_analyzed": total_images,
        "total_videos_analyzed": total_videos,
        "total_webcam_sessions": total_webcam,
        "total_potholes_detected": total_potholes,
        "average_confidence": round(avg_conf, 4),
        "average_processing_time_ms": round(avg_time, 2),
        "recent_activity": recent_activity
    }

# Initialize database on module load
init_db()

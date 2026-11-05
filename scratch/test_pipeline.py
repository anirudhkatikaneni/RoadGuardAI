import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.detection import detector
from backend.database import add_detection_record, get_dashboard_statistics

print("Detector ready:", detector.is_ready())
test_imgs = glob.glob("datasets/pothole/images/test/*.jpg")
print(f"Test images available: {len(test_imgs)}")

for img in test_imgs[:3]:
    res = detector.detect_image(img, conf_threshold=0.25)
    fname = os.path.basename(img)
    print(f"Detection on {fname}: Potholes={res['pothole_count']}, AvgConf={res['avg_confidence']*100:.1f}%, Latency={res['processing_time_ms']}ms")
    add_detection_record(
        file_name=fname,
        detection_type="image",
        pothole_count=res["pothole_count"],
        avg_confidence=res["avg_confidence"],
        processing_time_ms=res["processing_time_ms"],
        result_path=res["output_filename"],
        thumbnail_path=res["output_filename"],
        details={"boxes": res["boxes"]}
    )

print("\nDashboard statistics after real test scans:")
stats = get_dashboard_statistics()
for k, v in stats.items():
    if k != "recent_activity":
        print(f"  {k}: {v}")
print(f"Recent activity entries: {len(stats['recent_activity'])}")

import os
import sys
import glob
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.video_processing import process_video_file

# Create a short test video from test images
test_imgs = sorted(glob.glob("datasets/pothole/images/test/*.jpg"))[:30]
print(f"Creating sample test video from {len(test_imgs)} test road images...")

video_out_path = "uploads/sample_road_test.mp4"
h, w = 640, 640
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_out_path, fourcc, 10.0, (w, h))

for img_p in test_imgs:
    img = cv2.imread(img_p)
    if img is not None:
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        out.write(img)
out.release()
print(f"Created sample video: {video_out_path}")

# Run video processing pipeline
print("Running video processing pipeline...")
res = process_video_file(video_out_path)
print("Video Processing Result:")
for k, v in res.items():
    print(f"  {k}: {v}")

"""
בדיקה עם סרטון - עם OUTPUT
הרץ YOLO על סרטון וישמור סרטון חדש עם בounding boxes
"""

import cv2
import time
from ultralytics import YOLO
import os
import torch

# Check for GPU
print("🔍 Checking for GPU...")
if torch.backends.mps.is_available():
    print("✅ GPU (Metal) is available - using GPU!")
    device = "mps"  # macOS Metal
elif torch.cuda.is_available():
    print("✅ GPU (CUDA) is available - using GPU!")
    device = "cuda"
else:
    print("⚠️ GPU not available - using CPU")
    device = "cpu"

# Load model
print("📦 Loading YOLO model...")
model = YOLO("best.pt")
model.to(device)  # Move model to GPU
print(f"✅ Model loaded on {device.upper()}\n")

# בחר סרטון
VIDEO_PATH = input("📹 Enter video path (or press Enter for ../vid121.mp4): ").strip()
if not VIDEO_PATH:
    VIDEO_PATH = "../vid121.mp4"

print(f"🎬 Opening video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Error: Could not open video")
    exit(1)

# Get video info
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"\n📊 Video Info:")
print(f"   Resolution: {width}x{height}")
print(f"   FPS: {fps}")
print(f"   Total frames: {total_frames}")
print(f"   Duration: {total_frames/fps:.1f}s\n")

# Setup video writer (output)
output_path = "output_with_detections.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # או 'avc1' אם לא עובד
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

if not out.isOpened():
    print("⚠️ Warning: Could not create video writer, will show frames instead")
    out = None

# Process video
print("🔄 Processing...\n")

frame_count = 0
total_inference_time = 0
start_time = time.time()
all_detections = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run inference
    inference_start = time.time()
    results = model.predict(
        frame,
        imgsz=960,      # ✅ Resolution גבוהה = דיוק טוב
        conf=0.3,       # ✅ Confidence נמוך = תופס יותר detections
        iou=0.45,
        verbose=False
    )
    inference_time = (time.time() - inference_start) * 1000  # ms

    total_inference_time += inference_time

    # Extract detections
    detections = []
    frame_copy = frame.copy()

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            det_info = {
                "class": model.names[cls_id],
                "confidence": round(confidence, 3),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "frame": frame_count
            }
            detections.append(det_info)
            all_detections.append(det_info)

            # Draw bounding box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{model.names[cls_id]} {confidence:.2f}"
            cv2.putText(frame_copy, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write frame to output video
    if out:
        out.write(frame_copy)

    # Print progress כל 30 frames
    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        print(f"Frame {frame_count}/{total_frames} | "
              f"Objects: {len(detections)} | "
              f"Latency: {inference_time:.1f}ms | "
              f"Elapsed: {elapsed:.1f}s")

cap.release()
if out:
    out.release()

cv2.destroyAllWindows()

# Summary
elapsed_total = time.time() - start_time
avg_latency = total_inference_time / frame_count if frame_count > 0 else 0

print(f"\n✅ Done!\n")
print(f"📊 Summary:")
print(f"   Frames processed: {frame_count}")
print(f"   Total time: {elapsed_total:.1f}s")
print(f"   Average inference time: {avg_latency:.1f}ms")
print(f"   Processing speed: {frame_count/elapsed_total:.1f} fps")
print(f"   Total detections: {len(all_detections)}\n")

if out:
    print(f"✅ Output video saved: {output_path}")
    print(f"   📁 Location: {os.path.abspath(output_path)}")
    print(f"\n🎬 You can now open this file in any video player!")
else:
    print("⚠️ Could not save video")

# Show top detections
if all_detections:
    print(f"\n🎯 Top detected objects:")
    from collections import Counter
    classes = [d["class"] for d in all_detections]
    class_counts = Counter(classes)
    for cls, count in class_counts.most_common(5):
        print(f"   {cls}: {count} detections")

"""
בדיקה LIVE עם סרטון
הרץ YOLO על סרטון וראה את התוצאות בעצם הזמן בחלון
"""

import cv2
import time
from ultralytics import YOLO
import numpy as np
import torch

# Check for GPU
print("🔍 Checking for GPU...")
if torch.backends.mps.is_available():
    print("✅ GPU (Metal) is available - using GPU!")
    device = "mps"  # macOS Metal
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

print("🎬 Playing video (press 'q' to quit)...\n")

frame_count = 0
total_inference_time = 0
start_time = time.time()

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

    # Extract and draw detections
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
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            }
            detections.append(det_info)

            # Draw bounding box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label with confidence
            label = f"{model.names[cls_id]} {confidence:.2f}"
            cv2.putText(frame_copy, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw info on frame
    avg_latency = total_inference_time / frame_count

    # Frame counter
    cv2.putText(frame_copy, f"Frame: {frame_count}/{total_frames}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Latency
    cv2.putText(frame_copy, f"Latency: {inference_time:.1f}ms", (10, 65),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if inference_time < 100 else (0, 165, 255), 2)

    # Average latency
    cv2.putText(frame_copy, f"Avg: {avg_latency:.1f}ms", (10, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if avg_latency < 100 else (0, 165, 255), 2)

    # Object count
    cv2.putText(frame_copy, f"Objects: {len(detections)}", (10, 135),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Progress bar
    progress = frame_count / total_frames
    bar_width = int(width * 0.8)
    bar_height = 20
    bar_x = int((width - bar_width) / 2)
    bar_y = height - 40

    cv2.rectangle(frame_copy, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 1)
    filled_width = int(bar_width * progress)
    cv2.rectangle(frame_copy, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), (0, 255, 0), -1)

    # Progress percentage
    progress_text = f"{progress*100:.1f}%"
    cv2.putText(frame_copy, progress_text, (bar_x + bar_width + 10, bar_y + 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display frame
    cv2.imshow('YOLO Detection - Live', frame_copy)

    # Press 'q' to quit, 'p' to pause
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n⏹️  Stopped by user")
        break
    elif key == ord('p'):
        print("\n⏸️  Paused (press any key to continue)")
        cv2.waitKey(0)
        print("▶️  Resumed")

cap.release()
cv2.destroyAllWindows()

# Summary
elapsed_total = time.time() - start_time
avg_latency = total_inference_time / frame_count if frame_count > 0 else 0

print(f"\n✅ Finished!\n")
print(f"📊 Summary:")
print(f"   Frames processed: {frame_count}/{total_frames}")
print(f"   Total time: {elapsed_total:.1f}s")
print(f"   Average inference time: {avg_latency:.1f}ms")
print(f"   Processing speed: {frame_count/elapsed_total:.1f} fps")
print(f"\n🎯 Performance:")
if avg_latency < 50:
    print(f"   ✅✅ Excellent! (GPU or very fast CPU)")
elif avg_latency < 100:
    print(f"   ✅ Good (CPU or entry GPU)")
elif avg_latency < 150:
    print(f"   ⚠️ Acceptable (CPU berfore upgrade)")
else:
    print(f"   ⚠️ Slow (GPU recommended)")

print(f"\n💡 Shortcuts:")
print(f"   Press 'q' to quit")
print(f"   Press 'p' to pause/resume")

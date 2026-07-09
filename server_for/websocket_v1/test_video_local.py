"""
בדיקה לוקלית - הרץ YOLO על סרטון
זה לא צריך שרת, רק להרוץ locally על הסרטון שלך
"""

import cv2
import time
from ultralytics import YOLO
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
VIDEO_PATH = input("📹 Enter video path (or press Enter for vid121.mp4): ").strip()
if not VIDEO_PATH:
    VIDEO_PATH = "../vid121.mp4"  # Default

print(f"\n🎬 Opening video: {VIDEO_PATH}")
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
print(f"   Duration: {total_frames/fps:.1f}s")

# Process video
print("\n🔄 Processing (זה יקח זמן, בדוק את ה-latency!)...\n")

frame_count = 0
total_inference_time = 0
start_time = time.time()

# Optional: skip frames (כל כמה frames לעבד)
skip_frames = 1  # 1 = כל frame, 5 = כל פריים 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Skip frames אם רוצה (לטסט מהר)
    if frame_count % skip_frames != 0:
        continue

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
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class": model.names[cls_id],
                "confidence": round(confidence, 3),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

    # Print progress כל 30 frames
    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        print(f"Frame {frame_count}/{total_frames} | "
              f"Objects: {len(detections)} | "
              f"Latency: {inference_time:.1f}ms | "
              f"Avg Latency: {total_inference_time/(frame_count//skip_frames):.1f}ms | "
              f"Elapsed: {elapsed:.1f}s")

    # Show frame with bboxes (optional)
    # Uncomment כדי לראות video עם bounding boxes
    # if frame_count % 5 == 0:  # Show every 5th frame
    #     frame_copy = frame.copy()
    #     for det in detections:
    #         x1, y1, x2, y2 = det["bbox"]
    #         cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         cv2.putText(frame_copy, f"{det['class']}", (x1, y1-5),
    #                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    #     cv2.imshow('YOLO Detection', frame_copy)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

cap.release()
cv2.destroyAllWindows()

# Summary
elapsed_total = time.time() - start_time
avg_latency = total_inference_time / (frame_count // skip_frames) if frame_count > 0 else 0

print(f"\n✅ Done!\n")
print(f"📊 Summary:")
print(f"   Frames processed: {frame_count // skip_frames}")
print(f"   Total time: {elapsed_total:.1f}s")
print(f"   Average inference time: {avg_latency:.1f}ms")
print(f"   Processing speed: {(frame_count/skip_frames)/elapsed_total:.1f} fps")

# Predictions
print(f"\n🎯 Expected performance on cloud:")
if avg_latency < 50:
    print(f"   ✅ Excellent! (CPU or GPU)")
elif avg_latency < 150:
    print(f"   ⚠️ Acceptable for basic use, but GPU recommended")
else:
    print(f"   ⚠️ Slow - GPU is essential for real-time")

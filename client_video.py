import cv2
import requests
import time

SERVER_URL = "http://127.0.0.1:8000/detect"
VIDEO_PATH = "vid121.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

frame_id = 0
send_every = 5  # שולח כל פריים חמישי כדי לא להעמיס

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    if frame_id % send_every != 0:
        continue

    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        continue

    start = time.time()

    files = {
        "file": ("frame.jpg", encoded.tobytes(), "image/jpeg")
    }

    response = requests.post(SERVER_URL, files=files)
    data = response.json()

    latency = round((time.time() - start) * 1000, 1)
    detections = data.get("detections", [])

    print(f"Frame {frame_id} | detections={len(detections)} | roundtrip={latency}ms | server={data.get('latency_ms')}ms")

    for det in detections:
        print(det)

cap.release()
print("Done.")

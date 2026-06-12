import cv2
import requests
import time

SERVER_URL = "http://127.0.0.1:8000/detect"
VIDEO_PATH = "vid121.mp4"

CLASS_COLORS = {
    "hand": (0, 255, 0),
    "arm": (255, 0, 0),
    "head": (0, 0, 255),
    "leg": (255, 255, 0),
    "foot": (255, 0, 255),
    "person": (0, 255, 255),
}

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

frame_id = 0
send_every = 5

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    detections = []

    if frame_id % send_every == 0:

        ok, encoded = cv2.imencode(".jpg", frame)

        if ok:
            start = time.time()

            files = {
                "file": ("frame.jpg", encoded.tobytes(), "image/jpeg")
            }

            response = requests.post(SERVER_URL, files=files)

            data = response.json()

            detections = data.get("detections", [])

            latency = round((time.time() - start) * 1000, 1)

            print(
                f"Frame {frame_id} | "
                f"detections={len(detections)} | "
                f"roundtrip={latency}ms"
            )

    for det in detections:

        cls = det["class_name"]
        conf = det["confidence"]

        x1, y1, x2, y2 = det["bbox"]

        color = CLASS_COLORS.get(cls, (255, 255, 255))

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        label = f"{cls} {conf:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("Cloud YOLO Detection", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("Done.")

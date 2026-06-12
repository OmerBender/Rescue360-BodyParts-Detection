import cv2
import requests
import time

SERVER_URL = "http://127.0.0.1:8000/detect"
VIDEO_PATH = "vid121.mp4"
OUTPUT_PATH = "vid121_cloud_detected_yolo_style.mp4"

COLORS = {
    "hand": (0, 255, 0),
    "arm": (255, 120, 0),
    "head": (0, 0, 255),
    "leg": (255, 255, 0),
    "foot": (255, 0, 255),
    "person": (0, 255, 255),
}


def draw_yolo_box(frame, det):
    x1, y1, x2, y2 = map(int, det["bbox"])
    cls = det["class_name"]
    conf = float(det["confidence"])

    color = COLORS.get(cls, (0, 255, 0))
    label = f"{cls} {conf:.2f}"

    thickness = max(2, round(frame.shape[0] / 500))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, frame.shape[0] / 1400)
    text_thickness = max(1, thickness - 1)

    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
    y_text = max(y1, th + 8)

    cv2.rectangle(
        frame,
        (x1, y_text - th - baseline - 6),
        (x1 + tw + 8, y_text + baseline),
        color,
        -1,
    )

    cv2.putText(
        frame,
        label,
        (x1 + 4, y_text - 4),
        font,
        font_scale,
        (0, 0, 0),
        text_thickness,
        cv2.LINE_AA,
    )


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        detections = []

        if ok:
            files = {"file": ("frame.jpg", encoded.tobytes(), "image/jpeg")}
            start = time.time()

            try:
                response = requests.post(SERVER_URL, files=files, timeout=10)
                response.raise_for_status()
                data = response.json()
                detections = data.get("detections", [])
                latency = round((time.time() - start) * 1000, 1)
                print(
                    f"Frame {frame_id} | "
                    f"detections={len(detections)} | "
                    f"latency={latency}ms"
                )
            except Exception as e:
                print(f"Frame {frame_id} | server error: {e}")

        for det in detections:
            draw_yolo_box(frame, det)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Done. Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
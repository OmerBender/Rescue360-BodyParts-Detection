import cv2
import requests
import threading
import time

SERVER_URL = "http://127.0.0.1:8000/detect"
VIDEO_SOURCE = 0  # Insta360 X4

COLORS = {
    "hand": (0, 255, 0),
    "arm": (255, 120, 0),
    "head": (0, 0, 255),
    "leg": (255, 255, 0),
    "foot": (255, 0, 255),
    "person": (0, 255, 255),
}

latest_detections = []
latest_detections_time = 0.0
latest_latency_ms = 0.0
lock = threading.Lock()
busy = False


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

    y_text = max(y1, th + 10)

    cv2.rectangle(
        frame,
        (x1, y_text - th - baseline - 8),
        (x1 + tw + 10, y_text + baseline),
        color,
        -1,
    )

    cv2.putText(
        frame,
        label,
        (x1 + 5, y_text - 5),
        font,
        font_scale,
        (0, 0, 0),
        text_thickness,
        cv2.LINE_AA,
    )


def send_frame_async(frame):
    global latest_detections, latest_detections_time, latest_latency_ms, busy

    try:
        # קטן יותר = פחות דיליי, עדיין מספיק טוב ללייב
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if not ok:
            return

        files = {"file": ("frame.jpg", encoded.tobytes(), "image/jpeg")}

        start = time.time()
        response = requests.post(SERVER_URL, files=files, timeout=2)
        latency_ms = (time.time() - start) * 1000

        response.raise_for_status()
        data = response.json()

        with lock:
            latest_detections = data.get("detections", [])
            latest_detections_time = time.time()
            latest_latency_ms = latency_ms

    except Exception as e:
        print("server error:", e)

    finally:
        busy = False


def main():
    global busy

    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {VIDEO_SOURCE}")

    # ניסיון לקבע רזולוציה ו-FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps} FPS")

    frame_id = 0

    # שולח לשרת בערך כל פריים שני
    send_every = 2

    # לא מצייר בוקס אם הוא ישן מדי
    max_detection_age = 0.15

    # תצוגה יציבה
    delay = 1

    while True:
        ret, frame = cap.read()

        if not ret:
            print("No frame from camera")
            continue

        frame_id += 1

        if frame_id % send_every == 0 and not busy:
            busy = True
            threading.Thread(
                target=send_frame_async,
                args=(frame.copy(),),
                daemon=True,
            ).start()

        display = frame.copy()

        with lock:
            age = time.time() - latest_detections_time
            detections = list(latest_detections) if age < max_detection_age else []
            latency = latest_latency_ms

        for det in detections:
            draw_yolo_box(display, det)

        info = f"detections={len(detections)} latency={latency:.0f}ms age={age:.2f}s"
        cv2.putText(
            display,
            info,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("LIVE Insta360 -> Server YOLO", display)

        key = cv2.waitKey(delay) & 0xFF

        if key == 27:  # ESC
            break

        if key == ord("s"):
            cv2.imwrite("insta360_debug_frame.jpg", frame)
            print("Saved insta360_debug_frame.jpg")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
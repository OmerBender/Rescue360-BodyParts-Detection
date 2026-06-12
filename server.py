from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
import time

app = FastAPI()

model = YOLO("best.pt")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    start = time.time()

    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model.predict(
        frame,
        imgsz=960,
        conf=0.25,
        iou=0.45,
        verbose=False
    )

    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class_id": cls_id,
                "class_name": model.names[cls_id],
                "confidence": round(conf, 3),
                "bbox": [round(x1), round(y1), round(x2), round(y2)]
            })

    return {
        "detections": detections,
        "latency_ms": round((time.time() - start) * 1000, 2)
    }
"""
WebSocket Server for Real-time YOLO Detection
תומך במרובות מצלמות בו-זמנית
"""

import asyncio
import time
import cv2
import numpy as np
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import logging
import torch

# logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Camera YOLO Detection Server")

# Check for GPU
logger.info("🔍 Checking for GPU...")
if torch.backends.mps.is_available():
    logger.info("✅ GPU (Metal) is available - using GPU!")
    device = "mps"  # macOS Metal
elif torch.cuda.is_available():
    logger.info("✅ GPU (CUDA) is available - using GPU!")
    device = "cuda"
else:
    logger.info("⚠️ GPU not available - using CPU")
    device = "cpu"

# Load YOLO model
try:
    model = YOLO("best.pt")
    model.to(device)  # Move model to GPU
    logger.info(f"✅ YOLO model loaded successfully on {device.upper()}")
except Exception as e:
    logger.error(f"❌ Failed to load YOLO model: {e}")
    model = None

# אוספת חיבורים פעילים
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict = {}  # {client_id: websocket}
        self.stats = {}  # {client_id: stats}

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.stats[client_id] = {
            "frames_received": 0,
            "frames_processed": 0,
            "avg_latency_ms": 0,
            "connected_at": time.time()
        }
        logger.info(f"✅ Client '{client_id}' connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            del self.stats[client_id]
            logger.info(f"❌ Client '{client_id}' disconnected. Total clients: {len(self.active_connections)}")

    async def send_json(self, client_id: str, data: dict):
        try:
            if client_id in self.active_connections:
                await self.active_connections[client_id].send_json(data)
        except Exception as e:
            logger.error(f"Error sending to {client_id}: {e}")

    async def send_bytes(self, client_id: str, data: bytes):
        try:
            if client_id in self.active_connections:
                await self.active_connections[client_id].send_bytes(data)
        except Exception as e:
            logger.error(f"Error sending bytes to {client_id}: {e}")

    def get_stats(self):
        return {
            "total_clients": len(self.active_connections),
            "clients": self.stats
        }

manager = ConnectionManager()

@app.get("/")
async def get_root():
    """הצג דף HTML ראשי"""
    with open("client_websocket.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/stats")
async def get_stats():
    """קבל סטטיסטיקות על החיבורים"""
    return manager.get_stats()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint לקבלת וידאו בזמן אמת

    Protocol:
    - Client שולח: JPEG frames כ-binary data
    - Server מחזיר: JSON עם detections
    """

    await manager.connect(client_id, websocket)

    try:
        while True:
            # קבל frame מהלקוח
            frame_data = await websocket.receive_bytes()

            start_time = time.time()
            manager.stats[client_id]["frames_received"] += 1

            try:
                # המר bytes ל-numpy array
                np_arr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    logger.warning(f"Failed to decode frame from {client_id}")
                    continue

                # הרץ YOLO inference
                if model is not None:
                    results = model.predict(
                        frame,
                        imgsz=960,      # ✅ Resolution גבוהה = דיוק טוב
                        conf=0.3,       # ✅ Confidence נמוך = תופס יותר detections
                        iou=0.45,
                        verbose=False
                    )

                    # חלץ detections
                    detections = []
                    for r in results:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            x1, y1, x2, y2 = box.xyxy[0].tolist()

                            detections.append({
                                "class_id": cls_id,
                                "class_name": model.names[cls_id],
                                "confidence": round(confidence, 3),
                                "bbox": [int(x1), int(y1), int(x2), int(y2)]
                            })

                    # חשב latency
                    inference_time = (time.time() - start_time) * 1000
                    manager.stats[client_id]["frames_processed"] += 1
                    manager.stats[client_id]["avg_latency_ms"] = round(inference_time, 2)

                    # שלח תוצאות חזרה
                    response = {
                        "client_id": client_id,
                        "detections": detections,
                        "latency_ms": round(inference_time, 2),
                        "timestamp": time.time(),
                        "frame_count": manager.stats[client_id]["frames_received"]
                    }

                    await manager.send_json(client_id, response)

                    # לוג כל 30 frames
                    if manager.stats[client_id]["frames_received"] % 30 == 0:
                        logger.info(
                            f"📹 {client_id}: {len(detections)} objects, "
                            f"latency={inference_time:.1f}ms, "
                            f"frames={manager.stats[client_id]['frames_received']}"
                        )

            except Exception as e:
                logger.error(f"Error processing frame from {client_id}: {e}")
                continue

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)

if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting WebSocket Server...")
    logger.info("📍 Access at: http://0.0.0.0:8000")
    logger.info("📊 Stats at: http://0.0.0.0:8000/stats")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

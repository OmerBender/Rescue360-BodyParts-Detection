import json
import time
import asyncio

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from aiortc import RTCPeerConnection, RTCSessionDescription
from ultralytics import YOLO

app = FastAPI()

model = YOLO("../best.pt")
pcs = set()


class Offer(BaseModel):
    sdp: str
    type: str


@app.get("/")
async def index():
    with open("client_browser.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/video")
async def video():
    return FileResponse("../vid121.mp4", media_type="video/mp4")


@app.post("/offer")
async def offer(offer: Offer):
    pc = RTCPeerConnection()
    pcs.add(pc)

    channel_holder = {"channel": None}

    @pc.on("datachannel")
    def on_datachannel(channel):
        channel_holder["channel"] = channel
        print("DataChannel opened:", channel.label)

    @pc.on("track")
    def on_track(track):
        print("Track received:", track.kind)

        if track.kind == "video":

            async def consume_video():
                frame_id = 0

                while True:
                    try:
                        frame = await track.recv()
                    except Exception as e:
                        print("Track recv ended:", e)
                        break

                    frame_id += 1

                    if frame_id % 5 != 0:
                        continue

                    img = frame.to_ndarray(format="bgr24")
                    start = time.time()

                    results = model.predict(
                        img,
                        imgsz=960,
                        conf=0.45,
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

                    ch = channel_holder["channel"]

                    if ch and ch.readyState == "open":
                        h, w = img.shape[:2]

                        ch.send(json.dumps({
                            "frame_id": frame_id,
                            "frame_width": w,
                            "frame_height": h,
                            "latency_ms": round((time.time() - start) * 1000, 2),
                            "detections": detections
                        }))

            asyncio.create_task(consume_video())

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=offer.sdp, type=offer.type)
    )

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }


@app.on_event("shutdown")
async def on_shutdown():
    for pc in pcs:
        await pc.close()
    pcs.clear()

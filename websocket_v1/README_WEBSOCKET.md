# WebSocket Real-time YOLO Detection Server

שרת מרכזי לזיהוי objects בזמן אמת עם תמיכה במרובות מצלמות בו-זמנית.

## 📁 קבצים

- `server_websocket.py` - השרת הראשי (WebSocket)
- `client_websocket.html` - קליינט ברווזר (טלפון/דסקטופ)
- `requirements_websocket.txt` - dependencies

## 🚀 איך להתחיל

### 1. התקנת Dependencies

```bash
pip install -r requirements_websocket.txt --break-system-packages
```

### 2. הרץ את השרת

```bash
python3 server_websocket.py
```

תראה:
```
🚀 Starting WebSocket Server...
📍 Access at: http://0.0.0.0:8000
📊 Stats at: http://0.0.0.0:8000/stats
```

### 3. גישה מהטלפון/ברווזר

**מהמחשב בו השרת רץ:**
```
http://localhost:8000
```

**מטלפון על אותה רשת:**
```
http://<server_ip>:8000
```

(החלף `<server_ip>` ב-IP של המחשב שבו השרת רץ, למשל `192.168.1.100`)

## 🎥 איך להשתמש

1. **Start Camera** - פתח את המצלמה
2. **Connect Server** - התחבר לשרת
3. כל frame ישלח לשרת, YOLO יעבד, והתוצאות יחזרו
4. Bounding boxes יופיעו על המסך בעצם הזמן

## 📊 Monitor בזמן אמת

### Latency
```
Latency = inference_time (זה שקובע עיקרית ה-CPU/GPU)
       + network_time (קטן מאוד, ~10ms)
```

### Stats
גישה ל-stats בקישור:
```
http://localhost:8000/stats
```

תראה:
```json
{
  "total_clients": 2,
  "clients": {
    "camera1": {
      "frames_received": 150,
      "frames_processed": 148,
      "avg_latency_ms": 142.5
    },
    "camera2": {
      "frames_received": 89,
      "frames_processed": 87,
      "avg_latency_ms": 156.2
    }
  }
}
```

## ⚙️ הגדרות בקליינט

| הגדרה | משמעות | ברירת מחדל |
|------|--------|---------|
| Client ID | שם המצלמה (כדי להבדיל בין מרובות) | camera1 |
| Server URL | כתובת ה-WebSocket | ws://localhost:8000 |
| FPS Limit | כמה frames לשניה לשלוח | 10 FPS |

## 📈 ציפיות Latency

| Scenario | Expected | סטטוס |
|----------|----------|-------|
| CPU 4-cores | 100-200ms | ⚠️ Acceptable for some use cases |
| CPU 4-cores + GPU (T4) | 30-50ms | ✅ Good for real-time |
| CPU 8-cores + GPU (L4) | 15-30ms | ✅✅ Excellent |

## 🔧 Troubleshooting

### "Connection refused"
- ודא שהשרת רץ: `python3 server_websocket.py`
- ודא שה-IP נכון (זה צריך להיות IP של המחשב עליו השרת רץ)

### "Model loading failed"
- ודא ש-`best.pt` נמצא בתיקייה
- בדוק שהמודל לא משובש: `ls -lh best.pt`

### "High latency (>300ms)"
- זה normal בלא GPU
- GPU יורידה ל-30-50ms
- צמצום resolution או FPS יכול לעזור

### "Frames dropped"
- אם יש הרבה frames אבל `frames_processed` קטן מ-`frames_received`
- זה אומר שה-server לא מספיק מהיר
- ניסיון:
  - הורידו FPS
  - צמצומו ה-resolution
  - הוסיפו GPU

## 📱 Example - 3 מצלמות בו-זמנית

**Client 1 (טלפון A):**
- Client ID: `camera1`
- URL: `ws://192.168.1.10:8000`

**Client 2 (טלפון B):**
- Client ID: `camera2`
- URL: `ws://192.168.1.10:8000`

**Client 3 (טלפון C):**
- Client ID: `camera3`
- URL: `ws://192.168.1.10:8000`

כל אחד שולח frames בעצמאות, השרת עיבוד הכל בParallel.

## 🔍 Logs

כדי לראות לוגים מפורטים, שנה את `log_level` ב-`server_websocket.py`:

```python
uvicorn.run(
    app,
    host="0.0.0.0",
    port=8000,
    log_level="debug"  # ← change to "debug" for more details
)
```

## 📝 הערות

1. **WebSocket vs HTTP:**
   - HTTP: סוגר חיבור אחרי כל request (overhead)
   - WebSocket: חיבור קבוע (יותר מהר)

2. **Multi-client:**
   - כל לקוח מחובר בעצמאות
   - השרת טופל בהם בParallel (asyncio)
   - אין "תור" - כולם עובדים ביחד

3. **Performance:**
   - Bottleneck הראשי: YOLO inference time
   - Network קטן מאוד בהשוואה
   - GPU יעזור יותר מכל דבר אחר

## 🎯 הצעדים הבאים

1. ✅ שדרוג חומרה (CPU 4+ cores, RAM 8GB+)
2. ✅ Test עם 2-3 מצלמות
3. ⏳ אם latency גדול מדי → הוסף GPU (T4)
4. ⏳ Optimization עם TensorRT/ONNX (שלב 3)

---
- FPS setting שבחרת

# 🚀 Setup Local Testing - בדיקה למחשבך

## 📁 מבנה הקבצים

```
server_for/
├── [קבצים ישנים]
│   ├── server.py
│   ├── client_video.py
│   └── ... (וכו')
│
└── websocket_v1/  ← התיקייה החדשה שלנו
    ├── server_websocket.py        ← Server עם WebSocket
    ├── client_websocket.html      ← Client בברווזר
    ├── test_video_local.py        ← בדיקה עם סרטון (פשוט!)
    ├── requirements_websocket.txt ← dependencies
    ├── best.pt                    ← המודל YOLO
    └── README_WEBSOCKET.md        ← הוראות
```

---

## 🎬 אפשרות 1: בדיקה מהירה עם סרטון (קל ביותר)

זה בדיוק מה שתרצה - **אין צורך ב-WebSocket, מחשב בלבד**.

### צעד 1: פתח Terminal

```bash
cd /Users/omerbender/Documents/360to_mp4/server_for/websocket_v1
```

### צעד 2: הרץ את הסקריפט

```bash
python3 test_video_local.py
```

### צעד 3: בחר סרטון

```
📹 Enter video path (or press Enter for vid121.mp4): 
```

פשוט לחץ **Enter** אם יש לך `vid121.mp4` בתיקייה הקודמת.

או הזן path למהלך סרטון כמו:
```
../vid121.mp4
../vid120_detected_server.mp4
./test_5s.mp4
```

### צעד 4: חכה וראה תוצאות

```
🎬 Opening video: ../vid121.mp4

📊 Video Info:
   Resolution: 1280x720
   FPS: 30
   Total frames: 900
   Duration: 30.0s

🔄 Processing...

Frame 30/900 | Objects: 5 | Latency: 145.2ms | Avg Latency: 148.5ms | Elapsed: 4.5s
Frame 60/900 | Objects: 3 | Latency: 142.8ms | Avg Latency: 146.2ms | Elapsed: 8.2s
...

✅ Done!

📊 Summary:
   Frames processed: 900
   Total time: 12.5s
   Average inference time: 142.5ms
   Processing speed: 72.0 fps

🎯 Expected performance on cloud:
   ⚠️ Acceptable for basic use, but GPU recommended
```

---

## 🌐 אפשרות 2: WebSocket Server + Client (בהמשך)

כשתהיה מוכן להריץ שרת + טלפון:

### צעד 1: התקן dependencies

```bash
pip install -r requirements_websocket.txt
```

### צעד 2: הרץ את השרת

```bash
python3 server_websocket.py
```

תראה:
```
🚀 Starting WebSocket Server...
📍 Access at: http://0.0.0.0:8000
📊 Stats at: http://0.0.0.0:8000/stats
```

### צעד 3: גישה מהברווזר

בדסקטופ בו השרת רץ:
```
http://localhost:8000
```

או טלפון על אותה רשת:
```
http://192.168.1.X:8000
(החלף X בIP של המחשב)
```

---

## ❓ שאלות נפוצות

### Q: "No such file or directory: best.pt"
**A:** ודא שהמודל `best.pt` נמצא בתיקייה `websocket_v1/`

```bash
ls -lh best.pt
```

אם לא קיים, העתק מהתיקייה הקודמת:
```bash
cp ../best.pt .
```

### Q: "ModuleNotFoundError: No module named 'ultralytics'"
**A:** התקן requirements:
```bash
pip install -r requirements_websocket.txt
```

### Q: הסרטון נתמס ולא טוען
**A:** ניסיון סרטון קטן יותר:
- `./test_5s.mp4` - 5 שניות בלבד
- או צור סרטון קטן: `ffmpeg -i vid121.mp4 -t 10 -c copy test_10s.mp4`

### Q: Latency גבוה מדי (>200ms)
**A:** זה נורמלי בלא GPU! 
- `test_video_local.py` רץ על CPU בלבד
- ענן עם GPU יהיה הרבה יותר מהיר (30-50ms)

### Q: איך אוכל לראות את ה-bounding boxes?
**A:** בקובץ `test_video_local.py`, פתח את הקומנטים:
```python
# Show frame with bboxes (optional)
# Uncomment כדי לראות video עם bounding boxes
if frame_count % 5 == 0:
    ...
```

---

## 📊 מה לצפות

| Metric | Value |
|--------|-------|
| Latency (CPU בלבד) | 100-200ms ⚠️ |
| Latency (+ GPU T4) | 30-50ms ✅ |
| Frames/sec | תלוי בCPU |
| Memory used | ~2GB |

---

## ✅ Checklist

- [ ] `cd` לתיקייה `websocket_v1`
- [ ] `python3 test_video_local.py`
- [ ] בחר סרטון
- [ ] חכה לתוצאות
- [ ] בדוק את ה-Latency
- [ ] שתף לי את התוצאות!

---

**כמה זמן זה לוקח?**
- סרטון בן 30 שניות → כ-6-12 דקות של עיבוד (תלוי במחשבך)

**כל הפרמטרים אפשר לשנות?**
כן! ב-`test_video_local.py`:
- `skip_frames` - כל כמה frames לעבד (1 = הכל, 5 = כל 5)
- `imgsz=960` - גודל ה-inference (קטן יותר = מהר יותר, פחות דיוק)
- `conf=0.5` - confidence threshold (גבוה יותר = פחות false positives)

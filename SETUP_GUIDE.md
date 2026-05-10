# CoinSight Setup & Testing Guide

## 🚀 Installation Steps

### Step 1: Install Dependencies
```bash
cd "c:\Users\pepit\OneDrive\Documents\Mayong\3rd year\2nd Sem\Big Data\C-Ten"
pip install -r requirements.txt
```

This will install:
- **Flask 3.0+** — Web framework
- **Flask-SocketIO 5.0+** — Real-time WebSocket support
- **ultralytics** — YOLOv8 model framework
- **opencv-python** — Image/video processing
- **python-socketio & python-engineio** — WebSocket client/server

### Step 2: Verify Model File
Ensure you have one of these in the project root:
- `best.pt` (YOLOv8 weights) - Priority 1
- `coin_model.pkl` (pickled model) - Priority 2

If neither exists, the app will run in **demo mode**.

### Step 3: Start the Application
```bash
python app.py
```

You should see:
```
[INFO] Loading model from: model/best.pt
[OK] Model object type: ...
[OK] YOLO model loaded
[STATUS] MODEL_MODE = live
 * Running on http://0.0.0.0:5000
```

Open your browser: **http://localhost:5000**

---

## 🧪 Testing Guide

### Test 1: Image Detection
1. Open the app at `http://localhost:5000`
2. Click the **"📷 Image"** tab (should be selected by default)
3. Drag an image with coins onto the upload area (or click to browse)
4. Wait 1-2 seconds for processing
5. ✅ Expected: Annotated image appears with bounding boxes and detection details

### Test 2: Video Processing
1. Click the **"🎬 Video"** tab
2. Click **"Browse Video"**
3. Select a video file (MP4, AVI, MOV, etc.)
4. Wait for processing (progress shown in loading overlay)
5. ✅ Expected: First frame preview, frame count, total detections, and thumbnail grid

**Video Format Support:**
- MP4 ✅
- AVI ✅
- MOV ✅
- MKV ✅
- WEBM ✅

### Test 3: Camera Detection (Desktop)
1. Click the **"📹 Camera"** tab
2. Click **"Start Camera"**
3. Allow camera access when prompted
4. Point at coins and see real-time detections
5. ✅ Expected: Live detection with FPS and inference time displayed

### Test 4: Camera Detection (Mobile)
1. Open app on mobile device: `http://<your-pc-ip>:5000`
2. Click **"📹 Camera"** tab
3. Click **"Start Camera"**
4. Allow camera access
5. Point phone camera at coins
6. ✅ Expected: Real-time detection on mobile camera feed

**Note:** Replace `<your-pc-ip>` with your computer's IP address (check with `ipconfig` on Windows).

### Test 5: Mode Switching
1. Upload an image (📷 tab)
2. Switch to 🎬 tab - previous results should clear
3. Switch to 📹 tab - camera should be ready
4. Switch back to 📷 tab - upload area should be fresh
5. ✅ Expected: Smooth transitions between modes, no lingering data

---

## 🐛 Troubleshooting

### Issue: Model not loading
**Check:**
```bash
# Verify model file exists
ls model/best.pt
# or
ls coin_model.pkl
```

**Fix:** Place `best.pt` in the `model/` directory or project root.

---

### Issue: WebSocket connection fails
**Error:** `WebSocket connection failed` in browser console

**Fix:**
1. Verify Socket.IO is installed: `pip show python-socketio`
2. Reinstall if needed: `pip install --upgrade python-socketio python-engineio`
3. Restart the app: `python app.py`

---

### Issue: Camera access denied
**Error:** `NotAllowedError: Permission denied`

**Fix:**
1. **Desktop:** Check browser permissions in settings
2. **Mobile:** Use HTTPS (not HTTP) - or use `http://localhost:5000` directly
3. **Chrome:** Go to Settings → Privacy & Security → Site Settings → Camera

---

### Issue: Video processing is slow
**Cause:** Large video files or high resolution

**Fix:** Edit `app.py` line 190:
```python
while frame_count < min(total_frames, 300):  # Reduce 300 to lower number
```

Or reduce video resolution before uploading.

---

### Issue: "No coins detected"
**Cause:** Poor image quality, lighting, or angle

**Fix:**
- Ensure coins are clearly visible
- Good lighting (avoid shadows)
- Take photo from 30-45cm distance
- Try the demo mode to test UI

---

## 📊 Performance Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Single image detection | 40-100ms | Depends on image size |
| Video frame (720p) | 50-80ms | Per frame average |
| Camera frame (real-time) | 70-120ms | Includes WebSocket overhead |
| Demo mode detection | 5-10ms | No model inference |

---

## 🔧 Configuration Tips

### Increase Upload Limit
Edit `app.py` line 19:
```python
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB
```

### Adjust Camera FPS
Edit `templates/index.html` in the camera script (around line 290):
```javascript
}, 100);  // Change 100ms to 200ms for 5 FPS, or 50ms for 20 FPS
```

### Increase Confidence Threshold
Edit `app.py` line ~155 in `process_frame()`:
```python
results = model.predict(tmp_path, verbose=False, conf=0.7)[0]  # 0.7 = 70% confidence
```

---

## 📱 Mobile Setup

### Access from Mobile Device

1. Find your computer's IP:
   ```bash
   ipconfig  # Look for "IPv4 Address"
   ```

2. On mobile browser, navigate to:
   ```
   http://<your-ip>:5000
   ```

3. Test camera feature - it will access the phone's camera!

### Limitations
- ⚠️ HTTP only (not HTTPS) for local networks
- Camera access requires browser permission
- Best on 5GHz WiFi for lower latency

---

## ✅ Success Checklist

- [ ] Dependencies installed
- [ ] Model file (best.pt or coin_model.pkl) exists
- [ ] App starts without errors
- [ ] Image detection works
- [ ] Video upload works
- [ ] Camera access works (desktop)
- [ ] Camera access works (mobile, if tested)
- [ ] Mode switching works smoothly
- [ ] Detection results display correctly
- [ ] Real-time statistics update

---

## 📞 Support

**If you encounter issues:**

1. Check the terminal console for error messages
2. Open browser DevTools (F12) and check Console tab for JavaScript errors
3. Verify all files exist in the correct directories
4. Try running in demo mode first (remove model files temporarily)

---

**Happy detecting! 🪙✨**

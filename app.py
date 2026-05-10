import os
import io
import time
import base64
import pickle
import traceback
import tempfile
import threading
import queue

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Increased for video uploads
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_IMAGES = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
ALLOWED_VIDEOS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm'}

# Camera streaming state
camera_active = {}  # Dictionary to track active camera streams per session
frame_queue = queue.Queue(maxsize=2)  # Frame buffer for camera stream

# ── Load model ────────────────────────────────────────────────────────────────
model      = None
MODEL_MODE = 'demo'

print(f"\n[DEBUG] BASE_DIR = {BASE_DIR}")
print(f"[DEBUG] Checking for model files:")

model_paths = [
    os.path.join(BASE_DIR, 'model', 'coin_model.pkl'),
    os.path.join(BASE_DIR, 'model', 'best.pt'),
    os.path.join(BASE_DIR, 'coin_model.pkl'),
    os.path.join(BASE_DIR, 'best.pt'),
]

for path in model_paths:
    exists = os.path.exists(path)
    print(f"  {'✓' if exists else '✗'} {path}")

for model_path in model_paths:
    if os.path.exists(model_path):
        try:
            print(f"\n[INFO] Loading model from: {model_path}")
            if model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    bundle = pickle.load(f)
                    # ✓ Extract model from bundle (handles both dict and raw model)
                    model = bundle.get("model") if isinstance(bundle, dict) else bundle
                print(f"[OK] Model object type: {type(model)}")
            else:
                from ultralytics import YOLO
                model = YOLO(model_path)
                print(f"[OK] YOLO model loaded")
            MODEL_MODE = 'live'
            print(f"[OK] Model loaded successfully!")
            break
        except Exception as e:
            print(f"[WARN] Could not load {model_path}: {e}")
            traceback.print_exc()
            model = None

print(f"\n[STATUS] MODEL_MODE = {MODEL_MODE}")
if MODEL_MODE == 'demo':
    print("[WARNING] Running in demo mode — all outputs will be identical")


def allowed(filename, file_type='image'):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if file_type == 'image':
        return ext in ALLOWED_IMAGES
    elif file_type == 'video':
        return ext in ALLOWED_VIDEOS
    return False


def resize_image_to_width(image_path, target_width=640):
    """
    Resize image to target width while maintaining aspect ratio.
    Overwrites the original image file.
    
    Args:
        image_path: Path to the image file
        target_width: Target width in pixels (default: 640)
    """
    try:
        img = Image.open(image_path)
        # Calculate new height to maintain aspect ratio
        ratio = target_width / img.width
        new_height = int(img.height * ratio)
        
        # Resize image using high-quality resampling
        img_resized = img.resize((target_width, new_height), Image.Resampling.LANCZOS)
        
        # Save back to original path, maintaining format
        img_resized.save(image_path, quality=95)
        print(f"[INFO] Image resized to {target_width}px width (new size: {target_width}x{new_height})")
        return True
    except Exception as e:
        print(f"[ERROR] Image resize failed: {e}")
        traceback.print_exc()
        return False


def run_inference(img_path: str) -> dict:
    """Run YOLO inference or return demo detections."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return {'error': 'Could not read image', 'detections': [], 'count': 0, 'mode': 'demo'}
    
    h, w    = img_bgr.shape[:2]
    t0      = time.time()
    
    # Confidence threshold: 80%
    CONFIDENCE_THRESHOLD = 0.80

    if MODEL_MODE == 'live' and model is not None:
        try:
            results    = model.predict(img_path, verbose=False, conf=CONFIDENCE_THRESHOLD)[0]
            detections = []
            annotated  = img_bgr.copy()

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf  = float(box.conf[0]) * 100
                cls   = int(box.cls[0])
                # ✓ Use the model's actual trained class names
                label = results.names.get(cls, f"class_{cls}") if results.names else f"class_{cls}"

                # Draw box
                color = (0, 200, 255) if 'new' in label.lower() else (0, 140, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                tag = f"{label} {conf:.1f}%"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(annotated, tag, (x1 + 3, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

                detections.append({
                    'label':      label,
                    'confidence': round(conf, 1),
                    'bbox':       [x1, y1, x2, y2],
                })
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            traceback.print_exc()
            # Fallback to demo mode
            annotated  = img_bgr.copy()
            detections = []
    else:
        # Demo: fake two coins - only include if confidence >= 80%
        annotated  = img_bgr.copy()
        detections = []
        demo_boxes = [
            ('New 10 Peso', 88.4, [int(w*.1), int(h*.15), int(w*.45), int(h*.80)]),
            ('Old 10 Peso', 76.1, [int(w*.55), int(h*.15), int(w*.90), int(h*.80)]),
        ]
        for label, conf, (x1, y1, x2, y2) in demo_boxes:
            # Only include if confidence >= 80%
            if conf >= 80:
                color = (30, 200, 120) if 'new' in label.lower() else (30, 140, 220)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                tag = f"{label} {conf:.1f}%"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(annotated, tag, (x1 + 3, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
                detections.append({'label': label, 'confidence': conf, 'bbox': [x1, y1, x2, y2]})

    ms = round((time.time() - t0) * 1000, 1)

    # Encode annotated image to base64
    _, buf    = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    img_b64   = base64.b64encode(buf).decode()

    return {
        'image_b64':        img_b64,
        'detections':       detections,
        'inference_time_ms': ms,
        'count':            len(detections),
        'total_value':      len(detections) * 10,
        'mode':             MODEL_MODE,
    }


def process_frame(frame) -> dict:
    """Process a single video frame with YOLO inference."""
    if frame is None:
        return {'detections': [], 'count': 0, 'error': 'Invalid frame'}
    
    h, w = frame.shape[:2]
    t0 = time.time()
    annotated = frame.copy()
    detections = []

    if MODEL_MODE == 'live' and model is not None:
        try:
            # Save frame temporarily for inference
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                cv2.imwrite(tmp.name, frame)
                tmp_path = tmp.name
            
            results = model.predict(tmp_path, verbose=False, conf=0.5)[0]
            os.unlink(tmp_path)
            annotated = frame.copy()

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0]) * 100
                cls = int(box.cls[0])
                label = results.names.get(cls, f"class_{cls}") if results.names else f"class_{cls}"

                # Draw box
                color = (0, 200, 255) if 'new' in label.lower() else (0, 140, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                tag = f"{label} {conf:.1f}%"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(annotated, tag, (x1 + 3, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

                detections.append({
                    'label': label,
                    'confidence': round(conf, 1),
                    'bbox': [x1, y1, x2, y2],
                })
        except Exception as e:
            print(f"[ERROR] Frame processing failed: {e}")
    else:
        # Demo mode: add demo detections
        demo_boxes = [
            ('New 10 Peso', 88.4, [int(w*.1), int(h*.15), int(w*.45), int(h*.80)]),
            ('Old 10 Peso', 76.1, [int(w*.55), int(h*.15), int(w*.90), int(h*.80)]),
        ]
        for label, conf, (x1, y1, x2, y2) in demo_boxes:
            color = (30, 200, 120) if 'new' in label.lower() else (30, 140, 220)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            tag = f"{label} {conf:.1f}%"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, tag, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            detections.append({'label': label, 'confidence': conf, 'bbox': [x1, y1, x2, y2]})

    ms = round((time.time() - t0) * 1000, 1)

    # Encode frame to base64
    _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    frame_b64 = base64.b64encode(buf).decode()

    return {
        'frame_b64': frame_b64,
        'detections': detections,
        'count': len(detections),
        'inference_time_ms': ms,
    }


def process_video_file(video_path: str) -> dict:
    """Process entire video file and return frame-by-frame detections."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'error': 'Could not open video file', 'frames': []}

    frames_data = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    try:
        while frame_count < min(total_frames, 300):  # Limit to first 300 frames for performance
            ret, frame = cap.read()
            if not ret:
                break

            # Resize for faster processing if needed
            if frame.shape[1] > 1280:
                scale = 1280 / frame.shape[1]
                frame = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))

            result = process_frame(frame)
            result['frame_idx'] = frame_count
            frames_data.append(result)
            frame_count += 1

    finally:
        cap.release()

    return {
        'frames': frames_data,
        'total_frames': frame_count,
        'fps': fps,
        'error': None if frame_count > 0 else 'No frames processed',
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', mode=MODEL_MODE)


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'mode': MODEL_MODE, 'model_loaded': model is not None})


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image field in request'}), 400

    file = request.files['image']
    if not file.filename or not allowed(file.filename, 'image'):
        return jsonify({'error': 'Unsupported file type'}), 400

    # Get target width from request (default: 640px)
    target_width = request.form.get('target_width', 640, type=int)
    # Limit to reasonable bounds: 320px minimum, 1920px maximum
    target_width = max(320, min(1920, target_width))

    suffix = '.' + secure_filename(file.filename).rsplit('.', 1)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Resize image to target width dynamically based on upload area
        resize_image_to_width(tmp_path, target_width=target_width)
        
        # Run inference on resized image
        result = run_inference(tmp_path)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        os.unlink(tmp_path)


@app.route('/video-upload', methods=['POST'])
def video_upload():
    """Upload and process a video file."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video field in request'}), 400

    file = request.files['video']
    if not file.filename or not allowed(file.filename, 'video'):
        return jsonify({'error': 'Unsupported video format'}), 400

    suffix = '.' + secure_filename(file.filename).rsplit('.', 1)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = process_video_file(tmp_path)
        
        # Convert frames to base64 for first 10 frames (for preview/summary)
        summary_frames = []
        for i, frame_data in enumerate(result['frames'][:10]):
            summary_frames.append({
                'frame_idx': frame_data['frame_idx'],
                'frame_b64': frame_data['frame_b64'],
                'detections': frame_data['detections'],
                'count': frame_data['count'],
            })
        
        # Calculate aggregate statistics
        total_detections = sum(f['count'] for f in result['frames'])
        
        return jsonify({
            'summary_frames': summary_frames,
            'total_frames_processed': result['total_frames'],
            'total_detections': total_detections,
            'fps': result['fps'],
            'error': result['error'],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        os.unlink(tmp_path)


# ── WebSocket events for camera streaming ────────────────────────────────────

@socketio.on('camera:start')
def camera_start(data):
    """Start camera stream from client."""
    camera_id = request.sid
    camera_active[camera_id] = True
    print(f"[INFO] Camera started for session {camera_id}")
    emit('camera:started', {'status': 'Camera stream initiated'})


@socketio.on('camera:frame')
def camera_frame(data):
    """Receive frame from client camera and process it."""
    camera_id = request.sid
    
    if camera_id not in camera_active or not camera_active[camera_id]:
        emit('camera:error', {'error': 'Camera not active'})
        return

    try:
        # Decode frame from base64
        frame_data = data.get('frame_data', '')
        if not frame_data:
            return

        # Remove data URI prefix if present
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]

        # Decode and convert to numpy array
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            emit('camera:error', {'error': 'Could not decode frame'})
            return

        # Process frame with model
        result = process_frame(frame)

        # Send back processed frame and detections
        emit('camera:processed', {
            'frame_b64': result['frame_b64'],
            'detections': result['detections'],
            'count': result['count'],
            'inference_time_ms': result['inference_time_ms'],
        })

    except Exception as e:
        print(f"[ERROR] Camera frame processing failed: {e}")
        emit('camera:error', {'error': str(e)})


@socketio.on('camera:stop')
def camera_stop(data):
    """Stop camera stream."""
    camera_id = request.sid
    if camera_id in camera_active:
        camera_active[camera_id] = False
        del camera_active[camera_id]
    print(f"[INFO] Camera stopped for session {camera_id}")
    emit('camera:stopped', {'status': 'Camera stream ended'})


@socketio.on('connect')
def handle_connect():
    print(f"[INFO] Client connected: {request.sid}")


@socketio.on('disconnect')
def handle_disconnect():
    camera_id = request.sid
    if camera_id in camera_active:
        camera_active[camera_id] = False
        del camera_active[camera_id]
    print(f"[INFO] Client disconnected: {camera_id}")


if __name__ == '__main__':
    socketio.run(
        app,
        debug=True,
        host='0.0.0.0',
        port=5000,
        use_reloader=False,
    )
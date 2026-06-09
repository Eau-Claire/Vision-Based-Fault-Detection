import cv2
import torch
import numpy as np
from fastapi import FastAPI, Response, UploadFile, File
from fastapi.responses import StreamingResponse
import uvicorn
from yolo_detector import YOLODetector
from cnn_classifier import CNNClassifier
import os
import requests
import json
from collections import Counter

app = FastAPI(title="UAV Fault Detection Real-time API")

IP_CAMERA_URL = os.getenv("IP_CAMERA_URL", "0")

# Ocelot Gateway and Service configurations for reporting faults
OCELOT_GATEWAY_URL = os.getenv("OCELOT_GATEWAY_URL", "http://localhost:5000")
JWT_TOKEN = os.getenv("JWT_TOKEN", "")
TOWER_ID = os.getenv("TOWER_ID", "T-110KV-01")
LATITUDE = float(os.getenv("LATITUDE", "21.0285"))
LONGITUDE = float(os.getenv("LONGITUDE", "105.8542"))

# Check if best_model.pth exists, otherwise fallback to legacy insulator_cnn.pth
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PTH_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../models/best_model.pth"))
DEFAULT_LEGACY_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../models/insulator_cnn.pth"))
DEFAULT_MODEL_PATH = DEFAULT_PTH_PATH if os.path.exists(DEFAULT_PTH_PATH) else DEFAULT_LEGACY_PATH
CNN_MODEL_PATH = os.getenv("CNN_MODEL_PATH", DEFAULT_MODEL_PATH)

DEFAULT_YOLO_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '../models/best_detector.pt'))
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", DEFAULT_YOLO_PATH)
yolo = YOLODetector(YOLO_MODEL_PATH)
cnn = None

if os.path.exists(CNN_MODEL_PATH):
    cnn = CNNClassifier(CNN_MODEL_PATH)
else:
    print(f"--- Cảnh báo: Không tìm thấy CNN model tại {CNN_MODEL_PATH}. Chế độ CNN Refine sẽ bị tắt. ---")

# ──────────────────────────────────────────────────────────────────────
# Tuning constants
# ──────────────────────────────────────────────────────────────────────
PERSIST_FRAMES = 5          # Minimum frames before reporting
HOLD_FRAMES = 15            # Keep drawing a lost box for this many frames
CLEANUP_FRAMES = 40         # Remove track state after this many lost frames
MIN_REPORT_CONF = 0.40      # Minimum average confidence to report
HISTORY_LEN = 15            # Rolling history window length
EMA_ALPHA = 0.3             # Exponential moving average factor for box smoothing
                            # Lower = smoother but slower to react; higher = snappier


def send_fault_to_backend(frame, label, confidence):
    """
    Sends detected faults to the central web server API (Ocelot Gateway -> Inspection Service).
    First uploads the image, then registers the fault log.
    """
    headers = {}
    if JWT_TOKEN:
        headers["Authorization"] = f"Bearer {JWT_TOKEN}"
        
    print(f"\n⚠️  [SERVER FAULT REPORT] Type: {label} (Confidence: {confidence:.2%})")
    
    # 1. Encode frame to JPEG
    ret, jpeg_buffer = cv2.imencode(".jpg", frame)
    if not ret:
        print("Error: Could not encode frame to JPEG for upload.")
        return False
        
    # 2. Upload image
    upload_url = f"{OCELOT_GATEWAY_URL}/api/faults/upload-image"
    files = {"image": ("fault_capture.jpg", jpeg_buffer.tobytes(), "image/jpeg")}
    image_path = "/uploads/mock_drone_capture.jpg"  # Default fallback path
    
    try:
        print(f"Uploading fault image to {upload_url}...")
        upload_response = requests.post(upload_url, files=files, headers=headers, timeout=5)
        if upload_response.status_code == 200:
            image_path = upload_response.json().get("imagePath", image_path)
            print(f"Image uploaded successfully ✅ Path: {image_path}")
        else:
            print(f"Warning: Image upload failed (Code {upload_response.status_code}). Using fallback path.")
    except Exception as e:
        print(f"Warning: Image upload connection error: {e}. Using fallback path.")
        
    # 3. Post fault details
    report_url = f"{OCELOT_GATEWAY_URL}/api/faults"
    payload = {
        "towerId": TOWER_ID,
        "faultType": label,
        "confidenceScore": confidence,
        "imagePath": image_path,
        "latitude": LATITUDE,
        "longitude": LONGITUDE
    }
    
    json_headers = {"Content-Type": "application/json"}
    if JWT_TOKEN:
        json_headers["Authorization"] = f"Bearer {JWT_TOKEN}"
        
    try:
        print(f"Posting fault details to {report_url}...")
        report_response = requests.post(report_url, data=json.dumps(payload), headers=json_headers, timeout=5)
        if report_response.status_code in [200, 201]:
            print(f"Fault details posted successfully ✅ Response: {report_response.text}")
            return True
        else:
            print(f"Failed to post fault details: Status Code {report_response.status_code}")
            return False
    except Exception as e:
        print(f"Error reporting fault details: {e}")
        return False


def ema_smooth(old_bbox, new_bbox, alpha=EMA_ALPHA):
    """Exponential moving average to smooth bounding box coordinates."""
    if old_bbox is None:
        return new_bbox
    return tuple(
        int(alpha * n + (1 - alpha) * o)
        for o, n in zip(old_bbox, new_bbox)
    )


def generate_frames():
    cap = cv2.VideoCapture(IP_CAMERA_URL)
    
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở stream tại {IP_CAMERA_URL}")
        return

    # ── Per-track state ──
    track_seen = {}             # track_id -> total frames seen
    track_lost = {}             # track_id -> consecutive frames lost
    track_smooth_bbox = {}      # track_id -> EMA-smoothed (x1, y1, x2, y2)
    track_conf_hist = {}        # track_id -> list of recent confidences
    track_label_hist = {}       # track_id -> list of recent labels
    track_display_label = {}    # track_id -> smoothed majority label
    track_display_conf = {}     # track_id -> smoothed average confidence
    reported_tracks = set()     # track_ids already reported to backend

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ── YOLO detection + ByteTrack ──
        results = yolo.detect(frame, persist=True)
        crops = yolo.get_crops(frame, results)

        # Collect track IDs seen this frame
        seen_this_frame = set()

        for item in crops:
            tid = item['track_id']
            if tid is None:
                continue

            raw_bbox = item['bbox']       # (x1, y1, x2, y2) from YOLO
            yolo_label = item['label']
            yolo_conf = item['conf']

            seen_this_frame.add(tid)

            # Update counters
            track_seen[tid] = track_seen.get(tid, 0) + 1
            track_lost[tid] = 0

            # ── EMA smooth the bounding box ──
            track_smooth_bbox[tid] = ema_smooth(
                track_smooth_bbox.get(tid), raw_bbox
            )

            # CNN refinement (only for insulator crops)
            label = yolo_label
            conf = yolo_conf
            if cnn and yolo_label.lower() == "insulator":
                label, conf = cnn.predict(item['image'])

            # Append to rolling history
            track_conf_hist.setdefault(tid, []).append(conf)
            track_label_hist.setdefault(tid, []).append(label)

            # Trim history
            if len(track_conf_hist[tid]) > HISTORY_LEN:
                track_conf_hist[tid] = track_conf_hist[tid][-HISTORY_LEN:]
                track_label_hist[tid] = track_label_hist[tid][-HISTORY_LEN:]

            # Temporal smoothing
            avg_conf = sum(track_conf_hist[tid]) / len(track_conf_hist[tid])
            majority_label = Counter(track_label_hist[tid]).most_common(1)[0][0]

            track_display_label[tid] = majority_label
            track_display_conf[tid] = avg_conf

            # ── Report to backend (once per track) ──
            if tid not in reported_tracks and track_seen[tid] >= PERSIST_FRAMES:
                if avg_conf >= MIN_REPORT_CONF:
                    report_ok = send_fault_to_backend(frame, majority_label, avg_conf)
                    if report_ok:
                        reported_tracks.add(tid)

        # ── Update lost counters for tracks NOT seen this frame ──
        for tid in list(track_seen.keys()):
            if tid not in seen_this_frame:
                track_lost[tid] = track_lost.get(tid, 0) + 1

        # ── Draw boxes (using smoothed coordinates) ──
        for tid in list(track_seen.keys()):
            lost_count = track_lost.get(tid, 0)

            # Skip if lost too long
            if lost_count > HOLD_FRAMES:
                continue

            # Need cached info to draw
            if tid not in track_smooth_bbox or tid not in track_display_label:
                continue

            bx1, by1, bx2, by2 = track_smooth_bbox[tid]
            lbl = track_display_label[tid]
            cnf = track_display_conf[tid]

            # Color logic
            if tid not in seen_this_frame:
                # Interpolated (lost but within hold window)
                color = (0, 165, 255)   # Orange
                thick = 1
                tag = f" [HOLD {lost_count}/{HOLD_FRAMES}]"
            elif tid in reported_tracks:
                color = (0, 255, 0)     # Green – already reported
                thick = 2
                tag = " [SENT]"
            else:
                color = (255, 200, 0)   # Cyan – tracking, not yet reported
                thick = 2
                tag = f" [{track_seen[tid]}/{PERSIST_FRAMES}]"

            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, thick)
            text = f"#{tid} {lbl} ({cnf:.0%}){tag}"
            cv2.putText(frame, text, (bx1, max(by1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # ── Cleanup stale tracks ──
        for tid in list(track_seen.keys()):
            if track_lost.get(tid, 0) > CLEANUP_FRAMES:
                for d in (track_seen, track_lost, track_smooth_bbox,
                          track_conf_hist, track_label_hist,
                          track_display_label, track_display_conf):
                    d.pop(tid, None)
                reported_tracks.discard(tid)

        # ── Encode and yield MJPEG frame ──
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.get("/")
def index():
    return {"message": "UAV Inspection API is running. Go to /video to see the stream."}

@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(), 
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Read image bytes
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"error": "Could not decode uploaded image."}
        
    results = yolo.detect(frame, persist=False)
    crops = yolo.get_crops(frame, results)
    
    detections = []
    for item in crops:
        x1, y1, x2, y2 = item['bbox']
        yolo_label = item['label']
        
        refined_label = yolo_label
        confidence = float(item['conf'])
        
        if cnn and yolo_label.lower() == "insulator":
            refined_label, confidence = cnn.predict(item['image'])
                
        detections.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "yolo_label": yolo_label,
            "refined_label": refined_label,
            "confidence": float(confidence),
        })
        
    return {"detections": detections}

if __name__ == "__main__":
    # Chạy server tại port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

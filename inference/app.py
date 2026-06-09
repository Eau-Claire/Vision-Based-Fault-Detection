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
import time

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

def generate_frames():
    cap = cv2.VideoCapture(IP_CAMERA_URL)
    
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở stream tại {IP_CAMERA_URL}")
        return

    # User-requested tracking and persistence filter states
    track_seen_frames = {}      # track_id -> count of frames seen
    track_lost_frames = {}      # track_id -> count of consecutive frames lost
    track_confidences = {}      # track_id -> history list of CNN classification confidences
    track_labels = {}           # track_id -> history list of CNN predicted labels
    track_bboxes = {}           # track_id -> last known bbox (x1, y1, x2, y2)
    track_labels_current = {}   # track_id -> current majority label
    track_confs_current = {}    # track_id -> current average confidence
    reported_tracks = set()     # set of track_ids that have been reported to central server

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLOv8 detection & tracking with ByteTrack (persist=True)
        results = yolo.detect(frame, persist=True)
        crops = yolo.get_crops(frame, results)
        
        # Track IDs seen in the current frame
        current_frame_track_ids = set()
        
        for item in crops:
            x1, y1, x2, y2 = item['bbox']
            yolo_label = item['label']
            yolo_conf = item['conf']
            track_id = item['track_id']
            
            # If track_id is None, it means the tracker is initializing or not assigned yet
            if track_id is None:
                continue
                
            current_frame_track_ids.add(track_id)
            
            # 1. Update seen frame count, reset lost counter, and update bbox
            track_seen_frames[track_id] = track_seen_frames.get(track_id, 0) + 1
            track_lost_frames[track_id] = 0
            track_bboxes[track_id] = (x1, y1, x2, y2)
            
            # 2. Run CNN classification (only for insulator crops)
            refined_label = yolo_label
            confidence = yolo_conf
            if cnn and yolo_label.lower() == "insulator":
                refined_label, confidence = cnn.predict(item['image'])
                
            # 3. Store confidence and label history
            if track_id not in track_confidences:
                track_confidences[track_id] = []
                track_labels[track_id] = []
            track_confidences[track_id].append(confidence)
            track_labels[track_id].append(refined_label)
            
            # Keep history short (last 30 frames) to limit memory growth
            if len(track_confidences[track_id]) > 30:
                track_confidences[track_id].pop(0)
                track_labels[track_id].pop(0)
                
            # 4. Calculate average confidence and majority label (temporal smoothing)
            avg_conf = sum(track_confidences[track_id]) / len(track_confidences[track_id])
            
            from collections import Counter
            votes = Counter(track_labels[track_id])
            majority_label = votes.most_common(1)[0][0]
            
            # Cache the smoothed results for drawing
            track_labels_current[track_id] = majority_label
            track_confs_current[track_id] = avg_conf
            
            # 5. Check reporting conditions (only when actively detected):
            # - Is a fault
            # - Has been seen for >= 10 frames (Persistence Filter)
            # - Has not been reported yet (Deduplication)
            # - Average confidence is >= 50% (Avg confidence check)
            is_fault = False
            if majority_label in ["damaged", "disconnected", "misroute"]:
                is_fault = True
            elif "Clean" not in majority_label and "normal" not in majority_label.lower():
                is_fault = True
                
            if is_fault:
                seen_frames = track_seen_frames[track_id]
                if track_id not in reported_tracks and seen_frames >= 10:
                    if avg_conf >= 0.50:
                        success = send_fault_to_backend(frame, majority_label, avg_conf)
                        if success:
                            reported_tracks.add(track_id)

        # 6. Render active and interpolated tracks
        for track_id in list(track_seen_frames.keys()):
            # If not seen in current frame, increment lost frames counter
            if track_id not in current_frame_track_ids:
                track_lost_frames[track_id] = track_lost_frames.get(track_id, 0) + 1
                
            # Interpolation: Keep drawing box if lost for <= 15 frames
            if track_id in current_frame_track_ids or track_lost_frames.get(track_id, 0) <= 15:
                # Retrieve cached drawing parameters
                if track_id in track_bboxes and track_id in track_labels_current:
                    x1, y1, x2, y2 = track_bboxes[track_id]
                    majority_label = track_labels_current[track_id]
                    avg_conf = track_confs_current[track_id]
                    seen_frames = track_seen_frames[track_id]
                    
                    is_fault = False
                    if majority_label in ["damaged", "disconnected", "misroute"]:
                        is_fault = True
                    elif "Clean" not in majority_label and "normal" not in majority_label.lower():
                        is_fault = True
                        
                    # Choose box colors based on detection status (yellow for interpolation, green/red for active)
                    if track_id not in current_frame_track_ids:
                        box_color = (0, 165, 255)  # Orange/Yellow for interpolation
                        thickness = 2
                        status_tag = f" [LOST - HOLD {track_lost_frames[track_id]}/15]"
                    else:
                        box_color = (0, 0, 255) if is_fault else (0, 255, 0)
                        thickness = 3 if is_fault else 2
                        status_tag = " [REPORTED]" if track_id in reported_tracks else f" [{seen_frames}/10]"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
                    
                    display_text = f"Track #{track_id} | {majority_label} ({avg_conf:.1%}){status_tag}"
                    cv2.putText(frame, display_text, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 7. Cleanup old tracks (lost > 30 frames) to free memory
        for track_id in list(track_seen_frames.keys()):
            if track_lost_frames.get(track_id, 0) > 30:
                track_seen_frames.pop(track_id, None)
                track_lost_frames.pop(track_id, None)
                track_confidences.pop(track_id, None)
                track_labels.pop(track_id, None)
                track_bboxes.pop(track_id, None)
                track_labels_current.pop(track_id, None)
                track_confs_current.pop(track_id, None)
                reported_tracks.discard(track_id)

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
        is_fault = False
        
        if cnn and yolo_label.lower() == "insulator":
            refined_label, confidence = cnn.predict(item['image'])
            
            # Check for faults
            if refined_label in ["damaged", "disconnected", "misroute"]:
                is_fault = True
            elif "Clean" not in refined_label and "normal" not in refined_label.lower():
                is_fault = True
                
        detections.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "yolo_label": yolo_label,
            "refined_label": refined_label,
            "confidence": float(confidence),
            "is_fault": is_fault
        })
        
    return {"detections": detections}

if __name__ == "__main__":
    # Chạy server tại port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

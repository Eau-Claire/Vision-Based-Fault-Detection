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

DEFAULT_YOLO_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, 'best.pt'))
yolo = YOLODetector(DEFAULT_YOLO_PATH)
cnn = None

if os.path.exists(CNN_MODEL_PATH):
    cnn = CNNClassifier(CNN_MODEL_PATH)
else:
    print(f"--- Cảnh báo: Không tìm thấy CNN model tại {CNN_MODEL_PATH}. Chế độ CNN Refine sẽ bị tắt. ---")

def get_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    unionArea = box1Area + box2Area - interArea
    if unionArea == 0:
        return 0.0
    return interArea / unionArea

def get_distance(box1, box2):
    c1_x = (box1[0] + box1[2]) / 2.0
    c1_y = (box1[1] + box1[3]) / 2.0
    c2_x = (box2[0] + box2[2]) / 2.0
    c2_y = (box2[1] + box2[3]) / 2.0
    return ((c1_x - c2_x)**2 + (c1_y - c2_y)**2) ** 0.5

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

    tracked_objects = []
    next_obj_id = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Get frame dimensions to make centroid distance threshold adaptive
        h_frame, w_frame, _ = frame.shape
        max_dim = max(h_frame, w_frame)
        # Adaptive centroid match distance (15% of max image dimension)
        dist_threshold = max(150.0, max_dim * 0.15)

        results = yolo.detect(frame)
        crops = yolo.get_crops(frame, results)
        
        # Keep track of matched detections and matched tracked objects
        matched_detections = set()
        matched_tracked = set()
        
        # 1. Match current detections to tracked objects using combined Match Score
        # Match Score combines Overlap (IoU) and Centroid proximity.
        for det_idx, item in enumerate(crops):
            bbox = item['bbox']
            best_score = -1.0
            best_track_idx = -1
            
            for t_idx, obj in enumerate(tracked_objects):
                if t_idx in matched_tracked:
                    continue
                
                iou = get_iou(bbox, obj['bbox'])
                dist = get_distance(bbox, obj['bbox'])
                
                # Combine IoU with distance score
                # Normalized distance score: 1.0 at 0 dist, 0.0 at dist_threshold
                dist_score = max(0.0, 1.0 - (dist / dist_threshold))
                score = iou + dist_score * 0.6
                
                if score > best_score:
                    best_score = score
                    best_track_idx = t_idx
            
            # Match is considered valid if score is high enough (e.g. > 0.40)
            if best_score > 0.40 and best_track_idx != -1:
                obj = tracked_objects[best_track_idx]
                obj['bbox'] = bbox
                obj['missing_frames'] = 0
                
                if cnn:
                    refined_label, confidence = cnn.predict(item['image'])
                    obj['label_history'].append(refined_label)
                    obj['conf_history'].append(confidence)
                    if len(obj['label_history']) > 5:
                        obj['label_history'].pop(0)
                        obj['conf_history'].pop(0)
                        
                    from collections import Counter
                    votes = Counter(obj['label_history'])
                    smooth_label = votes.most_common(1)[0][0]
                    winning_confs = [c for l, c in zip(obj['label_history'], obj['conf_history']) if l == smooth_label]
                    smooth_conf = sum(winning_confs) / len(winning_confs) if winning_confs else 0.0
                    
                    obj['smooth_label'] = smooth_label
                    obj['smooth_conf'] = smooth_conf
                else:
                    obj['smooth_label'] = item['label']
                    obj['smooth_conf'] = item['conf']
                    
                matched_tracked.add(best_track_idx)
                matched_detections.add(det_idx)
        
        # 2. For unmatched detections, create new tracks
        for det_idx, item in enumerate(crops):
            if det_idx in matched_detections:
                continue
            bbox = item['bbox']
            
            if cnn:
                refined_label, confidence = cnn.predict(item['image'])
                new_obj = {
                    'id': next_obj_id,
                    'bbox': bbox,
                    'label_history': [refined_label],
                    'conf_history': [confidence],
                    'missing_frames': 0,
                    'smooth_label': refined_label,
                    'smooth_conf': confidence,
                    'reported': False
                }
            else:
                new_obj = {
                    'id': next_obj_id,
                    'bbox': bbox,
                    'label_history': [item['label']],
                    'conf_history': [item['conf']],
                    'missing_frames': 0,
                    'smooth_label': item['label'],
                    'smooth_conf': item['conf'],
                    'reported': False
                }
            next_obj_id += 1
            tracked_objects.append(new_obj)
            
        # 3. For unmatched tracked objects, mark them as missing
        active_tracks = []
        for t_idx, obj in enumerate(tracked_objects):
            if t_idx not in matched_tracked:
                obj['missing_frames'] += 1
            
            # Interpolation hold buffer: keep drawing for up to 15 frames during camera shakes
            if obj['missing_frames'] <= 15:
                active_tracks.append(obj)
        tracked_objects = active_tracks
        
        # 4. Render active tracks and report faults
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj['bbox']
            smooth_label = obj['smooth_label']
            smooth_conf = obj['smooth_conf']
            
            # Draw green box for YOLO detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            display_text = f"Insulator #{obj['id']}"
            if cnn:
                display_text += f" | CNN: {smooth_label} ({smooth_conf:.1%})"
                
                # Check for faults
                is_fault = False
                if smooth_label in ["damaged", "disconnected", "misroute"]:
                    # Require minimum 50% confidence for fault alert
                    if smooth_conf >= 0.50:
                        is_fault = True
                elif "Clean" not in smooth_label and "normal" not in smooth_label.lower():
                    if smooth_conf >= 0.50:
                        is_fault = True
                        
                if is_fault:
                    # Draw thick red box for fault
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    
                    # REPORT TO WEB API ONLY ONCE per unique object ID (no duplicates)
                    if not obj.get('reported', False):
                        # Send report to backend Web API
                        success = send_fault_to_backend(frame, smooth_label, smooth_conf)
                        if success:
                            obj['reported'] = True
            else:
                display_text += f" | {smooth_label} ({smooth_conf:.1%})"
                
            # Draw label text
            cv2.putText(frame, display_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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
        
    results = yolo.detect(frame)
    crops = yolo.get_crops(frame, results)
    
    detections = []
    for item in crops:
        x1, y1, x2, y2 = item['bbox']
        yolo_label = item['label']
        
        refined_label = "unknown"
        confidence = 0.0
        is_fault = False
        
        if cnn:
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

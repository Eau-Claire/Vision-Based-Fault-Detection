import cv2
import torch
import numpy as np
from fastapi import FastAPI, Response, UploadFile, File
from fastapi.responses import StreamingResponse
import uvicorn
from yolo_detector import YOLODetector
from cnn_classifier import CNNClassifier
import os

app = FastAPI(title="UAV Fault Detection Real-time API")

IP_CAMERA_URL = os.getenv("IP_CAMERA_URL", "0")

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

        results = yolo.detect(frame)
        crops = yolo.get_crops(frame, results)
        
        # Keep track of matched detections and matched tracked objects
        matched_detections = set()
        matched_tracked = set()
        
        # 1. Match current detections to tracked objects
        for det_idx, item in enumerate(crops):
            bbox = item['bbox']
            best_iou = 0.0
            best_track_idx = -1
            
            for t_idx, obj in enumerate(tracked_objects):
                if t_idx in matched_tracked:
                    continue
                iou = get_iou(bbox, obj['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_track_idx = t_idx
            
            if best_iou > 0.3:
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
                    'smooth_conf': confidence
                }
            else:
                new_obj = {
                    'id': next_obj_id,
                    'bbox': bbox,
                    'label_history': [item['label']],
                    'conf_history': [item['conf']],
                    'missing_frames': 0,
                    'smooth_label': item['label'],
                    'smooth_conf': item['conf']
                }
            next_obj_id += 1
            tracked_objects.append(new_obj)
            
        # 3. For unmatched tracked objects, mark them as missing
        active_tracks = []
        for t_idx, obj in enumerate(tracked_objects):
            if t_idx not in matched_tracked:
                obj['missing_frames'] += 1
            
            # Interpolation: keep drawing for up to 4 frames if temporarily missed
            if obj['missing_frames'] <= 4:
                active_tracks.append(obj)
        tracked_objects = active_tracks
        
        # 4. Render active tracks
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

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
DEFAULT_MODEL_PATH = "../models/best_model.pth" if os.path.exists("../models/best_model.pth") else "../models/insulator_cnn.pth"
CNN_MODEL_PATH = os.getenv("CNN_MODEL_PATH", DEFAULT_MODEL_PATH)

yolo = YOLODetector('best.pt')
cnn = None

if os.path.exists(CNN_MODEL_PATH):
    cnn = CNNClassifier(CNN_MODEL_PATH)
else:
    print(f"--- Cảnh báo: Không tìm thấy CNN model tại {CNN_MODEL_PATH}. Chế độ CNN Refine sẽ bị tắt. ---")

def generate_frames():
    cap = cv2.VideoCapture(IP_CAMERA_URL)
    
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở stream tại {IP_CAMERA_URL}")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = yolo.detect(frame)
        
        crops = yolo.get_crops(frame, results)
        
        for item in crops:
            x1, y1, x2, y2 = item['bbox']
            yolo_label = item['label']
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            display_text = f"YOLO: {yolo_label}"

            if cnn:
                refined_label, confidence = cnn.predict(item['image'])
                display_text += f" | CNN: {refined_label} ({confidence:.1%})"
                
                # Check for faults in both legacy model ('Clean-Insulator' is normal, others are faults) 
                # and new model ('normal' is normal, 'damaged', 'disconnected', 'misroute' are faults)
                is_fault = False
                if refined_label in ["damaged", "disconnected", "misroute"]:
                    is_fault = True
                elif "Clean" not in refined_label and "normal" not in refined_label.lower():
                    is_fault = True
                    
                if is_fault:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

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

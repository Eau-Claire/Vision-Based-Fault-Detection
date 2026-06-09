import os
import cv2
import time
import requests
import json
from PIL import Image
from cnn_classifier import CNNClassifier
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =====================================================================
# CONFIGURATION
# =====================================================================
# Gateway (Raspberry Pi 4) configurations
IP_CAMERA_URL = os.getenv("IP_CAMERA_URL", "0")  # Camera stream source (0 = webcam)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ONNX_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../models/best_model.onnx"))
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", DEFAULT_ONNX_PATH)

# Ocelot Gateway and Service configurations
OCELOT_GATEWAY_URL = os.getenv("OCELOT_GATEWAY_URL", "http://localhost:5000")
JWT_TOKEN = os.getenv("JWT_TOKEN", "")  # JWT Token for authorization header

# Metadata configurations
TOWER_ID = os.getenv("TOWER_ID", "T-110KV-01")
LATITUDE = float(os.getenv("LATITUDE", "21.0285"))
LONGITUDE = float(os.getenv("LONGITUDE", "105.8542"))
REPORT_COOLDOWN = 10.0  # Seconds to wait before reporting the same fault type again

# =====================================================================
# RUN TIME GLOBAL VARIABLES
# =====================================================================
last_reported_time = {}  # Cooldown map: {fault_type: last_sent_timestamp}

def send_fault_to_backend(frame, label, confidence):
    """
    Follow the workflow from Services:
    1. Upload frame to upload-image endpoint -> returns relative imagePath
    2. Post fault information with the imagePath to backend database
    """
    global last_reported_time
    current_time = time.time()
    
    # Check cooldown to prevent flooding database with duplicate detections
    if label in last_reported_time:
        elapsed = current_time - last_reported_time[label]
        if elapsed < REPORT_COOLDOWN:
            # We already reported this fault recently, skip it
            return False
            
    # Setup headers (Include JWT Authorization if available)
    headers = {}
    if JWT_TOKEN:
        headers["Authorization"] = f"Bearer {JWT_TOKEN}"
        
    print(f"\n⚠️  [FAULT DETECTED] Type: {label} (Confidence: {confidence:.2%})")
    
    # 1. Encode frame to JPEG format in memory
    ret, jpeg_buffer = cv2.imencode(".jpg", frame)
    if not ret:
        print("Error: Could not encode frame to JPEG.")
        return False
        
    # 2. Upload image to Web API
    upload_url = f"{OCELOT_GATEWAY_URL}/api/faults/upload-image"
    files = {"image": ("fault_capture.jpg", jpeg_buffer.tobytes(), "image/jpeg")}
    
    image_path = "/uploads/mock_drone_capture.jpg"  # Default fallback path
    
    try:
        print(f"Uploading fault image to {upload_url}...")
        upload_response = requests.post(upload_url, files=files, headers=headers, timeout=5)
        if upload_response.status_code == 200:
            image_path = upload_response.json().get("imagePath", image_path)
            print(f"Image uploaded successfully ✅ Relative path: {image_path}")
        else:
            print(f"Warning: Image upload failed with code {upload_response.status_code}. Using fallback path.")
    except Exception as e:
        print(f"Warning: Image upload error: {e}. Using fallback path.")
        
    # 3. Post fault information payload
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
        print(f"Reporting fault details to {report_url}...")
        report_response = requests.post(report_url, data=json.dumps(payload), headers=json_headers, timeout=5)
        if report_response.status_code in [200, 201]:
            print(f"Fault details posted successfully ✅ Response: {report_response.text}")
            # Update cooldown timestamp
            last_reported_time[label] = current_time
            return True
        else:
            print(f"Failed to post fault details: Status Code {report_response.status_code}")
            return False
    except Exception as e:
        print(f"Error reporting fault details: {e}")
        return False

def run_gateway():
    print("=====================================================================")
    print("🛰️  Edge Gateway Simulation (Raspberry Pi 4 Flow)")
    print("=====================================================================")
    print(f"Model path:     {ONNX_MODEL_PATH}")
    print(f"Video Source:   {IP_CAMERA_URL}")
    print(f"Ocelot Gateway: {OCELOT_GATEWAY_URL}")
    print(f"Tower ID:       {TOWER_ID}")
    print("=====================================================================")
    
    # Initialize ONNX classifier
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"Error: ONNX model file not found at '{ONNX_MODEL_PATH}'")
        print("Please export the model to ONNX using models/export_onnx.py first.")
        return
        
    classifier = CNNClassifier(ONNX_MODEL_PATH)
    
    # Determine video capture source (integer for webcam, string for stream URL)
    try:
        source = int(IP_CAMERA_URL)
    except ValueError:
        source = IP_CAMERA_URL
        
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open camera/video source: {IP_CAMERA_URL}")
        return
        
    print("Starting frame capture stream... (Press 'q' to quit)")
    
    # Performance indicators
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0.0
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Finished stream or failed to read frame. Exiting...")
            break
            
        fps_counter += 1
        # Calculate FPS every 1 second
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
            
        # Draw camera metadata on frame
        display_frame = frame.copy()
        h, w, _ = display_frame.shape
        
        # 1. Preprocess: Convert OpenCV frame to PIL Image (RGB)
        # Note: CNNClassifier handles resize 224x224 and normalization internally
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # 2. ONNX Inference
        label, confidence = classifier.predict(pil_img)
        
        # 3. fault? check (fault labels are: damaged, disconnected, misroute)
        is_fault = label in ["damaged", "disconnected", "misroute"]
        
        # Draw information overlay
        status_text = f"Class: {label} ({confidence:.1%}) | FPS: {fps:.1f}"
        box_color = (0, 0, 255) if is_fault else (0, 255, 0)
        
        # Draw status bar
        cv2.rectangle(display_frame, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(display_frame, status_text, (10, 26), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Bounding box around entire frame if fault
        if is_fault:
            cv2.rectangle(display_frame, (10, 10), (w-10, h-10), box_color, 4)
            cv2.putText(display_frame, "⚠️ FAULT WARNING", (30, h-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, box_color, 3)
            
            # 4. HTTP POST details to Services (Ocelot Gateway -> Inspection Service)
            # Run in a clean non-blocking/serial way for the simulation
            send_fault_to_backend(frame, label, confidence)
            
        # Show visualization window
        cv2.imshow("Edge Gateway Monitor (Raspberry Pi 4)", display_frame)
        
        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Stream stopped. Exiting gateway simulation.")

if __name__ == "__main__":
    run_gateway()

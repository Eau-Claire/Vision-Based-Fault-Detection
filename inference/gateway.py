import os
import sys
import cv2
import time
import requests
import json
from collections import Counter
from PIL import Image

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from inference.config import cfg, print_config
from inference.core.cnn_classifier import CNNClassifier

# =====================================================================
# GLOBAL STATE
# =====================================================================
last_reported_time = {}  # Cooldown map for CNN-only mode: {fault_type: last_sent_timestamp}

def send_fault_to_backend(frame, label, confidence):
    """
    Sends detected faults to the central web server API (Ocelot Gateway -> Inspection Service).
    First uploads the image, then registers the fault log.
    """
    headers = {}
    if cfg.JWT_TOKEN:
        headers["Authorization"] = f"Bearer {cfg.JWT_TOKEN}"
        
    print(f"\n⚠️  [FAULT DETECTED] Type: {label} (Confidence: {confidence:.2%})")
    
    # 1. Encode frame to JPEG format in memory
    ret, jpeg_buffer = cv2.imencode(".jpg", frame)
    if not ret:
        print("Error: Could not encode frame to JPEG.")
        return False
        
    # 2. Upload image to Web API
    upload_url = f"{cfg.OCELOT_GATEWAY_URL}/api/faults/upload-image"
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
    report_url = f"{cfg.OCELOT_GATEWAY_URL}/api/faults"
    payload = {
        "towerId": cfg.TOWER_ID,
        "faultType": label,
        "confidenceScore": confidence,
        "imagePath": image_path,
        "latitude": cfg.LATITUDE,
        "longitude": cfg.LONGITUDE
    }
    
    json_headers = {"Content-Type": "application/json"}
    if cfg.JWT_TOKEN:
        json_headers["Authorization"] = f"Bearer {cfg.JWT_TOKEN}"
        
    try:
        print(f"Reporting fault details to {report_url}...")
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


def ema_smooth(old_bbox, new_bbox, alpha=cfg.EMA_ALPHA):
    """Exponential moving average to smooth bounding box coordinates."""
    if old_bbox is None:
        return new_bbox
    return tuple(
        int(alpha * n + (1 - alpha) * o)
        for o, n in zip(old_bbox, new_bbox)
    )


def calculate_iou(boxA, boxB):
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea == 0:
        return 0.0
    return interArea / unionArea


def run_gateway():
    print_config()
    print("🛰️  Edge Client Gateway Started")
    print("=" * 65)
    
    # ── Initialize models based on config ──
    yolo = None
    cnn = None

    if cfg.USE_YOLO:
        from inference.core.yolo_detector import YOLODetector
        if cfg.YOLO_MODEL_PATH and os.path.exists(cfg.YOLO_MODEL_PATH):
            yolo = YOLODetector(cfg.YOLO_MODEL_PATH)
        else:
            print(f"⚠️  YOLO model not found at '{cfg.YOLO_MODEL_PATH}'. YOLO disabled.")

    if cfg.USE_CNN_REFINE:
        if cfg.CNN_MODEL_PATH and os.path.exists(cfg.CNN_MODEL_PATH):
            cnn = CNNClassifier(cfg.CNN_MODEL_PATH, device=cfg.DEVICE)
        else:
            print(f"⚠️  CNN model not found at '{cfg.CNN_MODEL_PATH}'. CNN disabled.")
            
    # Determine video capture source
    try:
        source = int(cfg.IP_CAMERA_URL)
    except ValueError:
        source = cfg.IP_CAMERA_URL
        
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open camera/video source: {cfg.IP_CAMERA_URL}")
        return
        
    print("Starting frame capture stream... (Press 'q' in visualizer to quit)")
    
    # Performance indicators
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0.0
    
    # ── Per-track state (only active in YOLO mode) ──
    track_seen = {}             # track_id -> total frames seen
    track_lost = {}             # track_id -> consecutive frames lost
    track_smooth_bbox = {}      # track_id -> EMA-smoothed (x1, y1, x2, y2)
    track_conf_hist = {}        # track_id -> list of recent confidences
    track_label_hist = {}       # track_id -> list of recent labels
    track_display_label = {}    # track_id -> smoothed majority label
    track_display_conf = {}     # track_id -> smoothed average confidence
    reported_tracks = set()     # track_ids already reported to backend
    recently_reported = []      # list of dicts: {"bbox": (x1,y1,x2,y2), "label": str, "timestamp": float}

    while True:
        success, frame = cap.read()
        if not success:
            print("Finished stream or failed to read frame. Exiting...")
            break
            
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
            
        display_frame = frame.copy()
        h, w, _ = display_frame.shape

        if not yolo:
            # ────────────────────────────────────────────────────────────────
            # CNN-only Mode (e.g. Raspberry Pi)
            # ────────────────────────────────────────────────────────────────
            if cnn:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                label, confidence = cnn.predict(pil_img)
                
                is_fault = label in ["damaged", "disconnected", "misroute"]
                box_color = (0, 0, 255) if is_fault else (0, 255, 0)
                
                status_text = f"[{cfg.DEVICE_PROFILE.upper()}] {label} ({confidence:.1%}) | FPS: {fps:.1f}"
                cv2.rectangle(display_frame, (0, 0), (w, 40), (0, 0, 0), -1)
                cv2.putText(display_frame, status_text, (10, 26), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if is_fault:
                    cv2.rectangle(display_frame, (10, 10), (w-10, h-10), box_color, 4)
                    cv2.putText(display_frame, "⚠️ FAULT WARNING", (30, h-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, box_color, 3)
                    
                    # Check cooldown
                    current_time = time.time()
                    if label not in last_reported_time or (current_time - last_reported_time[label] >= cfg.REPORT_COOLDOWN):
                        if send_fault_to_backend(frame, label, confidence):
                            last_reported_time[label] = current_time
            else:
                cv2.putText(display_frame, "No models loaded.", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # ────────────────────────────────────────────────────────────────
            # YOLO + CNN tracking mode (e.g. PC)
            # ────────────────────────────────────────────────────────────────
            results = yolo.detect(frame, persist=True)
            crops = yolo.get_crops(frame, results)
            seen_this_frame = set()

            for item in crops:
                tid = item['track_id']
                if tid is None:
                    continue

                raw_bbox = item['bbox']
                yolo_label = item['label']
                yolo_conf = item['conf']
                seen_this_frame.add(tid)

                track_seen[tid] = track_seen.get(tid, 0) + 1
                track_lost[tid] = 0
                track_smooth_bbox[tid] = ema_smooth(track_smooth_bbox.get(tid), raw_bbox)

                label = yolo_label
                conf = yolo_conf
                if cnn and yolo_label.lower() == "insulator":
                    label, conf = cnn.predict(item['image'])

                track_conf_hist.setdefault(tid, []).append(conf)
                track_label_hist.setdefault(tid, []).append(label)

                if len(track_conf_hist[tid]) > cfg.HISTORY_LEN:
                    track_conf_hist[tid] = track_conf_hist[tid][-cfg.HISTORY_LEN:]
                    track_label_hist[tid] = track_label_hist[tid][-cfg.HISTORY_LEN:]

                avg_conf = sum(track_conf_hist[tid]) / len(track_conf_hist[tid])
                majority_label = Counter(track_label_hist[tid]).most_common(1)[0][0]

                track_display_label[tid] = majority_label
                track_display_conf[tid] = avg_conf

                # Deduplicate and send to backend
                if tid not in reported_tracks and track_seen[tid] >= cfg.PERSIST_FRAMES:
                    if avg_conf >= cfg.MIN_REPORT_CONF:
                        now = time.time()
                        recently_reported = [r for r in recently_reported if now - r["timestamp"] < 10.0]
                        current_bbox = track_smooth_bbox[tid]
                        is_duplicate = False

                        for r in recently_reported:
                            if r["label"] == majority_label:
                                iou_score = calculate_iou(current_bbox, r["bbox"])
                                if iou_score > 0.3:
                                    is_duplicate = True
                                    print(f"Spatial Deduplication: Track #{tid} ({majority_label}) overlaps with recently reported object (IoU: {iou_score:.2%}). Skipping.")
                                    break
                                    
                        if is_duplicate:
                            reported_tracks.add(tid)
                        else:
                            if send_fault_to_backend(frame, majority_label, avg_conf):
                                reported_tracks.add(tid)
                                recently_reported.append({
                                    "bbox": current_bbox,
                                    "label": majority_label,
                                    "timestamp": now
                                })

            # Update lost counts
            for tid in list(track_seen.keys()):
                if tid not in seen_this_frame:
                    track_lost[tid] = track_lost.get(tid, 0) + 1

            # Bounding boxes rendering
            for tid in list(track_seen.keys()):
                lost_count = track_lost.get(tid, 0)
                if lost_count > cfg.HOLD_FRAMES:
                    continue

                if tid not in track_smooth_bbox or tid not in track_display_label:
                    continue

                bx1, by1, bx2, by2 = track_smooth_bbox[tid]
                lbl = track_display_label[tid]
                cnf = track_display_conf[tid]

                if tid not in seen_this_frame:
                    color = (0, 165, 255)  # Orange (hold)
                    thick = 1
                    tag = f" [HOLD {lost_count}/{cfg.HOLD_FRAMES}]"
                elif tid in reported_tracks:
                    color = (0, 255, 0)  # Green (reported)
                    thick = 2
                    tag = " [SENT]"
                else:
                    color = (255, 200, 0)  # Cyan (tracking)
                    thick = 2
                    tag = f" [{track_seen[tid]}/{cfg.PERSIST_FRAMES}]"

                cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), color, thick)
                text = f"#{tid} {lbl} ({cnf:.0%}){tag}"
                cv2.putText(display_frame, text, (bx1, max(by1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            # Cleanup old tracks
            for tid in list(track_seen.keys()):
                if track_lost.get(tid, 0) > cfg.CLEANUP_FRAMES:
                    for d in (track_seen, track_lost, track_smooth_bbox,
                              track_conf_hist, track_label_hist,
                              track_display_label, track_display_conf):
                        d.pop(tid, None)
                    reported_tracks.discard(tid)

            # Draw status bar
            status_text = f"[{cfg.DEVICE_PROFILE.upper()}] YOLO+CNN | FPS: {fps:.1f}"
            cv2.rectangle(display_frame, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.putText(display_frame, status_text, (10, 26), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        # Display visualization window
        cv2.imshow(f"Edge Gateway [{cfg.DEVICE_PROFILE.upper()}]", display_frame)
        
        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Stream stopped. Exiting gateway.")

if __name__ == "__main__":
    run_gateway()

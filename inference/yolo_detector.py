from ultralytics import YOLO
import cv2
from PIL import Image

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt'):
        # Load YOLOv8 model
        self.model = YOLO(model_path)
        
    def detect(self, frame, persist=True):
        """
        Detect and track objects in a frame using ByteTrack.
        """
        if persist:
            # Run Ultralytics tracking with ByteTrack configuration
            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        else:
            results = self.model(frame, verbose=False)
        return results[0]

    def get_crops(self, frame, results):
        """
        Extract cropped images from the frame based on YOLO detections.
        Returns a list of dicts containing image, bbox, label, conf, and track_id.
        """
        crops = []
        for box in results.boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            
            # Extract track_id if available
            track_id = int(box.id.item()) if box.id is not None else None
            
            # Crop using OpenCV (BGR)
            crop_cv2 = frame[y1:y2, x1:x2]
            
            if crop_cv2.size == 0:
                continue
                
            # Convert to PIL (RGB) for CNN
            crop_pil = Image.fromarray(cv2.cvtColor(crop_cv2, cv2.COLOR_BGR2RGB))
            
            crops.append({
                'image': crop_pil,
                'bbox': (x1, y1, x2, y2),
                'label': cls_name,
                'conf': float(box.conf[0]),
                'track_id': track_id
            })
            
        return crops

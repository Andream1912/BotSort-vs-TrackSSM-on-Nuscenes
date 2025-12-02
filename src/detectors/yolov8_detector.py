"""YOLOv8 Detector - Ultralytics implementation"""
import numpy as np
from typing import List, Dict
from ultralytics import YOLO


class YOLOv8Detector:
    """YOLOv8 detector wrapper using Ultralytics"""
    
    # COCO to NuScenes class mapping
    coco_to_nuscenes = {
        0: 5,   # person → pedestrian
        2: 1,   # car → car
        3: 6,   # motorcycle → motorcycle
        5: 2,   # bus → truck (approximation)
        7: 2,   # truck → truck
        1: 7,   # bicycle → bicycle
    }
    
    def __init__(self, model_path: str = 'yolov8x.pt', conf_thresh: float = 0.1, 
                 device: str = 'cuda', imgsz: int = 1280):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_path: Path to YOLOv8 weights or model name (yolov8n/s/m/l/x.pt)
            conf_thresh: Confidence threshold (default: 0.1)
            device: Device for inference (cuda/cpu)
            imgsz: Input image size (default: 1280 for better small object detection)
        """
        self.conf_thresh = conf_thresh
        self.device = device
        self.imgsz = imgsz
        
        print(f"Loading YOLOv8 from {model_path}...")
        self.model = YOLO(model_path)
        self.model.to(device)
        print(f"✓ Loaded YOLOv8 checkpoint")
    
    def detect(self, img: np.ndarray) -> List[Dict]:
        """
        Run detection on image.
        
        Args:
            img: numpy array (H, W, 3) in BGR format
        
        Returns:
            List of dicts with keys: bbox [x,y,w,h], confidence, class_id
        """
        # Run inference
        results = self.model.predict(
            img, 
            conf=self.conf_thresh,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )[0]
        
        detections = []
        
        # Parse results
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return []
        
        for box in boxes:
            # Get box coordinates (xyxy format)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            coco_class = int(box.cls[0])
            
            # Map COCO class to NuScenes
            if coco_class not in self.coco_to_nuscenes:
                continue  # Skip classes not relevant for autonomous driving
            
            nuscenes_class = self.coco_to_nuscenes[coco_class]
            
            # Convert to x,y,w,h format
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1
            
            detections.append({
                'bbox': [x, y, w, h],
                'confidence': conf,
                'class_id': nuscenes_class
            })
        
        return detections

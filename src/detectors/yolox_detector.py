"""YOLOX Detector - Shared by all trackers"""
import os
import sys
import torch
import numpy as np
from typing import List, Dict, Any

# Add YOLOX to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
yolox_path = os.path.join(project_root, 'external', 'YOLOX')
if yolox_path not in sys.path:
    sys.path.insert(0, yolox_path)

try:
    from yolox.exp import get_exp
    from yolox.utils import fuse_model, postprocess
    from yolox.data.data_augment import ValTransform
    YOLOX_AVAILABLE = True
except ImportError:
    YOLOX_AVAILABLE = False
    print("⚠️  YOLOX not available")


class YOLOXDetector:
    """YOLOX detector wrapper - shared by all trackers"""
    
    def __init__(self, model_path: str, conf_thresh: float = 0.1, 
                 nms_thresh: float = 0.65, device: str = 'cuda', test_size: tuple = (1280, 1280),
                 model_name: str = None, num_classes: int = None):
        """
        Initialize YOLOX detector.
        
        Args:
            model_path: Path to YOLOX checkpoint
            conf_thresh: Confidence threshold (default: 0.1)
            nms_thresh: NMS threshold (default: 0.65)
            device: Device for inference
            test_size: Input size for inference (default: 640x640)
            model_name: Model variant (yolox-s/m/l/x, auto-detect from path if None)
            num_classes: Number of classes (None=auto-detect from checkpoint, 80=COCO, 7=nuScenes)
        """
        if not YOLOX_AVAILABLE:
            raise ImportError("YOLOX not installed. Install with: pip install yolox")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLOX model not found: {model_path}")
        
        self.device = device
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.test_size = test_size
        
        # Auto-detect model variant from path
        if model_name is None:
            if 'yolox_l' in model_path.lower():
                model_name = 'yolox-l'
            elif 'yolox_x' in model_path.lower():
                model_name = 'yolox-x'
            elif 'yolox_m' in model_path.lower():
                model_name = 'yolox-m'
            elif 'yolox_s' in model_path.lower():
                model_name = 'yolox-s'
            else:
                model_name = 'yolox-x'  # default
        
        # Load YOLOX model
        exp = get_exp(None, model_name)
        
        # Auto-detect num_classes from checkpoint if not specified
        if num_classes is None:
            ckpt_temp = torch.load(model_path, map_location='cpu')
            if 'model' in ckpt_temp and 'head.cls_preds.0.bias' in ckpt_temp['model']:
                num_classes = ckpt_temp['model']['head.cls_preds.0.bias'].shape[0]
                print(f"Auto-detected {num_classes} classes from checkpoint")
            else:
                num_classes = 80  # Default COCO
            del ckpt_temp
        
        exp.num_classes = num_classes
        exp.test_size = test_size  # Set test size in exp
        self.num_classes = num_classes
        self.model = exp.get_model()
        self.model.to(device)
        self.model.eval()
        
        # Load checkpoint
        print(f"Loading YOLOX from {model_path}...")
        ckpt = torch.load(model_path, map_location=device)
        self.model.load_state_dict(ckpt["model"])
        print(f"✓ Loaded YOLOX-X checkpoint with {num_classes} classes at {test_size}")
        
        self.model = fuse_model(self.model)
        self.preprocess = ValTransform(legacy=False)
        
        # COCO to NuScenes class mapping (only for 80-class COCO models)
        # COCO: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
        # NuScenes: 1=car, 2=truck, 3=bus, 4=trailer, 5=pedestrian, 6=motorcycle, 7=bicycle
        self.coco_to_nuscenes = {
            0: 5,  # person -> pedestrian
            1: 7,  # bicycle
            2: 1,  # car
            3: 6,  # motorcycle
            5: 3,  # bus
            7: 2,  # truck
        }
        
        # Check if we need class mapping
        self.use_class_mapping = (num_classes == 80)  # Only map for COCO models
    
    def detect(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run detection on image.
        
        Args:
            img: numpy array (H, W, 3) in BGR format
        
        Returns:
            List of dicts with keys: bbox [x,y,w,h], confidence, class_id
        """
        # Preprocess
        img_info = {"height": img.shape[0], "width": img.shape[1]}
        img_preprocessed, _ = self.preprocess(img, None, self.test_size)
        img_tensor = torch.from_numpy(img_preprocessed).unsqueeze(0).float().to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            outputs = postprocess(
                outputs, 
                num_classes=self.num_classes,  # Use actual model classes
                conf_thre=self.conf_thresh,
                nms_thre=self.nms_thresh
            )[0]
        
        if outputs is None:
            return []
        
        # Convert to standard detection format
        detections = []
        ratio = min(self.test_size[0] / img_info["height"], self.test_size[1] / img_info["width"])
        
        outputs = outputs.cpu().numpy()
        for detection in outputs:
            x1, y1, x2, y2, obj_conf, class_conf, class_id = detection
            
            # Rescale to original image
            x1 /= ratio
            y1 /= ratio
            x2 /= ratio
            y2 /= ratio
            
            # Convert to xywh format
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            
            # Map COCO class to NuScenes (only for 80-class models)
            if self.use_class_mapping:
                coco_class = int(class_id)
                if coco_class not in self.coco_to_nuscenes:
                    continue  # Skip irrelevant classes
                nuscenes_class = self.coco_to_nuscenes[coco_class]
            else:
                # Already in nuScenes format (7-class model)
                nuscenes_class = int(class_id) + 1  # Convert from 0-indexed to 1-indexed
            
            detections.append({
                'bbox': [x, y, w, h],
                'confidence': float(obj_conf * class_conf),
                'class_id': nuscenes_class  # Use native 7-class NuScenes IDs
            })
        
        return detections

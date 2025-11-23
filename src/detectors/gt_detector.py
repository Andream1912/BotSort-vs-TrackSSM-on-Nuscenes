"""GT Detector - Use ground truth as detections (oracle mode)"""
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


class GTDetector:
    """Ground truth detector - reads GT from MOT format files"""
    
    def __init__(self, data_root: str):
        """
        Initialize GT detector.
        
        Args:
            data_root: Path to dataset root (e.g., data/nuscenes_mot_front/val)
        """
        self.data_root = Path(data_root)
        self.gt_cache = {}
        
    def load_scene_gt(self, scene_name: str):
        """Load GT for a scene"""
        if scene_name in self.gt_cache:
            return
        
        gt_file = self.data_root / scene_name / 'gt' / 'gt.txt'
        
        if not gt_file.exists():
            print(f"⚠️  GT file not found: {gt_file}")
            self.gt_cache[scene_name] = {}
            return
        
        # Parse GT file
        scene_gt = {}
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue
                
                frame_id = int(parts[0])
                x, y, w, h = map(float, parts[2:6])
                conf = float(parts[6]) if len(parts) > 6 else 1.0
                class_id = int(parts[7]) if len(parts) > 7 else 1
                
                if frame_id not in scene_gt:
                    scene_gt[frame_id] = []
                
                scene_gt[frame_id].append({
                    'bbox': [x, y, w, h],
                    'confidence': conf,
                    'class_id': class_id
                })
        
        self.gt_cache[scene_name] = scene_gt
    
    def detect(self, scene_name: str, frame_id: int) -> List[Dict[str, Any]]:
        """
        Get GT detections for a frame.
        
        Args:
            scene_name: Name of the scene
            frame_id: Frame number (1-indexed)
        
        Returns:
            List of detections with bbox, confidence, class_id
        """
        # Load scene GT if not cached
        if scene_name not in self.gt_cache:
            self.load_scene_gt(scene_name)
        
        # Return detections for this frame
        return self.gt_cache[scene_name].get(frame_id, [])

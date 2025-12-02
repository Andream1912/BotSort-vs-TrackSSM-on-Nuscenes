"""BoT-SORT Tracker - Official implementation wrapper for NuScenes"""
import sys
import os
import numpy as np
from typing import List, Dict, Any
from argparse import Namespace

# Add BoT-SORT to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
botsort_path = os.path.join(project_root, 'external', 'BoT-SORT')
if botsort_path not in sys.path:
    sys.path.insert(0, botsort_path)

from src.trackers.base_tracker import BaseTracker

# Import BoT-SORT
from tracker.bot_sort import BoTSORT as BoTSORT_Single
from tracker.mc_bot_sort import BoTSORT as BoTSORT_Multi


class BotSortTracker(BaseTracker):
    """BoT-SORT tracker wrapper for NuScenes"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Multi-class support
        self.multi_class = config.get('multi_class', True)
        
        # Frame rate
        self.frame_rate = config.get('frame_rate', 12)  # nuScenes is 12Hz
        
        # Create args namespace for BoTSORT
        args = Namespace()
        # Map unified parameters to BoT-SORT naming
        args.track_high_thresh = config.get('track_thresh', config.get('track_high_thresh', 0.6))
        args.track_low_thresh = config.get('track_low_thresh', 0.1)
        args.new_track_thresh = config.get('new_track_thresh', 0.5)
        args.track_buffer = config.get('max_age', config.get('track_buffer', 30))  # max_age → track_buffer
        args.match_thresh = config.get('match_thresh', 0.8)
        args.proximity_thresh = config.get('proximity_thresh', 0.5)
        args.appearance_thresh = config.get('appearance_thresh', 0.25)
        args.cmc_method = config.get('cmc_method', 'sparseOptFlow')  # sparseOptFlow, orb, ecc
        args.with_reid = config.get('with_reid', True)
        args.fast_reid_config = config.get('fast_reid_config', 'external/BoT-SORT/fast_reid/configs/MOT17/sbs_S50.yml')
        args.fast_reid_weights = config.get('fast_reid_weights', 'weights/reid/mot17_sbs_S50.pth')
        args.device = config.get('device', 'cuda')
        args.name = config.get('name', 'botsort')
        args.ablation = config.get('ablation', '')
        args.mot20 = config.get('mot20', False)  # MOT20 dataset flag
        
        self.args = args
        
        # Initialize tracker
        if self.multi_class:
            print(f"Initializing MC-BoT-SORT (multi-class) with ReID={args.with_reid}, CMC={args.cmc_method}")
            self.tracker = BoTSORT_Multi(args, frame_rate=self.frame_rate)
        else:
            print(f"Initializing BoT-SORT (single-class) with ReID={args.with_reid}, CMC={args.cmc_method}")
            self.tracker = BoTSORT_Single(args, frame_rate=self.frame_rate)
    
    def update(self, detections: List[Dict], frame: Any = None) -> List[Dict]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of detection dicts with keys:
                - bbox: [x, y, w, h]
                - confidence: float
                - class_id: int (1-7 for nuScenes)
            frame: numpy array (H, W, 3) BGR image
        
        Returns:
            List of track dicts with keys:
                - track_id: int
                - bbox: [x, y, w, h]
                - class_id: int
                - confidence: float
        """
        self.frame_id += 1
        
        if len(detections) == 0:
            # Empty detections - just update with empty array
            if frame is not None:
                empty_dets = np.empty((0, 8))
                online_targets = self.tracker.update(empty_dets, frame)
            else:
                return []
        else:
            # Convert detections to BoT-SORT format
            # BoT-SORT expects: [x1, y1, x2, y2, conf, class_id, feat1, feat2, ...]
            # We don't have ReID features from detector, so use dummy features
            dets = []
            for det in detections:
                x, y, w, h = det['bbox']
                x1, y1, x2, y2 = x, y, x + w, y + h
                conf = det['confidence']
                cls_id = det['class_id']
                
                # BoT-SORT format: [x1, y1, x2, y2, conf, class] + dummy features
                # Add dummy ReID features (will be computed from image if with_reid=True)
                dets.append([x1, y1, x2, y2, conf, cls_id, 0, 0])
            
            dets = np.array(dets, dtype=np.float32)
            
            # Update tracker
            online_targets = self.tracker.update(dets, frame)
        
        # Convert tracks back to our format
        output_tracks = []
        for track in online_targets:
            # BoT-SORT returns STrack objects
            tlwh = track.tlwh  # [x, y, w, h]
            track_id = track.track_id
            cls_id = int(track.cls) if hasattr(track, 'cls') else 1  # Default to car
            score = track.score
            
            output_tracks.append({
                'track_id': track_id,
                'bbox': tlwh.tolist(),
                'class_id': cls_id,
                'confidence': score
            })
        
        return output_tracks
    
    def reset(self):
        """Reset tracker state"""
        self.frame_id = 0
        if self.multi_class:
            self.tracker = BoTSORT_Multi(self.args, frame_rate=self.frame_rate)
        else:
            self.tracker = BoTSORT_Single(self.args, frame_rate=self.frame_rate)

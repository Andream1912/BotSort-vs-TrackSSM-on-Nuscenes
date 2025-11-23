"""BotSort Tracker - Wrapper for BoT-SORT"""
import sys
import os
import numpy as np
from typing import List, Dict, Any

# Add paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.trackers.base_tracker import BaseTracker
from tracker.BYTETracker import BYTETracker as BYTETrackerImpl


class BotSortTracker(BaseTracker):
    """BotSort tracker implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # BotSort uses BYTETracker internally
        # TODO: Integrate actual BoT-SORT implementation
        # For now, use BYTE tracker as baseline
        
        self.track_thresh = config.get('track_thresh', 0.5)
        self.match_thresh = config.get('match_thresh', 0.8)
        self.max_age = config.get('max_age', 30)
        
        # Initialize BYTETracker
        # Note: This is a placeholder - actual BoT-SORT needs ReID model
        print("⚠️  BotSort requires ReID model - using BYTE tracker baseline")
        
    def update(self, detections: List[Dict], frame: Any = None) -> List[Dict]:
        """Update tracks"""
        # TODO: Implement BotSort properly with ReID features
        # For now, simple placeholder that returns detections as tracks
        
        self.frame_id += 1
        
        # Simple ID assignment for testing
        output_tracks = []
        for i, det in enumerate(detections):
            if det['confidence'] >= self.track_thresh:
                output_tracks.append({
                    'track_id': i + 1,
                    'bbox': det['bbox'],
                    'class_id': det['class_id'],
                    'confidence': det['confidence']
                })
        
        return output_tracks
    
    def reset(self):
        """Reset tracker"""
        self.frame_id = 0
        self.tracks = []

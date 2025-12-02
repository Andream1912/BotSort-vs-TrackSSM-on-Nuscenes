"""TrackSSM Tracker Wrapper - Adapts TrackSSMTracker to BaseTracker interface"""
import sys
import os
from typing import List, Dict, Any

# Add project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.trackers.base_tracker import BaseTracker

# Import original TrackSSMTracker implementation
from src.trackers.trackssm_tracker import TrackSSMTracker as TrackSSMTrackerImpl


class TrackSSMTracker(BaseTracker):
    """Wrapper for TrackSSMTracker to comply with BaseTracker interface"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Extract config
        model = config.get('model')
        device = config.get('device', 'cuda')
        img_width = config.get('img_width', 1600)
        img_height = config.get('img_height', 900)
        track_thresh = config.get('track_thresh', 0.2)
        match_thresh = config.get('match_thresh', 0.5)
        max_age = config.get('max_age', 30)
        min_hits = config.get('min_hits', 3)
        history_len = config.get('history_len', 5)
        oracle_mode = config.get('oracle_mode', False)  # GT track IDs mode
        
        # Initialize actual tracker
        self.tracker = TrackSSMTrackerImpl(
            model=model,
            device=device,
            img_width=img_width,
            img_height=img_height,
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            max_age=max_age,
            min_hits=min_hits,
            history_len=history_len,
            oracle_mode=False  # Always False: GT dets should be tracked normally
        )
    
    def update(self, detections: List[Dict], frame: Any = None) -> List[Dict]:
        """Update tracks with new detections"""
        return self.tracker.update(detections)
    
    def reset(self):
        """Reset tracker state"""
        self.tracker.reset()

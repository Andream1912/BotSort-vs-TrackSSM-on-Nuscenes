"""Base Tracker Interface - Strategy Pattern"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseTracker(ABC):
    """Abstract base class for all trackers"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize tracker with configuration.
        
        Args:
            config: Dictionary with tracker-specific configuration
        """
        self.config = config
        self.frame_id = 0
        self.tracks = []
    
    @abstractmethod
    def update(self, detections: List[Dict], frame: Any = None) -> List[Dict]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of dicts with keys: bbox [x,y,w,h], confidence, class_id
            frame: Optional frame image for appearance features
        
        Returns:
            List of active tracks with keys: track_id, bbox, class_id, confidence
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset tracker state"""
        pass
    
    def get_name(self) -> str:
        """Get tracker name"""
        return self.__class__.__name__.replace('Tracker', '').lower()

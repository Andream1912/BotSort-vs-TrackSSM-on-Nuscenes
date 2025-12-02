"""
TrackSSM Motion Predictor - Drop-in replacement for Kalman Filter in BoT-SORT

This module provides the same interface as KalmanFilter but uses TrackSSM
for motion prediction instead.
"""

import numpy as np
import torch
from collections import deque


class TrackSSMMotion:
    """
    TrackSSM-based motion predictor.
    
    Provides the same interface as KalmanFilter:
    - initiate(measurement): Initialize new track
    - predict(mean, covariance, track_id): Predict next position (FIXED: uses track_id)
    - update(mean, covariance, measurement, track_id): Update with new detection (FIXED: uses track_id)
    
    But uses TrackSSM neural network instead of Kalman dynamics.
    
    CRITICAL FIX: Now properly uses track_id from BoT-SORT instead of position-based matching
    """
    
    def __init__(self, model, device, img_width=1600, img_height=900, history_len=5):
        """
        Args:
            model: TrackSSM model
            device: torch device
            img_width: Image width
            img_height: Image height
            history_len: Number of history frames (default: 5)
        """
        self.model = model
        self.device = device
        self.img_width = img_width
        self.img_height = img_height
        self.history_len = history_len
        
        # Track history storage: dict[track_id] -> deque of bboxes
        # FIXED: Now uses actual BoT-SORT track_id, not internal counter
        self.track_histories = {}
        
        # Track metadata: dict[track_id] -> {'class_id': int, 'confidence': float}
        self.track_metadata = {}
    
    def initiate(self, measurement, track_id=None, class_id=1, confidence=1.0):
        """
        Initialize a new track (same interface as KalmanFilter).
        
        Args:
            measurement: [cx, cy, aspect_ratio, height] in image coordinates
                         WARNING: In practice BoT-SORT may pass [cx, cy, w, h] instead!
            track_id: BoT-SORT track_id (FIXED: now passed from BoT-SORT)
            class_id: Object class ID (1-7 for NuScenes)
            confidence: Detection confidence
        
        Returns:
            mean: Initial state [cx, cy, aspect_ratio, height, vx, vy, va, vh]
            covariance: Initial covariance (dummy for compatibility)
        """
        # BUG FIX: BoT-SORT passes [cx, cy, w, h], NOT [cx, cy, aspect_ratio, h]!
        # Check if measurement[2] looks like width (> 10) or aspect_ratio (< 10)
        cx, cy, param3, h = measurement
        
        if param3 > 10:  # Likely width in pixels
            w = param3
            aspect_ratio = w / max(h, 1e-6)
        else:  # Likely aspect_ratio
            aspect_ratio = param3
            w = aspect_ratio * h
        
        # DEBUG: First initiate
        if not hasattr(self, '_debug_initiate_logged'):
            print(f"      🎬 initiate() called: measurement={measurement}")
            print(f"         cx={cx:.1f}, cy={cy:.1f}, param3={param3:.3f}, h={h:.1f}")
            print(f"         computed w={w:.1f}, aspect={aspect_ratio:.3f}")
            self._debug_initiate_logged = True
        
        # Initialize history with current measurement
        # FIXED: Use BoT-SORT track_id instead of internal counter
        if track_id is not None and track_id not in self.track_histories:
            self.track_histories[track_id] = deque(maxlen=self.history_len)
            self.track_histories[track_id].append([cx, cy, w, h])
            # Store metadata
            self.track_metadata[track_id] = {'class_id': class_id, 'confidence': confidence}
        
        # FIX: BoT-SORT expects [cx, cy, WIDTH, h], NOT [cx, cy, aspect_ratio, h]
        # The tlwh property in mc_bot_sort.py does: ret[:2] -= ret[2:] / 2
        # This only works correctly if mean[2] is width, not aspect_ratio
        # Mean state: [cx, cy, w, h, vx, vy, vw, vh]
        # Velocities initialized to 0
        mean = np.array([cx, cy, w, h, 0., 0., 0., 0.])
        
        # Dummy covariance for compatibility
        covariance = np.eye(8)
        
        return mean, covariance
    
    def predict(self, mean, covariance, track_id=None):
        """
        Predict next position using TrackSSM.
        
        Args:
            mean: Current state [cx, cy, w, h, vx, vy, vw, vh]  (FIX: width, not aspect_ratio!)
            covariance: Current covariance (ignored, kept for compatibility)
            track_id: BoT-SORT track_id (FIXED: now passed from BoT-SORT)
        
        Returns:
            pred_mean: Predicted state
            pred_covariance: Predicted covariance (dummy)
        """
        # Extract current position from mean
        # FIX: mean format is now [cx, cy, w, h, vx, vy, vw, vh] (width, not aspect_ratio)
        cx, cy, w, h = mean[:4]
        
        # FIXED: Use track_id directly instead of position-based search
        if track_id is None or track_id not in self.track_histories or len(self.track_histories[track_id]) == 0:
            # No history - return constant velocity model
            vx, vy, vw, vh = mean[4:8]
            pred_cx = cx + vx
            pred_cy = cy + vy
            pred_w = w + vw
            pred_h = h + vh
            pred_mean = np.array([pred_cx, pred_cy, pred_w, pred_h, vx, vy, vw, vh])
            return pred_mean, covariance
        
        # Get history
        history = list(self.track_histories[track_id])
        
        # Pad if needed
        if len(history) < self.history_len:
            history = [history[0]] * (self.history_len - len(history)) + history
        else:
            history = history[-self.history_len:]
        
        # Get track metadata (class_id and confidence)
        track_class = self.track_metadata.get(track_id, {}).get('class_id', 1)
        track_conf = self.track_metadata.get(track_id, {}).get('confidence', 1.0)
        
        # DEBUG: Log first few predictions with metadata
        if not hasattr(self, '_metadata_logged'):
            self._metadata_logged = 0
        if self._metadata_logged < 5:
            print(f"🔍 TrackSSM.predict() track_id={track_id}, class_id={track_class}, conf={track_conf:.3f}")
            self._metadata_logged += 1
        
        # Prepare TrackSSM input: (history_len, 8)
        # Format: [cx, cy, w, h, delta_cx, delta_cy, delta_w, delta_h] normalized
        # NOTE: TrackSSM expects deltas (velocities), NOT class_id/confidence!
        condition = []
        for i, bbox in enumerate(history):
            cx_h, cy_h, w_h, h_h = bbox
            
            # DEBUG: First prediction RAW values
            if not hasattr(self, '_debug_raw_logged') and i == len(history) - 1:
                print(f"      📊 RAW bbox: cx={cx_h:.1f}, cy={cy_h:.1f}, w={w_h:.1f}, h={h_h:.1f}")
                print(f"      📐 img_size: {self.img_width}x{self.img_height}")
                self._debug_raw_logged = True
            
            # Normalize current bbox
            cx_norm = cx_h / self.img_width
            cy_norm = cy_h / self.img_height
            w_norm = w_h / self.img_width
            h_norm = h_h / self.img_height
            
            # Calculate deltas (velocities) from previous frame
            if i > 0:
                prev_cx, prev_cy, prev_w, prev_h = history[i-1]
                delta_cx = (cx_h - prev_cx) / self.img_width
                delta_cy = (cy_h - prev_cy) / self.img_height
                delta_w = (w_h - prev_w) / self.img_width
                delta_h = (h_h - prev_h) / self.img_height
            else:
                # First frame: zero velocity
                delta_cx = delta_cy = delta_w = delta_h = 0.0
            
            # FIXED: Use deltas instead of class_id/confidence
            condition.append([cx_norm, cy_norm, w_norm, h_norm, delta_cx, delta_cy, delta_w, delta_h])
        
        condition = np.array(condition, dtype=np.float32)
        
        # Run TrackSSM prediction
        try:
            with torch.no_grad():
                condition_tensor = torch.from_numpy(condition).unsqueeze(0).to(self.device)
                
                # Encode
                cond_encoded = self.model.encoder(condition_tensor)
                
                # Decode
                last_bbox_norm = condition_tensor[0, -1, :4]
                pred_bbox_norm = self.model.ssm_decoder(
                    last_bbox_norm.unsqueeze(0),
                    cond_encoded,
                    h=None
                )
                pred_bbox_norm = pred_bbox_norm.squeeze(0).cpu().numpy()
            
            # Denormalize
            cx_pred_norm, cy_pred_norm, w_pred_norm, h_pred_norm = pred_bbox_norm
            
            # Sanity checks
            if (np.any(np.isnan(pred_bbox_norm)) or 
                np.any(np.isinf(pred_bbox_norm)) or
                w_pred_norm <= 0 or h_pred_norm <= 0 or
                w_pred_norm > 2.0 or h_pred_norm > 2.0):
                # Fallback to constant velocity
                vx, vy, vw, vh = mean[4:8]
                pred_mean = np.array([cx + vx, cy + vy, w + vw, h + vh, vx, vy, vw, vh])
                return pred_mean, covariance
            
            # Convert back to BoT-SORT format
            cx_pred = cx_pred_norm * self.img_width
            cy_pred = cy_pred_norm * self.img_height
            w_pred = w_pred_norm * self.img_width
            h_pred = h_pred_norm * self.img_height
            
            aspect_pred = w_pred / max(h_pred, 1e-6)
            
            # Compute velocities
            vx = cx_pred - cx
            vy = cy_pred - cy
            vw = w_pred - w  # FIX: Use width velocity, not aspect_ratio velocity
            vh = h_pred - h
            
            # FIX: BoT-SORT expects [cx, cy, WIDTH, h], NOT [cx, cy, aspect_ratio, h]!
            # The tlwh property does: ret[:2] -= ret[2:] / 2, which only works if mean[2] is width
            pred_mean = np.array([cx_pred, cy_pred, w_pred, h_pred, vx, vy, vw, vh])
            
            # DON'T update history with prediction! History should only contain TRUE detections.
            # History will be updated in update() method when we receive the actual detection.
            
            return pred_mean, covariance
            
        except Exception as e:
            # Fallback to constant velocity on any error
            vx, vy, vw, vh = mean[4:8]
            pred_mean = np.array([cx + vx, cy + vy, w + vw, h + vh, vx, vy, vw, vh])
            return pred_mean, covariance
    
    def update(self, mean, covariance, measurement, track_id=None, class_id=None, confidence=None):
        """
        Update track with new measurement (same interface as KalmanFilter).
        
        Args:
            mean: Current state [cx, cy, w, h, vx, vy, vw, vh]  (FIX: width, not aspect_ratio!)
            covariance: Current covariance
            measurement: New measurement [cx, cy, w, h] (BoT-SORT passes width, not aspect_ratio!)
            track_id: BoT-SORT track_id (FIXED: now passed from BoT-SORT)
            class_id: Updated object class ID (optional)
            confidence: Updated detection confidence (optional)
        
        Returns:
            updated_mean: Updated state
            updated_covariance: Updated covariance
        """
        # BUG FIX: measurement might be [cx, cy, w, h] instead of [cx, cy, aspect_ratio, h]
        cx, cy, param3, h = measurement
        
        if param3 > 10:  # Likely width in pixels
            w = param3
            aspect_ratio = w / max(h, 1e-6)
        else:  # Likely aspect_ratio
            aspect_ratio = param3
            w = aspect_ratio * h
        
        # FIXED: Use track_id directly
        if track_id is not None:
            # Initialize history if needed
            if track_id not in self.track_histories:
                self.track_histories[track_id] = deque(maxlen=self.history_len)
            # Update history with TRUE detection
            self.track_histories[track_id].append([cx, cy, w, h])
            
            # Update metadata if provided
            if class_id is not None or confidence is not None:
                if track_id not in self.track_metadata:
                    self.track_metadata[track_id] = {}
                if class_id is not None:
                    self.track_metadata[track_id]['class_id'] = class_id
                if confidence is not None:
                    self.track_metadata[track_id]['confidence'] = confidence
        
        # Compute velocities
        # FIX: mean is now [cx, cy, w, h, vx, vy, vw, vh] (width, not aspect_ratio)
        vx = cx - mean[0]
        vy = cy - mean[1]
        vw = w - mean[2]
        vh = h - mean[3]
        
        updated_mean = np.array([cx, cy, w, h, vx, vy, vw, vh])
        
        return updated_mean, covariance
    
    def cleanup_track(self, track_id):
        """Remove track history when track is deleted"""
        if track_id in self.track_histories:
            del self.track_histories[track_id]

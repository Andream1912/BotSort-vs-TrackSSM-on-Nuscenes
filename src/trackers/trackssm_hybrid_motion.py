"""
Hybrid TrackSSM + Kalman Motion Predictor

Combina le predizioni di TrackSSM e Kalman per ottenere il meglio di entrambi:
- Kalman: robusto, velocity model semplice
- TrackSSM: può imparare pattern complessi (se trainato bene)

Weighted average: alpha * trackssm + (1 - alpha) * kalman
"""

import numpy as np
import torch
from collections import deque


class HybridMotion:
    """
    Hybrid motion predictor che combina TrackSSM e Kalman Filter.
    
    Args:
        model: TrackSSM model
        device: torch device
        alpha: Weight per TrackSSM (0.0 = solo Kalman, 1.0 = solo TrackSSM)
        img_width: Image width
        img_height: Image height
        history_len: Number of history frames
    """
    
    def __init__(self, model, device, alpha=0.3, img_width=1600, img_height=900, history_len=5):
        self.model = model
        self.device = device
        self.alpha = alpha  # TrackSSM weight
        self.beta = 1.0 - alpha  # Kalman weight
        self.img_width = img_width
        self.img_height = img_height
        self.history_len = history_len
        
        # Track history storage
        self.track_histories = {}
        self.track_metadata = {}
        
        # Statistics
        self.n_predict_calls = 0
        self.n_trackssm_used = 0
        self.n_kalman_only = 0
        self.n_hybrid = 0
        
        print(f"🔀 Hybrid Motion Predictor initialized:")
        print(f"   TrackSSM weight (α): {self.alpha:.2f}")
        print(f"   Kalman weight (β): {self.beta:.2f}")
    
    def initiate(self, measurement, track_id=None, class_id=1, confidence=1.0):
        """Initialize a new track."""
        cx, cy, aspect_ratio, h = measurement
        w = aspect_ratio * h
        
        if track_id is not None and track_id not in self.track_histories:
            self.track_histories[track_id] = deque(maxlen=self.history_len)
            self.track_histories[track_id].append([cx, cy, w, h])
            self.track_metadata[track_id] = {'class_id': class_id, 'confidence': confidence}
        
        mean = np.array([cx, cy, aspect_ratio, h, 0., 0., 0., 0.])
        covariance = np.eye(8)
        
        return mean, covariance
    
    def predict(self, mean, covariance, track_id=None):
        """
        Hybrid prediction: weighted combination of TrackSSM and Kalman.
        """
        self.n_predict_calls += 1
        
        # Extract current state
        cx, cy, aspect_ratio, h = mean[:4]
        vx, vy, va, vh = mean[4:8]
        w = aspect_ratio * h
        
        # 1. KALMAN PREDICTION (always computed as fallback)
        kalman_pred = self._predict_kalman(mean)
        
        # 2. TRACKSSM PREDICTION (if history available)
        if track_id is not None and track_id in self.track_histories and len(self.track_histories[track_id]) >= 2:
            try:
                trackssm_pred = self._predict_trackssm(mean, track_id)
                
                # 3. HYBRID: Weighted combination
                pred_mean = self.alpha * trackssm_pred + self.beta * kalman_pred
                self.n_hybrid += 1
                
            except Exception as e:
                # Fallback to Kalman on error
                pred_mean = kalman_pred
                self.n_kalman_only += 1
        else:
            # No history: use Kalman only
            pred_mean = kalman_pred
            self.n_kalman_only += 1
        
        return pred_mean, covariance
    
    def _predict_kalman(self, mean):
        """Standard Kalman constant velocity prediction."""
        cx, cy, aspect_ratio, h = mean[:4]
        vx, vy, va, vh = mean[4:8]
        
        pred_cx = cx + vx
        pred_cy = cy + vy
        pred_aspect = aspect_ratio + va
        pred_h = h + vh
        
        return np.array([pred_cx, pred_cy, pred_aspect, pred_h, vx, vy, va, vh])
    
    def _predict_trackssm(self, mean, track_id):
        """TrackSSM prediction using history."""
        cx, cy, aspect_ratio, h = mean[:4]
        vx, vy, va, vh = mean[4:8]
        w = aspect_ratio * h
        
        # Get history
        history = list(self.track_histories[track_id])
        
        # Pad if needed
        if len(history) < self.history_len:
            history = [history[0]] * (self.history_len - len(history)) + history
        else:
            history = history[-self.history_len:]
        
        # Prepare input: [cx, cy, w, h, delta_cx, delta_cy, delta_w, delta_h] normalized
        condition = []
        for i, bbox in enumerate(history):
            cx_h, cy_h, w_h, h_h = bbox
            
            # Normalize bbox
            cx_norm = cx_h / self.img_width
            cy_norm = cy_h / self.img_height
            w_norm = w_h / self.img_width
            h_norm = h_h / self.img_height
            
            # Calculate deltas
            if i > 0:
                prev_cx, prev_cy, prev_w, prev_h = history[i-1]
                delta_cx = (cx_h - prev_cx) / self.img_width
                delta_cy = (cy_h - prev_cy) / self.img_height
                delta_w = (w_h - prev_w) / self.img_width
                delta_h = (h_h - prev_h) / self.img_height
            else:
                delta_cx = delta_cy = delta_w = delta_h = 0.0
            
            condition.append([cx_norm, cy_norm, w_norm, h_norm, delta_cx, delta_cy, delta_w, delta_h])
        
        condition = np.array(condition, dtype=np.float32)
        
        # Run TrackSSM
        with torch.no_grad():
            condition_tensor = torch.from_numpy(condition).unsqueeze(0).to(self.device)
            cond_encoded = self.model.encoder(condition_tensor)
            last_bbox_norm = condition_tensor[0, -1, :4]
            pred_bbox_norm = self.model.ssm_decoder(last_bbox_norm.unsqueeze(0), cond_encoded, h=None)
            pred_bbox_norm = pred_bbox_norm.squeeze(0).cpu().numpy()
        
        # Denormalize
        cx_pred_norm, cy_pred_norm, w_pred_norm, h_pred_norm = pred_bbox_norm
        
        # Sanity checks
        if (np.any(np.isnan(pred_bbox_norm)) or 
            np.any(np.isinf(pred_bbox_norm)) or
            w_pred_norm <= 0 or h_pred_norm <= 0 or
            w_pred_norm > 2.0 or h_pred_norm > 2.0):
            raise ValueError("Invalid TrackSSM prediction")
        
        # Convert to image coordinates
        cx_pred = cx_pred_norm * self.img_width
        cy_pred = cy_pred_norm * self.img_height
        w_pred = w_pred_norm * self.img_width
        h_pred = h_pred_norm * self.img_height
        
        aspect_pred = w_pred / max(h_pred, 1e-6)
        
        # Compute velocities
        vx_pred = cx_pred - cx
        vy_pred = cy_pred - cy
        va_pred = aspect_pred - aspect_ratio
        vh_pred = h_pred - h
        
        self.n_trackssm_used += 1
        
        return np.array([cx_pred, cy_pred, aspect_pred, h_pred, vx_pred, vy_pred, va_pred, vh_pred])
    
    def update(self, mean, covariance, measurement, track_id=None):
        """Update track with new detection."""
        cx, cy, aspect_ratio, h = measurement
        w = aspect_ratio * h
        
        # Update history
        if track_id is not None and track_id in self.track_histories:
            self.track_histories[track_id].append([cx, cy, w, h])
        
        # Update mean with measurement
        vx = cx - mean[0]
        vy = cy - mean[1]
        va = aspect_ratio - mean[2]
        vh = h - mean[3]
        
        mean = np.array([cx, cy, aspect_ratio, h, vx, vy, va, vh])
        
        return mean, covariance
    
    def print_stats(self):
        """Print usage statistics."""
        print(f"\n📊 Hybrid Motion Predictor Statistics:")
        print(f"   Total predictions: {self.n_predict_calls}")
        print(f"   Hybrid (α·TrackSSM + β·Kalman): {self.n_hybrid} ({100*self.n_hybrid/max(self.n_predict_calls,1):.1f}%)")
        print(f"   Kalman only: {self.n_kalman_only} ({100*self.n_kalman_only/max(self.n_predict_calls,1):.1f}%)")
        print(f"   TrackSSM calls: {self.n_trackssm_used}")

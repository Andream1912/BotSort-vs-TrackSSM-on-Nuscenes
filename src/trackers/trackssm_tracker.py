"""
Complete TrackSSM Tracker with Detection and Association

Architecture:
1. YOLOX detector → raw detections (bbox + confidence + class)
2. TrackSSM motion predictor → predicted track positions
3. IoU matching → associate detections to tracks
4. Track management → create/update/delete tracks

This is a REAL tracker that doesn't use GT information.
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import copy


class Track:
    """Single track with history and state"""
    
    def __init__(self, track_id, bbox, class_id, frame_id, history_len=5, min_hits=3):
        """
        Args:
            track_id: Unique track ID
            bbox: [x, y, w, h] in image coordinates
            class_id: Object class
            frame_id: Frame where track was created
            history_len: Number of history frames to keep
            min_hits: Minimum hits before track is activated
        """
        self.track_id = track_id
        self.class_id = class_id
        self.history_len = history_len
        self.min_hits = min_hits
        
        # Track state
        self.bbox = bbox  # Current bbox [x, y, w, h]
        self.history = [bbox]  # History of bboxes
        self.frame_ids = [frame_id]  # Corresponding frame IDs
        
        # Track metadata
        self.age = 1  # Number of frames since creation
        self.hits = 1  # Number of matched detections
        self.time_since_update = 0  # Frames since last detection match
        self.confidence = 1.0
        
        # State flags
        self.is_activated = False  # Track confirmed after N hits
        self.is_lost = False
    
    def update(self, bbox, frame_id, confidence=1.0):
        """Update track with new detection"""
        self.bbox = bbox
        self.history.append(bbox)
        self.frame_ids.append(frame_id)
        
        # Keep only recent history
        if len(self.history) > self.history_len + 1:  # +1 for current
            self.history = self.history[-(self.history_len + 1):]
            self.frame_ids = self.frame_ids[-(self.history_len + 1):]
        
        self.hits += 1
        self.time_since_update = 0
        self.confidence = confidence
        
        # Activate track after enough hits
        if self.hits >= self.min_hits:
            self.is_activated = True
    
    def predict(self, model, device, img_width=1600, img_height=900):
        """
        Predict next bbox position using TrackSSM.
        
        Args:
            model: TrackSSM model
            device: torch device
            img_width: Image width for normalization
            img_height: Image height for normalization
        
        Returns:
            predicted_bbox: [x, y, w, h] predicted position
        """
        self.age += 1
        self.time_since_update += 1
        
        # Need at least 1 frame for prediction
        if len(self.history) < 1:
            return self.bbox
        
        # If we have less than history_len frames, pad with first frame
        history_bboxes = []
        if len(self.history) < self.history_len:
            # Pad with first bbox
            padding = [self.history[0]] * (self.history_len - len(self.history))
            history_bboxes = padding + list(self.history)
        else:
            history_bboxes = list(self.history[-self.history_len:])
        
        # Prepare condition tensor: (history_len, 8)
        # Format: [x, y, w, h, class_id, conf, -1, -1] normalized
        condition = []
        for bbox in history_bboxes:
            x, y, w, h = bbox
            # Normalize to [0, 1]
            x_norm = x / img_width
            y_norm = y / img_height
            w_norm = w / img_width
            h_norm = h / img_height
            
            # Convert to cxcywh
            cx = x_norm + w_norm / 2
            cy = y_norm + h_norm / 2
            
            condition.append([cx, cy, w_norm, h_norm, self.class_id, 1.0, -1, -1])
        
        condition = np.array(condition, dtype=np.float32)  # (5, 8)
        
        # Run prediction
        with torch.no_grad():
            condition_tensor = torch.from_numpy(condition).unsqueeze(0).to(device)  # (1, 5, 8)
            
            # Encode
            cond_encoded = model.encoder(condition_tensor)  # (1, 1, 256)
            
            # Decode - use last frame bbox as x_0
            last_bbox_norm = condition_tensor[0, -1, :4]  # (4,)
            
            pred_bbox_norm = model.ssm_decoder(
                last_bbox_norm.unsqueeze(0),  # (1, 4)
                cond_encoded,
                h=None
            )  # (1, 4)
            
            pred_bbox_norm = pred_bbox_norm.squeeze(0).cpu().numpy()  # (4,)
        
        # Denormalize from cxcywh to xyxy to xywh
        cx, cy, w_norm, h_norm = pred_bbox_norm
        
        w_pred = w_norm * img_width
        h_pred = h_norm * img_height
        x_pred = (cx * img_width) - w_pred / 2
        y_pred = (cy * img_height) - h_pred / 2
        
        predicted_bbox = [x_pred, y_pred, w_pred, h_pred]
        
        # Update bbox with prediction
        self.bbox = predicted_bbox
        
        return predicted_bbox
    
    def mark_lost(self):
        """Mark track as lost"""
        self.is_lost = True
    
    def mark_removed(self):
        """Mark track for removal"""
        self.is_lost = True
    
    def get_state(self):
        """Get current track state for output"""
        return {
            'track_id': self.track_id,
            'bbox': self.bbox,
            'class_id': self.class_id,
            'confidence': self.confidence,
            'is_activated': self.is_activated
        }


class TrackSSMTracker:
    """
    Complete tracker using TrackSSM for motion prediction
    """
    
    def __init__(
        self,
        model,
        device,
        img_width=1600,
        img_height=900,
        track_thresh=0.2,       # Min confidence for track activation (balanced)
        match_thresh=0.5,       # IoU threshold for matching
        max_age=30,             # Max frames to keep lost track
        min_hits=3,             # Min hits to activate track
        history_len=5
    ):
        """
        Args:
            model: TrackSSM model for motion prediction
            device: torch device
            img_width: Image width
            img_height: Image height
            track_thresh: Confidence threshold for starting tracks
            match_thresh: IoU threshold for detection-track matching
            max_age: Maximum frames to keep a lost track
            min_hits: Minimum hits before track is activated
            history_len: History length for TrackSSM
        """
        self.model = model
        self.device = device
        self.img_width = img_width
        self.img_height = img_height
        
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.history_len = history_len
        
        # Track management
        self.tracked_tracks = []  # Active tracks
        self.lost_tracks = []     # Recently lost tracks
        self.removed_tracks = []  # Removed tracks
        
        self.frame_id = 0
        self.track_id_counter = 0
    
    def reset(self):
        """Reset tracker state"""
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0
        self.track_id_counter = 0
    
    def update(self, detections):
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections, each is dict with:
                        {'bbox': [x, y, w, h], 'confidence': float, 'class_id': int}
        
        Returns:
            output_tracks: List of active tracks with their current state
        """
        self.frame_id += 1
        
        # Separate high and low confidence detections
        high_conf_dets = [d for d in detections if d['confidence'] >= self.track_thresh]
        low_conf_dets = [d for d in detections if d['confidence'] < self.track_thresh]
        
        # Predict new positions for all tracks
        for track in self.tracked_tracks:
            track.predict(self.model, self.device, self.img_width, self.img_height)
        
        # First matching: tracked tracks vs high confidence detections
        matched, unmatched_tracks, unmatched_dets = self._match(
            self.tracked_tracks, high_conf_dets, self.match_thresh
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched:
            track = self.tracked_tracks[track_idx]
            det = high_conf_dets[det_idx]
            track.update(det['bbox'], self.frame_id, det['confidence'])
        
        # Handle unmatched tracks
        unmatched_tracked = [self.tracked_tracks[i] for i in unmatched_tracks]
        
        # Second matching: unmatched tracked tracks vs low confidence detections
        matched_low, unmatched_tracks_low, _ = self._match(
            unmatched_tracked, low_conf_dets, self.match_thresh
        )
        
        # Update tracks matched with low confidence detections
        for track_idx, det_idx in matched_low:
            track = unmatched_tracked[track_idx]
            det = low_conf_dets[det_idx]
            track.update(det['bbox'], self.frame_id, det['confidence'])
        
        # Mark still unmatched tracks as lost
        for i in unmatched_tracks_low:
            track = unmatched_tracked[i]
            track.mark_lost()
            if track not in self.lost_tracks:
                self.lost_tracks.append(track)
        
        # Remove unmatched tracks from active tracks
        self.tracked_tracks = [t for t in self.tracked_tracks if not t.is_lost]
        
        # Create new tracks from unmatched high-confidence detections
        for det_idx in unmatched_dets:
            det = high_conf_dets[det_idx]
            new_track = Track(
                track_id=self._get_next_id(),
                bbox=det['bbox'],
                class_id=det['class_id'],
                frame_id=self.frame_id,
                history_len=self.history_len,
                min_hits=self.min_hits
            )
            self.tracked_tracks.append(new_track)
        
        # Try to re-identify lost tracks with remaining unmatched detections
        # (this handles occlusions)
        if len(self.lost_tracks) > 0 and len(low_conf_dets) > 0:
            matched_lost, unmatched_lost_tracks, _ = self._match(
                self.lost_tracks, low_conf_dets, self.match_thresh * 0.8  # Lower threshold
            )
            
            for track_idx, det_idx in matched_lost:
                track = self.lost_tracks[track_idx]
                det = low_conf_dets[det_idx]
                track.update(det['bbox'], self.frame_id, det['confidence'])
                track.is_lost = False
                # Fix: Prevent duplicate tracks in tracked_tracks
                if track not in self.tracked_tracks:
                    self.tracked_tracks.append(track)
            
            # Update lost tracks
            self.lost_tracks = [self.lost_tracks[i] for i in unmatched_lost_tracks]
        
        # Remove old lost tracks
        self.lost_tracks = [t for t in self.lost_tracks if t.time_since_update <= self.max_age]
        
        # Return only activated tracks
        output_tracks = []
        for track in self.tracked_tracks:
            if track.is_activated:
                output_tracks.append(track.get_state())
        
        return output_tracks
    
    def _match(self, tracks, detections, threshold):
        """
        Match tracks to detections using IoU distance and Hungarian algorithm.
        
        Returns:
            matched: List of (track_idx, detection_idx) pairs
            unmatched_tracks: List of unmatched track indices
            unmatched_dets: List of unmatched detection indices
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._compute_iou(track.bbox, det['bbox'])
        
        # Convert IoU to cost (1 - IoU)
        cost_matrix = 1 - iou_matrix
        
        # Apply threshold: IoU < threshold → cost = inf (invalid match)
        cost_matrix[iou_matrix < threshold] = 1e6
        
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Filter matches by threshold
        matched = []
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 1e6:  # Valid match
                matched.append((i, j))
        
        # Get unmatched
        unmatched_tracks = [i for i in range(len(tracks)) if i not in row_ind or cost_matrix[row_ind.tolist().index(i), col_ind[row_ind.tolist().index(i)]] >= 1e6]
        unmatched_dets = [j for j in range(len(detections)) if j not in col_ind or cost_matrix[row_ind[col_ind.tolist().index(j)], j] >= 1e6]
        
        return matched, unmatched_tracks, unmatched_dets
    
    def _compute_iou(self, bbox1, bbox2):
        """Compute IoU between two bboxes [x, y, w, h]"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        x1_max, y1_max = x1 + w1, y1 + h1
        x2_max, y2_max = x2 + w2, y2 + h2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1_max, x2_max)
        yi2 = min(y1_max, y2_max)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _get_next_id(self):
        """Get next unique track ID"""
        self.track_id_counter += 1
        return self.track_id_counter

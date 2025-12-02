"""
BoT-SORT with TrackSSM Motion Predictor

This is BoT-SORT but with TrackSSM replacing the Kalman Filter for motion prediction.
"""
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
from src.trackers.trackssm_motion import TrackSSMMotion

# Import BoT-SORT
from tracker.mc_bot_sort import BoTSORT as BoTSORT_Multi


class TrackSSMWrapper:
    """
    Wrapper for TrackSSMMotion that handles optional track_id parameter.
    BoT-SORT's KalmanFilter interface doesn't include track_id, so we wrap it.
    """
    def __init__(self, trackssm_motion):
        self.motion = trackssm_motion
    
    def initiate(self, measurement, track_id=None, class_id=1, confidence=1.0):
        return self.motion.initiate(measurement, track_id=track_id, class_id=class_id, confidence=confidence)
    
    def predict(self, mean, covariance, track_id=None):
        pred_mean, pred_cov = self.motion.predict(mean, covariance, track_id=track_id)
        
        # DEBUG: Log what we return to BoT-SORT
        if not hasattr(self, '_debug_predict_return'):
            self._debug_predict_return = 0
        if self._debug_predict_return < 3:
            print(f"      🔙 predict() returns: mean[:4]={pred_mean[:4]}")
            self._debug_predict_return += 1
        
        return pred_mean, pred_cov
    
    def multi_predict(self, multi_mean, multi_covariance, track_ids=None):
        """Batch prediction for multiple tracks"""
        try:
            if track_ids is None:
                track_ids = [None] * len(multi_mean)
            
            if not hasattr(self, '_multi_predict_count'):
                self._multi_predict_count = 0
            self._multi_predict_count += 1
            
            if self._multi_predict_count <= 3:
                print(f"🔥 TrackSSMWrapper.multi_predict() #{self._multi_predict_count} | {len(multi_mean)} tracks | track_ids: {track_ids[:5]}")
            
            results_mean = []
            results_cov = []
            for mean, cov, track_id in zip(multi_mean, multi_covariance, track_ids):
                pred_mean, pred_cov = self.motion.predict(mean, cov, track_id=track_id)
                results_mean.append(pred_mean)
                results_cov.append(pred_cov)
            
            return np.array(results_mean), np.array(results_cov)
        except Exception as e:
            print(f"❌ TrackSSMWrapper.multi_predict() FAILED: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def update(self, mean, covariance, measurement, track_id=None, class_id=None, confidence=None):
        updated_mean, updated_cov = self.motion.update(mean, covariance, measurement, track_id=track_id, class_id=class_id, confidence=confidence)
        
        # DEBUG: Log when update is called (matching succeeded!)
        if not hasattr(self, '_debug_update_count'):
            self._debug_update_count = 0
        self._debug_update_count += 1
        if self._debug_update_count <= 5:
            print(f"      ✅ update() called #{self._debug_update_count} track_id={track_id} - MATCH FOUND!")
        
        return updated_mean, updated_cov


class BoTSORTTrackSSM(BaseTracker):
    """
    BoT-SORT with TrackSSM motion predictor instead of Kalman Filter.
    
    This combines the best of both:
    - BoT-SORT: Robust tracking framework with ReID, CMC, and association logic
    - TrackSSM: Neural motion prediction instead of Kalman Filter
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Multi-class support
        self.multi_class = True  # Always use multi-class for NuScenes
        
        # Frame rate
        self.frame_rate = config.get('frame_rate', 12)  # nuScenes is 12Hz
        
        # Load TrackSSM model
        import torch
        
        device = config.get('device', 'cuda')
        trackssm_weights = config.get('trackssm_weights', 'weights/trackssm/phase2/phase2_full_best.pth')
        
        print(f"🔍 DEBUG: Loading TrackSSM from {trackssm_weights}...")
        self.trackssm_model = self._load_trackssm_model(trackssm_weights, device)
        print("✓ TrackSSM model loaded and ready")
        
        # Create TrackSSM motion predictor
        img_width = config.get('img_width', 1600)
        img_height = config.get('img_height', 900)
        
        self.trackssm_motion = TrackSSMMotion(
            model=self.trackssm_model,
            device=device,
            img_width=img_width,
            img_height=img_height,
            history_len=5
        )
        
        # Create args namespace for BoTSORT
        args = Namespace()
        args.track_high_thresh = config.get('track_high_thresh', 0.6)
        args.track_low_thresh = config.get('track_low_thresh', 0.1)
        args.new_track_thresh = config.get('new_track_thresh', 0.5)
        args.track_buffer = config.get('track_buffer', 30)
        args.match_thresh = config.get('match_thresh', 0.8)
        args.proximity_thresh = config.get('proximity_thresh', 0.5)
        args.appearance_thresh = config.get('appearance_thresh', 0.25)
        args.cmc_method = config.get('cmc_method', 'sparseOptFlow')
        args.with_reid = config.get('with_reid', True)
        args.fast_reid_config = config.get('fast_reid_config', 'external/BoT-SORT/fast_reid/configs/MOT17/sbs_S50.yml')
        args.fast_reid_weights = config.get('fast_reid_weights', 'weights/reid/mot17_sbs_S50.pth')
        args.device = device
        args.name = config.get('name', 'botsort_trackssm')
        args.ablation = config.get('ablation', '')
        args.mot20 = False
        
        self.args = args
        
        # Monkey-patch STrack methods BEFORE creating BoT-SORT instance
        # This ensures all STrack instances will use the patched methods
        self._patch_strack_methods()
        
        # Initialize BoT-SORT with multi-class support
        print(f"Initializing BoT-SORT+TrackSSM with ReID={args.with_reid}, CMC={args.cmc_method}")
        self.tracker = BoTSORT_Multi(args, frame_rate=self.frame_rate)
        
        # CRITICAL FIX: Replace BOTH kalman_filter AND STrack.shared_kalman
        # BoT-SORT uses STrack.shared_kalman (static), not self.tracker.kalman_filter!
        from tracker.mc_bot_sort import STrack
        
        trackssm_wrapper = TrackSSMWrapper(self.trackssm_motion)
        self.tracker.kalman_filter = trackssm_wrapper
        STrack.shared_kalman = trackssm_wrapper  # THIS IS THE KEY FIX!
        
        print("✓ Replaced Kalman Filter with TrackSSM motion predictor (with track_id wrapper)")
        print(f"🔍 DEBUG: BoT-SORT kalman_filter type: {type(self.tracker.kalman_filter)}")
        print(f"🔍 DEBUG: STrack.shared_kalman type: {type(STrack.shared_kalman)}")
        print(f"🔍 DEBUG: Is TrackSSMWrapper? {isinstance(self.tracker.kalman_filter, TrackSSMWrapper)}")
    
    def _patch_strack_methods(self):
        """
        Patch STrack methods to pass track_id to motion predictor.
        This is necessary because BoT-SORT's STrack doesn't pass track_id to KalmanFilter.
        """
        from tracker.mc_bot_sort import STrack
        
        # Store original methods
        original_predict = STrack.predict
        original_activate = STrack.activate
        original_re_activate = STrack.re_activate
        original_update = STrack.update
        
        # Create reference to trackssm_motion for closures
        trackssm_motion = self.trackssm_motion
        
        def patched_predict(self):
            """Patched predict that passes track_id"""
            # DEBUG: Log first call
            if not hasattr(STrack, '_predict_debug_logged'):
                STrack._predict_debug_logged = True
                print(f"🔍 DEBUG: patched_predict() called! track_id={self.track_id if hasattr(self, 'track_id') else 'N/A'}")
            
            mean_state = self.mean.copy()
            if self.state != 1:  # TrackState.Tracked
                mean_state[6] = 0
                mean_state[7] = 0
            # FIXED: Pass track_id
            self.mean, self.covariance = self.kalman_filter.predict(
                mean_state, self.covariance, track_id=self.track_id
            )
        
        def patched_activate(self, kalman_filter, frame_id):
            """Patched activate that passes track_id, class_id, and confidence"""
            self.kalman_filter = kalman_filter
            self.track_id = self.next_id()
            # FIXED: Pass track_id, class_id, and confidence to TrackSSM
            self.mean, self.covariance = self.kalman_filter.initiate(
                self.tlwh_to_xywh(self._tlwh), 
                track_id=self.track_id,
                class_id=int(self.cls),  # Force int
                confidence=float(self.score)
            )
            self.tracklet_len = 0
            self.state = 1  # TrackState.Tracked
            if frame_id == 1:
                self.is_activated = True
            self.frame_id = frame_id
            self.start_frame = frame_id
        
        def patched_re_activate(self, new_track, frame_id, new_id=False):
            """Patched re_activate that passes track_id, class_id, and confidence"""
            # FIXED: Pass track_id, class_id, and confidence
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh), 
                track_id=self.track_id,
                class_id=int(new_track.cls),  # Force int
                confidence=float(new_track.score)
            )
            if new_track.curr_feat is not None:
                self.update_features(new_track.curr_feat)
            self.tracklet_len = 0
            self.state = 1  # TrackState.Tracked
            self.is_activated = True
            self.frame_id = frame_id
            if new_id:
                self.track_id = self.next_id()
        
        def patched_update(self, new_track, frame_id):
            """Patched update that passes track_id, class_id, and confidence"""
            self.frame_id = frame_id
            self.tracklet_len += 1

            new_tlwh = new_track.tlwh

            # FIXED: Pass track_id, class_id, and confidence
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh),
                track_id=self.track_id,
                class_id=int(new_track.cls),  # Force int
                confidence=float(new_track.score)
            )

            if new_track.curr_feat is not None:
                self.update_features(new_track.curr_feat)

            self.state = 1  # TrackState.Tracked
            self.is_activated = True
            self.score = new_track.score
        
        def patched_multi_predict(stracks):
            """Patched multi_predict that passes track_ids to TrackSSMWrapper.multi_predict"""
            if len(stracks) == 0:
                return
            
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            track_ids = [st.track_id for st in stracks]
            
            # Zero out velocities for non-tracked states (same as original)
            for i, st in enumerate(stracks):
                if st.state != 1:  # TrackState.Tracked
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            
            # KEY FIX: Pass track_ids to multi_predict
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance, track_ids=track_ids
            )
            
            # Write back to tracks
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov
        
        # Apply patches
        STrack.predict = patched_predict
        STrack.activate = patched_activate
        STrack.re_activate = patched_re_activate
        STrack.update = patched_update
        STrack.multi_predict = patched_multi_predict  # 🔥 CRITICAL FIX!
        
        print("✓ Patched STrack methods to pass track_id to TrackSSM (including multi_predict)")
    
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
            if frame is not None:
                empty_dets = np.empty((0, 8))
                online_targets = self.tracker.update(empty_dets, frame)
            else:
                return []
        else:
            # Convert detections to BoT-SORT format
            dets = []
            for det in detections:
                x, y, w, h = det['bbox']
                x1, y1, x2, y2 = x, y, x + w, y + h
                conf = det['confidence']
                cls_id = det['class_id']
                
                # BoT-SORT format: [x1, y1, x2, y2, conf, class] + dummy features
                dets.append([x1, y1, x2, y2, conf, cls_id, 0, 0])
            
            dets = np.array(dets)
            
            # Update tracker
            if frame is not None:
                online_targets = self.tracker.update(dets, frame)
            else:
                # No frame - use dummy image
                dummy_frame = np.zeros((900, 1600, 3), dtype=np.uint8)
                online_targets = self.tracker.update(dets, dummy_frame)
        
        # Convert output to our format
        tracks = []
        for t in online_targets:
            # BoT-SORT output: tlwh format
            x1, y1, w, h = t.tlwh
            track_dict = {
                'track_id': int(t.track_id),
                'bbox': [float(x1), float(y1), float(w), float(h)],
                'class_id': int(t.cls),
                'confidence': float(t.score),
                'is_activated': True
            }
            tracks.append(track_dict)
        
        return tracks
    
    def reset(self):
        """Reset tracker state"""
        self.frame_id = 0
        # Reset BoT-SORT
        self.tracker = BoTSORT_Multi(self.args, frame_rate=self.frame_rate)
        self.tracker.kalman_filter = TrackSSMWrapper(self.trackssm_motion)
        # Re-apply patches
        self._patch_strack_methods()
        # Clear TrackSSM history
        self.trackssm_motion.track_histories.clear()
    
    def _load_trackssm_model(self, checkpoint_path: str, device: str):
        """
        Load TrackSSM model (D2MP) from checkpoint.
        
        Args:
            checkpoint_path: Path to phase2_full_best.pth
            device: 'cuda' or 'cpu'
            
        Returns:
            D2MP model ready for inference
        """
        import torch
        from types import SimpleNamespace
        
        # Add models to path
        sys.path.insert(0, os.path.join(project_root, 'models'))
        from models.mamba_encoder import Mamba_encoder
        from models.autoencoder import D2MP
        from models.condition_embedding import Time_info_aggregation
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Detect checkpoint format
        if 'config' in checkpoint:
            # Phase2 format (NuScenes fine-tuned)
            config_dict = checkpoint['config']
            model_state_dict = checkpoint['model_state_dict']
        elif 'ddpm' in checkpoint:
            # Phase1 format (MOT17 pretrained)
            # Extract config from ddpm state - MOT17 uses 256 dim encoder/decoder
            config_dict = {
                'encoder_dim': 256,  # MOT17 model dim
                'n_layer': 6,  # 6 encoder layers (0-5)
                'decoder_dim': 256,  # Same as encoder
                'decoder_n_layer': 8,  # Decoder layers
                'seq_len': 5,
                'dt_rank': 16,  # From dt_proj weight shape
                'dt_scale': 1.0,
                'd_state': 16,
                'd_conv': 4,
                'expand_factor': 2,  # 256 -> 512 expansion
                'dt_min': 0.001,
                'dt_max': 0.1,
                'dt_init': 'random',
                'dt_init_floor': 1e-4,
                'bias': False,
                'conv_bias': True,
                'pscan': True,
                'time_steps': 1000,
                'beta_schedule': 'linear',
                'beta_start': 0.0001,
                'beta_end': 0.02,
                'objective': 'pred_noise',
                'loss_type': 'l2',
                'predict_epsilon': True,
                'use_diffmot': False,  # MOT17 uses SSM decoder
                'diffnet': None,
                'tf_layer': 4
            }
            # Remove 'module.' prefix from MOT17 checkpoint keys
            model_state_dict = {}
            for key, value in checkpoint['ddpm'].items():
                new_key = key.replace('module.', '')
                model_state_dict[new_key] = value
        else:
            raise ValueError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")
        
        # Create config namespace
        config = SimpleNamespace(**config_dict)
        
        # Create encoder using Time_info_aggregation (which contains mamba_net)
        # NOTE: vocab_size in config is 6400, but the actual embedding is 8 (bbox features)
        encoder = Time_info_aggregation(
            d_model=config.encoder_dim,
            n_layer=config.n_layer,
            v_size=8  # Always 8 for bbox features [cx, cy, w, h, dx, dy, dw, dh]
        )
        
        # Create D2MP model
        model = D2MP(config=config, encoder=encoder, device=device)
        
        # Load weights
        incompatible = model.load_state_dict(model_state_dict, strict=False)
        print(f"  ✓ Loaded checkpoint weights (missing: {len(incompatible.missing_keys)}, unexpected: {len(incompatible.unexpected_keys)})")
        if len(incompatible.missing_keys) > 5:
            print(f"    ⚠️  Missing keys (first 5): {incompatible.missing_keys[:5]}")
        if len(incompatible.unexpected_keys) > 5:
            print(f"    ⚠️  Unexpected keys (first 5): {incompatible.unexpected_keys[:5]}")
        
        model.to(device)
        model.eval()
        
        # DEBUG: Verify checkpoint loaded correctly by checking a weight
        first_param = next(model.parameters())
        print(f"  🔍 First param mean: {first_param.mean().item():.6f}, std: {first_param.std().item():.6f}")
        
        return model
        model.to(device)
        model.eval()
        
        return model

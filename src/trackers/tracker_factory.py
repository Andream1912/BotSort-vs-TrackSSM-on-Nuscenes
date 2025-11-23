"""Tracker Factory - Creates tracker instances"""
import sys
import os
import torch
from typing import Dict, Any

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.trackers.base_tracker import BaseTracker


class TrackerFactory:
    """Factory for creating tracker instances"""
    
    @staticmethod
    def create(tracker_type: str, config: Dict[str, Any]) -> BaseTracker:
        """
        Create a tracker instance.
        
        Args:
            tracker_type: One of 'trackssm', 'botsort'
            config: Configuration dictionary
        
        Returns:
            BaseTracker instance
        """
        tracker_type = tracker_type.lower()
        
        if tracker_type == 'trackssm':
            from src.trackers.trackssm_wrapper import TrackSSMTracker
            
            # Load TrackSSM model
            if 'model' not in config:
                config['model'] = TrackerFactory._load_trackssm_model(
                    config.get('checkpoint_path'),
                    config.get('device', 'cuda')
                )
            
            return TrackSSMTracker(config)
        
        elif tracker_type == 'botsort':
            from src.trackers.botsort_tracker import BotSortTracker
            return BotSortTracker(config)
        
        else:
            raise ValueError(f"Unknown tracker type: {tracker_type}. "
                           f"Supported: trackssm, botsort")
    
    @staticmethod
    def _load_trackssm_model(checkpoint_path: str, device: str):
        """Load TrackSSM model"""
        # Import models - add trackssm_reference to path
        project_root_local = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        trackssm_path = os.path.join(project_root_local, 'trackssm_reference')
        if trackssm_path not in sys.path:
            sys.path.insert(0, trackssm_path)
        
        from models.autoencoder import D2MP
        from models.condition_embedding import Time_info_aggregation
        
        print(f"Loading TrackSSM from {checkpoint_path}...")
        
        # Create config
        class Config:
            def __init__(self):
                self.encoder_dim = 256
                self.use_diffmot = False
        
        config = Config()
        
        # Initialize model
        encoder = Time_info_aggregation(d_model=256, n_layer=2, v_size=8)
        decoder_model = D2MP(config=config, encoder=encoder, device=device)
        
        # Create wrapper module
        model = torch.nn.Module()
        model.encoder = encoder
        model.ssm_decoder = decoder_model.ssm_decoder
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        # Extract state dict properly
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        
        # Load encoder
        encoder_state = {k.replace('encoder.', ''): v 
                        for k, v in state_dict.items() if 'encoder' in k}
        if encoder_state:
            encoder.load_state_dict(encoder_state, strict=False)
        
        # Load decoder
        decoder_state = {k.replace('ssm_decoder.', ''): v 
                        for k, v in state_dict.items() if 'ssm_decoder' in k}
        if decoder_state:
            model.ssm_decoder.load_state_dict(decoder_state, strict=False)
        
        model.to(device)
        model.eval()
        
        print("✓ Loaded TrackSSM model")
        
        return model

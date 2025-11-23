"""
NuScenes Track-level Dataset for TrackSSM
Converts MOT format sequences into individual track samples compatible with TrackSSM architecture.

Format:
- Input: MOT format gt.txt (frame_id, track_id, x, y, w, h, conf, class_id, vis)
- Output: {"condition": (5, 8), "cur_bbox": (4,), "cur_gt": (7,)} per track
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset


class NuScenesTrackDataset(Dataset):
    """
    Dataset that extracts individual tracks from NuScenes MOT format sequences.
    Each sample is a single track at a specific time with 5 history frames.
    
    Compatible with TrackSSM original architecture expecting (B, 5, 8) condition format.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        history_len: int = 5,
        min_track_len: int = 6,
        normalize: bool = True,
        img_width: int = 1600,
        img_height: int = 900,
        sample_stride: int = 1
    ):
        """
        Args:
            data_root: Root directory containing train/val/test splits
            split: 'train', 'val', or 'test'
            history_len: Number of history frames (default 5)
            min_track_len: Minimum track length to include (default 6 = 5 history + 1 current)
            normalize: Whether to normalize bbox coordinates
            img_width: Image width for normalization
            img_height: Image height for normalization
        """
        self.data_root = Path(data_root)
        self.split = split
        self.history_len = history_len
        self.min_track_len = min_track_len
        self.normalize = normalize
        self.img_width = img_width
        self.img_height = img_height
        self.sample_stride = sample_stride
        # Sliding window size: history + 1 current frame
        self.window_size = history_len + 1
        
        # Load all tracks from all sequences
        self.tracks = []  # List of (sequence_path, track_data, sequence_name)
        self.samples = []  # List of (track_idx, start_frame_idx)
        
        self._load_tracks()
        self._build_samples()
        
        print(f"NuScenesTrackDataset [{split}]:")
        print(f"  Sequences: {len(self.tracks)}")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  History length: {history_len}")
        print(f"  Window size: {self.window_size}")
    
    def _load_tracks(self):
        """Load all tracks from all sequences in the split"""
        split_dir = self.data_root / self.split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        sequences = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        
        for seq_path in sequences:
            gt_file = seq_path / 'gt' / 'gt.txt'
            
            if not gt_file.exists():
                print(f"Warning: gt.txt not found in {seq_path}")
                continue
            
            # Read all detections
            try:
                data = np.loadtxt(gt_file, delimiter=',', dtype=np.float32)
            except (ValueError, IndexError):
                # Empty file or malformed
                continue
            
            if len(data) == 0:
                continue
            
            # Handle single detection case (1D array)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            # Group by track_id
            unique_tracks = np.unique(data[:, 1]).astype(int)
            
            for track_id in unique_tracks:
                track_data = data[data[:, 1] == track_id]
                
                # Sort by frame_id
                track_data = track_data[track_data[:, 0].argsort()]
                
                # Filter tracks that are too short
                if len(track_data) < self.min_track_len:
                    continue
                
                self.tracks.append({
                    'sequence': seq_path.name,
                    'track_id': track_id,
                    'data': track_data,  # [frame_id, track_id, x, y, w, h, conf, class_id, vis]
                })
    
    def _build_samples(self):
        """Build sliding window samples from all tracks"""
        for track_idx, track_info in enumerate(self.tracks):
            track_data = track_info['data']
            track_len = len(track_data)
            
            # Create sliding windows
            # For a track of length L, we can create L - window_size + 1 samples
            num_windows = track_len - self.window_size + 1


            for start_idx in range(0, num_windows, self.sample_stride):
                self.samples.append((track_idx, start_idx))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a single track sample in TrackSSM format.
        
        Returns:
            {
                'condition': Tensor (5, 8) - history frames [bbox(4) + delta_bbox(4)]
                'cur_bbox': Tensor (4,) - current frame bbox to predict
                'cur_gt': Tensor (7,) - current frame ground truth [frame_id, x, y, w, h, class_id, vis]
                'delta_bbox': Tensor (4,) - delta between last history and current
            }
        """
        track_idx, start_idx = self.samples[idx]
        track_info = self.tracks[track_idx]
        track_data = track_info['data']
        
        # Extract window: [start_idx : start_idx + window_size]
        window_data = track_data[start_idx : start_idx + self.window_size]
        
        # window_data shape: (window_size, 9)
        # Columns: [frame_id, track_id, x, y, w, h, conf, class_id, vis]
        
        # Extract bboxes: columns 2:6 (x, y, w, h)
        bboxes = window_data[:, 2:6].copy()  # (window_size, 4)
        
        # Normalize if needed
        if self.normalize:
            # Convert to [cx, cy, w, h] normalized to [0, 1]
            bboxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2] / 2) / self.img_width   # cx
            bboxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3] / 2) / self.img_height  # cy
            bboxes[:, 2] = bboxes[:, 2] / self.img_width   # w
            bboxes[:, 3] = bboxes[:, 3] / self.img_height  # h
        
        # Split into history and current
        history_bboxes = bboxes[:-1]  # (5, 4) - first 5 frames
        cur_bbox = bboxes[-1]          # (4,) - last frame
        
        # Compute delta_bbox between consecutive frames in history
        delta_bboxes = np.diff(bboxes, axis=0)  # (5, 4) - differences between consecutive frames
        
        # Build condition: [history_bboxes[1:] | delta_bboxes]
        # We use frames 1-5 of history with their deltas
        condition = np.concatenate([
            history_bboxes[1:],  # frames 1,2,3,4,5 (skip frame 0)
            delta_bboxes[1:]     # deltas 1-2, 2-3, 3-4, 4-5, 5-current
        ], axis=1)  # (5, 8)
        
        # Wait, we need 5 frames history but we have deltas. Let me recalculate:
        # History frames: 0, 1, 2, 3, 4 (indices in window)
        # Current frame: 5 (index in window)
        # 
        # For TrackSSM original format:
        # condition should be: [bbox_1, bbox_2, bbox_3, bbox_4, bbox_5, delta_01, delta_12, delta_23, delta_34, delta_45]
        # That's (5, 8): 5 frames, each with [bbox(4) + delta(4)]
        
        # Let's rebuild properly:
        # bboxes[1:6] are frames 1-5 (5 history frames)
        # deltas[0:5] are delta_01, delta_12, delta_23, delta_34, delta_45
        
        history_bboxes = bboxes[1:]  # frames 1,2,3,4,5 (indices 1-5 in window)
        delta_bboxes_full = np.diff(bboxes, axis=0)  # shape (5, 4): delta_01, delta_12, delta_23, delta_34, delta_45
        
        # Condition: [bbox | delta] for each of 5 history frames
        condition = np.concatenate([history_bboxes, delta_bboxes_full], axis=1)  # (5, 8)
        
        # Current bbox (frame 5, which is index -1)
        cur_bbox = bboxes[-1]  # (4,)
        
        # Delta between last history (frame 4) and current (frame 5)
        delta_bbox = delta_bboxes_full[-1]  # (4,)
        
        # Current ground truth: [frame_id, x, y, w, h, class_id, vis]
        cur_gt_raw = window_data[-1]  # last frame
        cur_gt = np.array([
            cur_gt_raw[0],  # frame_id
            cur_bbox[0],    # cx (normalized)
            cur_bbox[1],    # cy (normalized)
            cur_bbox[2],    # w (normalized)
            cur_bbox[3],    # h (normalized)
            cur_gt_raw[7],  # class_id
            cur_gt_raw[8]   # visibility
        ], dtype=np.float32)
        
        return {
            'condition': torch.from_numpy(condition).float(),      # (5, 8)
            'cur_bbox': torch.from_numpy(cur_bbox).float(),        # (4,)
            'cur_gt': torch.from_numpy(cur_gt).float(),            # (7,)
            'delta_bbox': torch.from_numpy(delta_bbox).float()     # (4,)
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching track samples.
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        Batched dictionary with proper shapes for TrackSSM
    """
    return {
        'condition': torch.stack([item['condition'] for item in batch]),      # (B, 5, 8)
        'cur_bbox': torch.stack([item['cur_bbox'] for item in batch]),        # (B, 4)
        'cur_gt': torch.stack([item['cur_gt'] for item in batch]),            # (B, 7)
        'delta_bbox': torch.stack([item['delta_bbox'] for item in batch])     # (B, 4)
    }


if __name__ == "__main__":
    # Test the dataset
    print("Testing NuScenesTrackDataset...")
    
    data_root = "./data/nuscenes_mot_6cams_interpolated"
    
    # Test train split
    dataset = NuScenesTrackDataset(
        data_root=data_root,
        split='train',
        history_len=5
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test single sample
    sample = dataset[0]
    print("\nSample 0:")
    for key, value in sample.items():
        print(f"  {key}: {value.shape} - {value.dtype}")
    
    # Test batch collation
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))
    
    print("\nBatch:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")
    
    print("\n✅ Test passed!")

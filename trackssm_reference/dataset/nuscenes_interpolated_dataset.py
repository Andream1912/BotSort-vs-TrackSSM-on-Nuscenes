#!/usr/bin/env python3
"""
Dataset loader per NuScenes interpolato per fine-tuning TrackSSM.

Carica sequenze multi-camera interpolate in formato MOT:
- frame_id, track_id, x, y, w, h, conf, class_id, visibility, -1
- Gestisce 6 camere come sequenze separate
- Supporta 7 classi veicoli/pedoni
- Frame rate 12fps (interpolato da 2Hz)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import random


class NuScenesInterpolatedDataset(Dataset):
    """
    Dataset per sequenze NuScenes interpolate in formato MOT.
    
    Struttura directory:
        data_root/
            train/
                scene-0001-CAM_FRONT/
                    gt/
                        gt.txt  # MOT format
                    seqinfo.ini
                scene-0001-CAM_FRONT_LEFT/
                    ...
            val/
                ...
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        sequence_length: int = 20,
        num_queries: int = 300,
        augment: bool = True,
        min_track_len: int = 3
    ):
        """
        Args:
            data_root: Root directory del dataset
            split: 'train', 'val', o 'test'
            sequence_length: Numero di frame per sequenza
            num_queries: Numero massimo di track per sample
            augment: Applica data augmentation
            min_track_len: Lunghezza minima track (per training)
        """
        self.data_root = Path(data_root) / split
        self.split = split
        self.sequence_length = sequence_length
        self.num_queries = num_queries
        self.augment = augment and (split == 'train')
        self.min_track_len = min_track_len
        
        # Trova tutte le sequenze (scene-camera)
        self.sequences = self._discover_sequences()
        
        # Carica annotazioni per ogni sequenza
        self.sequence_data = {}
        for seq_name in self.sequences:
            self.sequence_data[seq_name] = self._load_sequence(seq_name)
        
        # Crea samples (sequenze di frame consecutivi)
        self.samples = self._create_samples()
        
        print(f"NuScenes {split} dataset: {len(self.sequences)} sequences, {len(self.samples)} samples")
    
    def _discover_sequences(self) -> List[str]:
        """Trova tutte le scene-camera nel split."""
        sequences = []
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")
        
        for seq_dir in sorted(self.data_root.iterdir()):
            if seq_dir.is_dir() and (seq_dir / 'gt' / 'gt.txt').exists():
                sequences.append(seq_dir.name)
        
        return sequences
    
    def _load_sequence(self, seq_name: str) -> Dict:
        """
        Carica annotazioni MOT per una sequenza.
        
        Returns:
            {
                'frames': {frame_id: [detection_dict, ...]},
                'tracks': {track_id: [frame_id1, frame_id2, ...]},
                'num_frames': int,
                'img_width': int,
                'img_height': int
            }
        """
        gt_file = self.data_root / seq_name / 'gt' / 'gt.txt'
        seqinfo_file = self.data_root / seq_name / 'seqinfo.ini'
        
        # Read image dimensions from seqinfo.ini
        img_width, img_height = 1600, 900  # defaults
        if seqinfo_file.exists():
            with open(seqinfo_file, 'r') as f:
                for line in f:
                    if line.startswith('imWidth='):
                        img_width = int(line.strip().split('=')[1])
                    elif line.startswith('imHeight='):
                        img_height = int(line.strip().split('=')[1])
        
        # Parse MOT format
        frame_detections = defaultdict(list)
        track_frames = defaultdict(list)
        
        with open(gt_file, 'r') as f:
            for line in f:
                # Format: frame_id, track_id, x, y, w, h, conf, class_id, visibility, -1
                parts = line.strip().split(',')
                if len(parts) < 10:
                    continue
                
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                conf = float(parts[6])
                class_id = int(parts[7])
                visibility = float(parts[8])
                
                detection = {
                    'track_id': track_id,
                    'bbox': np.array([x, y, w, h], dtype=np.float32),
                    'class_id': class_id,
                    'visibility': visibility,
                    'conf': conf
                }
                
                frame_detections[frame_id].append(detection)
                track_frames[track_id].append(frame_id)
        
        # Filtra track troppo corti
        valid_tracks = {tid: fids for tid, fids in track_frames.items() 
                       if len(fids) >= self.min_track_len}
        
        # Rimuovi detection di track non validi
        for frame_id in list(frame_detections.keys()):
            frame_detections[frame_id] = [det for det in frame_detections[frame_id] 
                                         if det['track_id'] in valid_tracks]
        
        num_frames = max(frame_detections.keys()) if frame_detections else 0
        
        return {
            'frames': dict(frame_detections),
            'tracks': dict(valid_tracks),
            'num_frames': num_frames,
            'img_width': img_width,
            'img_height': img_height
        }
    
    def _create_samples(self) -> List[Tuple[str, int]]:
        """
        Crea samples (seq_name, start_frame) per training.
        Un sample è una finestra di sequence_length frame consecutivi.
        """
        samples = []
        
        for seq_name in self.sequences:
            seq_data = self.sequence_data[seq_name]
            num_frames = seq_data['num_frames']
            
            # Sliding window con stride
            stride = self.sequence_length // 2 if self.split == 'train' else self.sequence_length
            
            for start_frame in range(1, num_frames - self.sequence_length + 2, stride):
                # Verifica che ci siano abbastanza frame e bbox
                frame_range = range(start_frame, start_frame + self.sequence_length)
                total_boxes = sum(len(seq_data['frames'].get(f, [])) for f in frame_range)
                
                if total_boxes >= self.min_track_len:  # Almeno qualche bbox
                    samples.append((seq_name, start_frame))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Restituisce un sample per training.
        
        Returns:
            {
                'condition': (T, N, 4),  # T=seq_len, N=num_queries, format=cxcywh normalized
                'cur_bbox': (N, 4),       # Last frame bbox for prediction
                'track_ids': (T, N),      # track_id per timestep
                'class_ids': (T, N),      # class_id per timestep
                'padding_mask': (T, N),   # True=padding, False=valid bbox
                'seq_name': str,
                'start_frame': int
            }
        """
        seq_name, start_frame = self.samples[idx]
        seq_data = self.sequence_data[seq_name]
        
        # Prendi sequence_length frame
        frame_ids = list(range(start_frame, start_frame + self.sequence_length))
        
        # Raccogli tutti i track_id nella sequenza
        all_track_ids = set()
        for fid in frame_ids:
            for det in seq_data['frames'].get(fid, []):
                all_track_ids.add(det['track_id'])
        
        # Limita a num_queries track (sample random se troppi)
        all_track_ids = list(all_track_ids)
        if len(all_track_ids) > self.num_queries:
            all_track_ids = random.sample(all_track_ids, self.num_queries)
        
        # Crea mapping track_id -> query_idx
        track_to_idx = {tid: i for i, tid in enumerate(all_track_ids)}
        num_tracks = len(all_track_ids)
        
        # Inizializza tensori
        bboxes = np.zeros((self.sequence_length, self.num_queries, 4), dtype=np.float32)
        labels = np.zeros((self.sequence_length, self.num_queries), dtype=np.int64)
        classes = np.zeros((self.sequence_length, self.num_queries), dtype=np.int64)
        mask = np.zeros((self.sequence_length, self.num_queries), dtype=np.bool_)
        
        # Get image dimensions from sequence data
        img_w = seq_data['img_width']
        img_h = seq_data['img_height']
        
        # Riempi tensori
        for t, fid in enumerate(frame_ids):
            for det in seq_data['frames'].get(fid, []):
                track_id = det['track_id']
                if track_id not in track_to_idx:
                    continue
                
                idx = track_to_idx[track_id]
                
                # Normalizza bbox a formato cxcywh [0,1]
                x, y, w, h = det['bbox']
                
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                
                # Clip a [0, 1]
                cx = np.clip(cx, 0, 1)
                cy = np.clip(cy, 0, 1)
                w_norm = np.clip(w_norm, 0, 1)
                h_norm = np.clip(h_norm, 0, 1)
                
                bboxes[t, idx] = [cx, cy, w_norm, h_norm]
                labels[t, idx] = track_id
                classes[t, idx] = det['class_id']
                mask[t, idx] = True
        
        # Data augmentation (se training)
        if self.augment:
            bboxes = self._augment_bboxes(bboxes, mask)
        
        # Prepare model-compatible format
        return {
            'condition': torch.from_numpy(bboxes),  # (T, N, 4) history
            'cur_bbox': torch.from_numpy(bboxes[-1]),  # (N, 4) last frame for prediction
            'track_ids': torch.from_numpy(labels),
            'class_ids': torch.from_numpy(classes),
            'padding_mask': torch.from_numpy(~mask),  # Inverted: True=padding, False=valid
            'seq_name': seq_name,
            'start_frame': start_frame
        }
    
    def _augment_bboxes(self, bboxes: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Applica random augmentation alle bbox.
        - Random jitter (±5% su position/size)
        """
        if not self.augment:
            return bboxes
        
        # Jitter su bbox valide
        jitter_pos = 0.02  # ±2% su cx, cy
        jitter_size = 0.05  # ±5% su w, h
        
        for t in range(bboxes.shape[0]):
            for i in range(bboxes.shape[1]):
                if mask[t, i]:
                    # Jitter position
                    bboxes[t, i, 0] += np.random.uniform(-jitter_pos, jitter_pos)
                    bboxes[t, i, 1] += np.random.uniform(-jitter_pos, jitter_pos)
                    
                    # Jitter size
                    bboxes[t, i, 2] *= np.random.uniform(1 - jitter_size, 1 + jitter_size)
                    bboxes[t, i, 3] *= np.random.uniform(1 - jitter_size, 1 + jitter_size)
                    
                    # Clip a [0, 1]
                    bboxes[t, i] = np.clip(bboxes[t, i], 0, 1)
        
        return bboxes


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function per DataLoader.
    Stack batch mantenendo padding corretto.
    """
    condition = torch.stack([item['condition'] for item in batch])  # (B, T, N, 4)
    cur_bbox = torch.stack([item['cur_bbox'] for item in batch])  # (B, N, 4)
    track_ids = torch.stack([item['track_ids'] for item in batch])  # (B, T, N)
    class_ids = torch.stack([item['class_ids'] for item in batch])  # (B, T, N)
    padding_mask = torch.stack([item['padding_mask'] for item in batch])  # (B, T, N)
    
    return {
        'condition': condition,
        'cur_bbox': cur_bbox,
        'track_ids': track_ids,
        'class_ids': class_ids,
        'padding_mask': padding_mask,
        'seq_names': [item['seq_name'] for item in batch],
        'start_frames': [item['start_frame'] for item in batch]
    }


if __name__ == '__main__':
    # Test dataset
    dataset = NuScenesInterpolatedDataset(
        data_root='./data/nuscenes_test_interpolation',
        split='train',
        sequence_length=20,
        num_queries=50
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Sequences: {len(dataset.sequences)}")
    
    # Test sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Condition: {sample['condition'].shape}")
    print(f"  Cur bbox: {sample['cur_bbox'].shape}")
    print(f"  Track IDs: {sample['track_ids'].shape}")
    print(f"  Class IDs: {sample['class_ids'].shape}")
    print(f"  Padding mask: {sample['padding_mask'].shape}")
    print(f"  Valid boxes: {(~sample['padding_mask']).sum().item()}")
    print(f"  Sequence: {sample['seq_name']}")
    print(f"  Start frame: {sample['start_frame']}")
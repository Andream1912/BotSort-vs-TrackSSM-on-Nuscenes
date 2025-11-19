#!/usr/bin/env python3
"""
Compute per-class metrics for TrackSSM on NuScenes to compare with BotSort.
Requires ground truth with class annotations.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import motmetrics as mm

# NuScenes 7 classes mapping
CLASS_MAPPING = {
    1: 'car',
    2: 'truck', 
    3: 'bus',
    4: 'trailer',
    5: 'pedestrian',
    6: 'motorcycle',
    7: 'bicycle'
}

def load_mot_with_classes(filepath, is_tlbr=False):
    """Load MOT format file with class info (column 8).
    
    Args:
        filepath: Path to MOT format file
        is_tlbr: If True, assumes format is TLBR (x1,y1,x2,y2) and converts to TLWH
    """
    data = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    frame = int(parts[0])
                    track_id = int(parts[1])
                    
                    if is_tlbr:
                        # Format: frame,id,x1,y1,x2,y2,conf,...
                        x1, y1, x2, y2 = map(float, parts[2:6])
                        # Convert TLBR to TLWH
                        x, y = x1, y1
                        w, h = x2 - x1, y2 - y1
                    else:
                        # Format: frame,id,x,y,w,h,conf,class,...
                        x, y, w, h = map(float, parts[2:6])
                    
                    conf = float(parts[6]) if len(parts) > 6 else 1.0
                    cls_id = int(parts[7]) if len(parts) > 7 else -1
                    
                    data.append({
                        'frame': frame,
                        'id': track_id,
                        'bb_left': x,
                        'bb_top': y,
                        'bb_width': w,
                        'bb_height': h,
                        'conf': conf,
                        'class': cls_id
                    })
    return pd.DataFrame(data)

def compute_class_metrics(gt_dir, pred_dir, output_file):
    """Compute metrics for each class separately."""
    
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    
    # Initialize accumulators per class
    class_accs = {cls_id: [] for cls_id in CLASS_MAPPING.keys()}
    sequence_names = {cls_id: [] for cls_id in CLASS_MAPPING.keys()}
    
    # Get all sequences
    gt_files = sorted(gt_dir.glob('*.txt'))
    
    print(f"\n📊 Computing per-class metrics for {len(gt_files)} sequences...")
    
    for gt_file in tqdm(gt_files, desc="Processing sequences"):
        seq_name = gt_file.stem
        pred_file = pred_dir / f"{seq_name}.txt"
        
        if not pred_file.exists():
            continue
        
        # Load data
        gt_data = load_mot_with_classes(gt_file)
        pred_data = load_mot_with_classes(pred_file)
        
        if gt_data.empty or pred_data.empty:
            continue
        
        # Process each class
        for cls_id, cls_name in CLASS_MAPPING.items():
            # Filter by class
            gt_cls = gt_data[gt_data['class'] == cls_id]
            pred_cls = pred_data[pred_data['class'] == cls_id]
            
            if gt_cls.empty:
                continue
            
            # Create accumulator for this class in this sequence
            acc = mm.MOTAccumulator(auto_id=True)
            
            # Get all frames
            all_frames = sorted(set(gt_cls['frame'].unique()) | set(pred_cls['frame'].unique()))
            
            for frame in all_frames:
                gt_frame = gt_cls[gt_cls['frame'] == frame]
                pred_frame = pred_cls[pred_cls['frame'] == frame]
                
                # Extract bounding boxes
                gt_ids = gt_frame['id'].values
                pred_ids = pred_frame['id'].values
                
                if len(gt_ids) == 0 and len(pred_ids) == 0:
                    continue
                
                # Compute IoU distance matrix
                if len(gt_ids) > 0 and len(pred_ids) > 0:
                    gt_boxes = gt_frame[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values
                    pred_boxes = pred_frame[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values
                    
                    # Convert TLWH to TLBR for IoU
                    gt_tlbr = np.column_stack([
                        gt_boxes[:, 0],
                        gt_boxes[:, 1],
                        gt_boxes[:, 0] + gt_boxes[:, 2],
                        gt_boxes[:, 1] + gt_boxes[:, 3]
                    ])
                    pred_tlbr = np.column_stack([
                        pred_boxes[:, 0],
                        pred_boxes[:, 1],
                        pred_boxes[:, 0] + pred_boxes[:, 2],
                        pred_boxes[:, 1] + pred_boxes[:, 3]
                    ])
                    
                    # Compute IoU matrix
                    dists = mm.distances.iou_matrix(gt_tlbr, pred_tlbr, max_iou=0.5)
                else:
                    dists = np.empty((len(gt_ids), len(pred_ids)))
                
                acc.update(gt_ids, pred_ids, dists)
            
            # Add accumulator if it has events
            if len(acc.events) > 0:
                class_accs[cls_id].append(acc)
                sequence_names[cls_id].append(seq_name)
    
    # Compute metrics per class
    print("\n📈 Computing metrics per class...")
    mh = mm.metrics.create()
    
    class_metrics = {}
    
    for cls_id, cls_name in CLASS_MAPPING.items():
        if len(class_accs[cls_id]) == 0:
            print(f"⚠️  No data for class {cls_name}")
            continue
        
        print(f"\n🔍 Computing metrics for {cls_name} ({len(class_accs[cls_id])} sequences)...")
        
        # Compute summary for this class
        summary = mh.compute_many(
            class_accs[cls_id],
            metrics=[
                'num_frames', 'mota', 'motp', 'idf1',
                'num_switches', 'num_fragmentations',
                'num_false_positives', 'num_misses',
                'mostly_tracked', 'mostly_lost',
                'precision', 'recall', 'num_objects'
            ],
            names=sequence_names[cls_id]
        )
        
        # Aggregate overall for this class using weighted average
        # Based on number of objects per sequence
        weights = summary['num_objects'].values
        total_objects = weights.sum()
        
        if total_objects == 0:
            continue
        
        # Weighted averages for percentage metrics
        mota_avg = (summary['mota'] * weights).sum() / total_objects
        motp_avg = (summary['motp'] * weights).sum() / total_objects
        idf1_avg = (summary['idf1'] * weights).sum() / total_objects
        precision_avg = (summary['precision'] * weights).sum() / total_objects
        recall_avg = (summary['recall'] * weights).sum() / total_objects
        
        # Sum for count metrics
        num_switches = int(summary['num_switches'].sum())
        num_fragmentations = int(summary['num_fragmentations'].sum())
        num_fp = int(summary['num_false_positives'].sum())
        num_fn = int(summary['num_misses'].sum())
        num_mt = int(summary['mostly_tracked'].sum())
        num_ml = int(summary['mostly_lost'].sum())
        
        class_metrics[cls_name] = {
            'MOTA': float(mota_avg * 100),
            'MOTP': float(motp_avg) if not np.isnan(motp_avg) else None,
            'IDF1': float(idf1_avg * 100),
            'IDSW': num_switches,
            'Frag': num_fragmentations,
            'FP': num_fp,
            'FN': num_fn,
            'MT': num_mt,
            'ML': num_ml,
            'Precision': float(precision_avg * 100),
            'Recall': float(recall_avg * 100),
            'num_sequences': len(class_accs[cls_id]),
            'num_objects': int(total_objects)
        }
        
        print(f"  ✓ {cls_name}: MOTA={mota_avg*100:.2f}%, IDF1={idf1_avg*100:.2f}%, IDSW={num_switches}")
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        'method': 'TrackSSM (Zero-Shot)',
        'dataset': 'NuScenes val CAM_FRONT',
        'per_class_metrics': class_metrics
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"\n✅ Per-class metrics saved to: {output_path}")
    
    return class_metrics

def get_gt_pred_pairs(val_dir, pred_dir):
    """Get pairs of GT and prediction files."""
    val_dir = Path(val_dir)
    pred_dir = Path(pred_dir)
    
    pairs = []
    for seq_dir in sorted(val_dir.iterdir()):
        if seq_dir.is_dir():
            gt_file = seq_dir / 'gt' / 'gt.txt'
            pred_file = pred_dir / f"{seq_dir.name}.txt"
            if gt_file.exists() and pred_file.exists():
                pairs.append((gt_file, pred_file, seq_dir.name))
    
    return pairs

def assign_classes_to_tracks(gt_data, pred_data):
    """
    Assign class labels to predictions based on GT overlap.
    For each prediction, find the most overlapping GT and assign its class.
    """
    pred_with_class = pred_data.copy()
    pred_with_class['class'] = -1  # Initialize
    
    for frame in pred_data['frame'].unique():
        gt_frame = gt_data[gt_data['frame'] == frame]
        pred_frame_idx = pred_data['frame'] == frame
        pred_frame = pred_data[pred_frame_idx]
        
        if gt_frame.empty or pred_frame.empty:
            continue
        
        # Get boxes
        gt_boxes = gt_frame[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values
        pred_boxes = pred_frame[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values
        
        # Convert to TLBR
        gt_tlbr = np.column_stack([
            gt_boxes[:, 0], gt_boxes[:, 1],
            gt_boxes[:, 0] + gt_boxes[:, 2],
            gt_boxes[:, 1] + gt_boxes[:, 3]
        ])
        pred_tlbr = np.column_stack([
            pred_boxes[:, 0], pred_boxes[:, 1],
            pred_boxes[:, 0] + pred_boxes[:, 2],
            pred_boxes[:, 1] + pred_boxes[:, 3]
        ])
        
        # Compute IoU for assignment
        from scipy.optimize import linear_sum_assignment
        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        
        for i, pred_box in enumerate(pred_tlbr):
            for j, gt_box in enumerate(gt_tlbr):
                # Calculate IoU
                x1 = max(pred_box[0], gt_box[0])
                y1 = max(pred_box[1], gt_box[1])
                x2 = min(pred_box[2], gt_box[2])
                y2 = min(pred_box[3], gt_box[3])
                
                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                    union = pred_area + gt_area - intersection
                    iou_matrix[i, j] = intersection / union if union > 0 else 0
        
        # Assign classes based on best IoU match (threshold 0.5)
        pred_indices = pred_frame.index.values
        for i, pred_idx in enumerate(pred_indices):
            best_gt_idx = np.argmax(iou_matrix[i])
            if iou_matrix[i, best_gt_idx] > 0.5:
                assigned_class = gt_frame.iloc[best_gt_idx]['class']
                pred_with_class.at[pred_idx, 'class'] = assigned_class
    
    return pred_with_class

def compute_class_metrics_v2(val_dir, pred_dir, output_file):
    """Compute metrics for each class separately - version for NuScenes folder structure."""
    
    # Get GT/pred pairs
    pairs = get_gt_pred_pairs(val_dir, pred_dir)
    
    print(f"\n📊 Found {len(pairs)} sequence pairs...")
    
    # Initialize accumulators per class
    class_accs = {cls_id: [] for cls_id in CLASS_MAPPING.keys()}
    sequence_names = {cls_id: [] for cls_id in CLASS_MAPPING.keys()}
    
    for gt_file, pred_file, seq_name in tqdm(pairs, desc="Processing sequences"):
        # Load data
        gt_data = load_mot_with_classes(gt_file, is_tlbr=False)  # GT is TLWH
        pred_data_raw = load_mot_with_classes(pred_file, is_tlbr=True)  # Tracking results are TLBR
        
        if gt_data.empty or pred_data_raw.empty:
            continue
        
        # Assign classes to predictions based on GT overlap
        pred_data = assign_classes_to_tracks(gt_data, pred_data_raw)
        
        if gt_data.empty or pred_data.empty:
            continue
        
        # Process each class
        for cls_id, cls_name in CLASS_MAPPING.items():
            # Filter by class
            gt_cls = gt_data[gt_data['class'] == cls_id]
            pred_cls = pred_data[pred_data['class'] == cls_id]
            
            if gt_cls.empty:
                continue
            
            # Create accumulator for this class in this sequence
            acc = mm.MOTAccumulator(auto_id=True)
            
            # Get all frames
            all_frames = sorted(set(gt_cls['frame'].unique()) | set(pred_cls['frame'].unique()))
            
            for frame in all_frames:
                gt_frame = gt_cls[gt_cls['frame'] == frame]
                pred_frame = pred_cls[pred_cls['frame'] == frame]
                
                # Extract bounding boxes
                gt_ids = gt_frame['id'].values
                pred_ids = pred_frame['id'].values
                
                if len(gt_ids) == 0 and len(pred_ids) == 0:
                    continue
                
                # Compute IoU distance matrix
                if len(gt_ids) > 0 and len(pred_ids) > 0:
                    gt_boxes = gt_frame[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values
                    pred_boxes = pred_frame[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values
                    
                    # Convert TLWH to TLBR for IoU
                    gt_tlbr = np.column_stack([
                        gt_boxes[:, 0],
                        gt_boxes[:, 1],
                        gt_boxes[:, 0] + gt_boxes[:, 2],
                        gt_boxes[:, 1] + gt_boxes[:, 3]
                    ])
                    pred_tlbr = np.column_stack([
                        pred_boxes[:, 0],
                        pred_boxes[:, 1],
                        pred_boxes[:, 0] + pred_boxes[:, 2],
                        pred_boxes[:, 1] + pred_boxes[:, 3]
                    ])
                    
                    # Compute IoU matrix
                    dists = mm.distances.iou_matrix(gt_tlbr, pred_tlbr, max_iou=0.5)
                else:
                    dists = np.empty((len(gt_ids), len(pred_ids)))
                
                acc.update(gt_ids, pred_ids, dists)
            
            # Add accumulator if it has events
            if len(acc.events) > 0:
                class_accs[cls_id].append(acc)
                sequence_names[cls_id].append(seq_name)
    
    # Compute metrics per class
    print("\n📈 Computing metrics per class...")
    mh = mm.metrics.create()
    
    class_metrics = {}
    
    for cls_id, cls_name in CLASS_MAPPING.items():
        if len(class_accs[cls_id]) == 0:
            print(f"⚠️  No data for class {cls_name}")
            continue
        
        print(f"\n🔍 Computing metrics for {cls_name} ({len(class_accs[cls_id])} sequences)...")
        
        # Compute summary for this class
        summary = mh.compute_many(
            class_accs[cls_id],
            metrics=[
                'num_frames', 'mota', 'motp', 'idf1',
                'num_switches', 'num_fragmentations',
                'num_false_positives', 'num_misses',
                'mostly_tracked', 'mostly_lost',
                'precision', 'recall', 'num_objects'
            ],
            names=sequence_names[cls_id]
        )
        
        # Aggregate overall for this class using weighted average
        weights = summary['num_objects'].values
        total_objects = weights.sum()
        
        if total_objects == 0:
            continue
        
        # Weighted averages for percentage metrics
        mota_avg = (summary['mota'] * weights).sum() / total_objects
        motp_avg = (summary['motp'] * weights).sum() / total_objects
        idf1_avg = (summary['idf1'] * weights).sum() / total_objects
        precision_avg = (summary['precision'] * weights).sum() / total_objects
        recall_avg = (summary['recall'] * weights).sum() / total_objects
        
        # Sum for count metrics
        num_switches = int(summary['num_switches'].sum())
        num_fragmentations = int(summary['num_fragmentations'].sum())
        num_fp = int(summary['num_false_positives'].sum())
        num_fn = int(summary['num_misses'].sum())
        num_mt = int(summary['mostly_tracked'].sum())
        num_ml = int(summary['mostly_lost'].sum())
        
        class_metrics[cls_name] = {
            'MOTA': float(mota_avg * 100),
            'MOTP': float(motp_avg) if not np.isnan(motp_avg) else None,
            'IDF1': float(idf1_avg * 100),
            'IDSW': num_switches,
            'Frag': num_fragmentations,
            'FP': num_fp,
            'FN': num_fn,
            'MT': num_mt,
            'ML': num_ml,
            'Precision': float(precision_avg * 100),
            'Recall': float(recall_avg * 100),
            'num_sequences': len(class_accs[cls_id]),
            'num_objects': int(total_objects)
        }
        
        print(f"  ✓ {cls_name}: MOTA={mota_avg*100:.2f}%, IDF1={idf1_avg*100:.2f}%, IDSW={num_switches}")
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        'method': 'TrackSSM (Zero-Shot)',
        'dataset': 'NuScenes val CAM_FRONT',
        'per_class_metrics': class_metrics
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"\n✅ Per-class metrics saved to: {output_path}")
    
    return class_metrics

if __name__ == '__main__':
    VAL_DIR = './data/nuscenes_mot_front_7classes/val'
    PRED_DIR = './results/nuscenes_trackssm_7classes'
    OUTPUT_FILE = './results/final_evaluation/trackssm_7classes_per_class_metrics.json'
    
    metrics = compute_class_metrics_v2(VAL_DIR, PRED_DIR, OUTPUT_FILE)
    
    print("\n" + "="*80)
    print("PER-CLASS METRICS SUMMARY")
    print("="*80)
    for cls_name, m in metrics.items():
        print(f"\n{cls_name.upper()}:")
        print(f"  MOTA: {m['MOTA']:.2f}%")
        print(f"  IDF1: {m['IDF1']:.2f}%")
        print(f"  IDSW: {m['IDSW']}")
        print(f"  Objects: {m['num_objects']}")

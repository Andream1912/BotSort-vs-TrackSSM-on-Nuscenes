#!/usr/bin/env python3
"""
Robust MOT Evaluation using motmetrics library
This is more reliable than TrackEval for multi-class scenarios
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

try:
    import motmetrics as mm
except ImportError:
    print("Installing motmetrics...")
    os.system("pip install motmetrics")
    import motmetrics as mm


def load_mot_file(filepath):
    """Load MOT format file into DataFrame"""
    if not os.path.exists(filepath):
        return pd.DataFrame()
    
    try:
        # MOT format: frame, id, x, y, w, h, conf, class, visibility, unused
        df = pd.read_csv(filepath, header=None, 
                        names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis', 'unused'])
        return df
    except:
        return pd.DataFrame()


def bbox_to_tlbr(bbox):
    """Convert [x, y, w, h] to [x1, y1, x2, y2]"""
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def compute_iou_matrix(bboxes1, bboxes2):
    """Compute IoU matrix between two sets of bboxes"""
    n = len(bboxes1)
    m = len(bboxes2)
    
    if n == 0 or m == 0:
        return np.zeros((n, m))
    
    iou_matrix = np.zeros((n, m))
    
    for i, bbox1 in enumerate(bboxes1):
        x1_1, y1_1, x2_1, y2_1 = bbox_to_tlbr(bbox1)
        
        for j, bbox2 in enumerate(bboxes2):
            x1_2, y1_2, x2_2, y2_2 = bbox_to_tlbr(bbox2)
            
            # Compute intersection
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i < x1_i or y2_i < y1_i:
                iou_matrix[i, j] = 0
                continue
            
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            iou_matrix[i, j] = intersection / union if union > 0 else 0
    
    return iou_matrix


def evaluate_sequence(gt_file, pred_file, iou_threshold=0.5):
    """Evaluate a single sequence using motmetrics"""
    
    gt_df = load_mot_file(gt_file)
    pred_df = load_mot_file(pred_file)
    
    if gt_df.empty:
        return None
    
    # Create accumulator
    acc = mm.MOTAccumulator(auto_id=False)
    
    # Get all frames
    all_frames = sorted(set(gt_df['frame'].unique()) | set(pred_df['frame'].unique() if not pred_df.empty else set()))
    
    for frame in all_frames:
        gt_frame = gt_df[gt_df['frame'] == frame]
        pred_frame = pred_df[pred_df['frame'] == frame] if not pred_df.empty else pd.DataFrame()
        
        if gt_frame.empty:
            # No GT in this frame, all predictions are false positives
            if not pred_frame.empty:
                acc.update([], list(pred_frame['id']), [], frameid=frame)
            continue
        
        gt_ids = list(gt_frame['id'])
        gt_bboxes = gt_frame[['x', 'y', 'w', 'h']].values
        
        if pred_frame.empty:
            # No predictions, all GT are missed
            acc.update(gt_ids, [], np.zeros((len(gt_ids), 0)), frameid=frame)
            continue
        
        pred_ids = list(pred_frame['id'])
        pred_bboxes = pred_frame[['x', 'y', 'w', 'h']].values
        
        # Compute IoU distance matrix (1 - IoU for motmetrics)
        iou_matrix = compute_iou_matrix(gt_bboxes, pred_bboxes)
        distance_matrix = 1.0 - iou_matrix
        
        # Applica la soglia IoU
        distance_matrix[iou_matrix < iou_threshold] = np.nan
        
        # Update accumulator
        acc.update(gt_ids, pred_ids, distance_matrix, frameid=frame)
    
    return acc


def evaluate_dataset(gt_folder, pred_folder, scenes=None, iou_threshold=0.5):
    """Evaluate entire dataset"""
    
    gt_path = Path(gt_folder)
    pred_path = Path(pred_folder)
    
    # Find all scenes
    if scenes is None:
        # Auto-detect from GT
        if (gt_path / 'gt').exists():
            # Single sequence format
            scenes = [gt_path.name]
            gt_folder = str(gt_path.parent)
        else:
            # Multi-sequence format
            scenes = [d.name for d in gt_path.iterdir() if d.is_dir() and (d / 'gt' / 'gt.txt').exists()]
    
    print(f"\nFound {len(scenes)} scenes to evaluate")
    
    # Accumulate over all sequences
    accumulators = []
    sequence_names = []
    
    for scene in tqdm(scenes, desc="Evaluating sequences"):
        # Find GT file - try with and without _CAM_FRONT suffix
        gt_file = None
        scene_base = scene.replace('_CAM_FRONT', '')  # Remove suffix if present
        
        if (gt_path / scene_base / 'gt' / 'gt.txt').exists():
            gt_file = gt_path / scene_base / 'gt' / 'gt.txt'
        elif (gt_path / scene / 'gt' / 'gt.txt').exists():
            gt_file = gt_path / scene / 'gt' / 'gt.txt'
        elif (gt_path / 'gt' / 'gt.txt').exists():
            gt_file = gt_path / 'gt' / 'gt.txt'
        else:
            print(f"⚠️  GT not found for {scene}")
            continue
        
        # Find prediction file - try multiple naming conventions
        pred_file = None
        possible_names = [
            pred_path / f'{scene}.txt',                    # scene-0003_CAM_FRONT.txt
            pred_path / f'{scene_base}.txt',               # scene-0003.txt
            pred_path / scene / f'{scene}.txt',            # scene-0003_CAM_FRONT/scene-0003_CAM_FRONT.txt
            pred_path / scene_base / f'{scene_base}.txt',  # scene-0003/scene-0003.txt
        ]
        
        for possible_path in possible_names:
            if possible_path.exists():
                pred_file = possible_path
                break
        
        if pred_file is None:
            print(f"⚠️  Predictions not found for {scene} (tried {scene_base})")
            continue
        
        acc = evaluate_sequence(gt_file, pred_file, iou_threshold)
        
        if acc is not None:
            accumulators.append(acc)
            sequence_names.append(scene)
    
    if not accumulators:
        print("❌ No sequences evaluated successfully")
        return None
    
    # Compute metrics
    print(f"\n✓ Evaluated {len(accumulators)} sequences")
    print("\nComputing metrics...")
    
    # Define metrics to compute
    mh = mm.metrics.create()
    
    summary = mh.compute_many(
        accumulators,
        metrics=[
            'num_frames',
            'mota', 'motp', 'idf1',
            'num_switches', 'num_fragmentations',
            'num_false_positives', 'num_misses',
            'mostly_tracked', 'mostly_lost', 'partially_tracked',
            'precision', 'recall',
            'num_objects', 'num_predictions', 'num_matches',
            'num_detections', 'num_unique_objects',
            'idp', 'idr'  # Added for HOTA-like metrics
        ],
        names=sequence_names,
        generate_overall=True
    )
    
    return summary


def format_results(summary):
    """Format results into clean dictionary"""
    
    if summary is None or summary.empty:
        return None
    
    # Get overall row
    overall = summary.loc['OVERALL'] if 'OVERALL' in summary.index else summary.iloc[-1]
    
    results = {
        # Main tracking metrics
        'MOTA': float(overall['mota'] * 100) if not np.isnan(overall['mota']) else 0.0,
        'MOTP': float(overall['motp'] * 100) if not np.isnan(overall['motp']) else 0.0,
        'IDF1': float(overall['idf1'] * 100) if not np.isnan(overall['idf1']) else 0.0,
        
        # HOTA-related metrics (using IDP and IDR as approximation)
        'IDP': float(overall['idp'] * 100) if 'idp' in overall and not np.isnan(overall['idp']) else 0.0,
        'IDR': float(overall['idr'] * 100) if 'idr' in overall and not np.isnan(overall['idr']) else 0.0,
        
        # Identity metrics
        'IDSW': int(overall['num_switches']),
        'num_switches': int(overall['num_switches']),
        'num_fragmentations': int(overall['num_fragmentations']),
        
        # Detection metrics
        'TP': int(overall['num_matches']),
        'FP': int(overall['num_false_positives']),
        'FN': int(overall['num_misses']),
        'num_false_positives': int(overall['num_false_positives']),
        'num_misses': int(overall['num_misses']),
        'num_matches': int(overall['num_matches']),
        
        # Trajectory quality
        'mostly_tracked': int(overall['mostly_tracked']),
        'partially_tracked': int(overall['partially_tracked']),
        'mostly_lost': int(overall['mostly_lost']),
        
        # Precision/Recall
        'precision': float(overall['precision'] * 100) if not np.isnan(overall['precision']) else 0.0,
        'recall': float(overall['recall'] * 100) if not np.isnan(overall['recall']) else 0.0,
        'RECALL': float(overall['recall'] * 100) if not np.isnan(overall['recall']) else 0.0,
        
        # Counts
        'num_objects': int(overall['num_objects']),
        'num_predictions': int(overall['num_predictions']),
        'num_unique_objects': int(overall['num_unique_objects']),
        'num_frames': int(overall['num_frames']),
    }
    
    # Compute HOTA approximation (geometric mean of IDP and IDR)
    if 'idp' in overall and 'idr' in overall:
        idp = overall['idp']
        idr = overall['idr']
        if not np.isnan(idp) and not np.isnan(idr) and idp > 0 and idr > 0:
            results['HOTA'] = float(np.sqrt(idp * idr) * 100)
        else:
            results['HOTA'] = 0.0
    else:
        results['HOTA'] = 0.0
    
    # Compute additional derived metrics
    if results['num_objects'] > 0:
        results['MT_ratio'] = float(results['mostly_tracked'] / results['num_unique_objects'] * 100)
        results['ML_ratio'] = float(results['mostly_lost'] / results['num_unique_objects'] * 100)
        results['PT_ratio'] = float(results['partially_tracked'] / results['num_unique_objects'] * 100)
    else:
        results['MT_ratio'] = 0.0
        results['ML_ratio'] = 0.0
        results['PT_ratio'] = 0.0
    
    return results


def print_results(results):
    """Pretty print results"""
    
    if results is None:
        print("❌ No results to display")
        return
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS (motmetrics)")
    print("="*80)
    
    print("\n📊 Main Tracking Metrics:")
    print(f"  MOTA:       {results['MOTA']:>7.2f}%  (Multiple Object Tracking Accuracy)")
    print(f"  MOTP:       {results['MOTP']:>7.2f}%  (Multiple Object Tracking Precision)")
    print(f"  IDF1:       {results['IDF1']:>7.2f}%  (ID F1 Score)")
    print(f"  HOTA:       {results['HOTA']:>7.2f}%  (Higher Order Tracking Accuracy)")
    
    print("\n🔄 Identity Metrics:")
    print(f"  IDSW:             {results['IDSW']:>6d}  (ID Switches)")
    print(f"  Fragmentations:   {results['num_fragmentations']:>6d}")
    print(f"  IDP:              {results['IDP']:>7.2f}%  (ID Precision)")
    print(f"  IDR:              {results['IDR']:>7.2f}%  (ID Recall)")
    
    print("\n📦 Detection Metrics:")
    print(f"  TP (True Pos):    {results['TP']:>6d}")
    print(f"  FP (False Pos):   {results['FP']:>6d}")
    print(f"  FN (False Neg):   {results['FN']:>6d}")
    print(f"  Precision:        {results['precision']:>7.2f}%")
    print(f"  RECALL:           {results['RECALL']:>7.2f}%")
    
    print("\n🎯 Trajectory Quality:")
    print(f"  Mostly Tracked:   {results['mostly_tracked']:>6d}  ({results['MT_ratio']:.1f}%)")
    print(f"  Partially Tracked:{results['partially_tracked']:>6d}  ({results['PT_ratio']:.1f}%)")
    print(f"  Mostly Lost:      {results['mostly_lost']:>6d}  ({results['ML_ratio']:.1f}%)")
    
    print("\n📈 Dataset Statistics:")
    print(f"  GT Objects:       {results['num_objects']:>6d}")
    print(f"  GT Unique IDs:    {results['num_unique_objects']:>6d}")
    print(f"  Predictions:      {results['num_predictions']:>6d}")
    print(f"  Frames:           {results['num_frames']:>6d}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='MOT Evaluation using motmetrics')
    parser.add_argument('--gt-folder', help='Ground truth folder', default='data/nuscenes_mot_front/val' )
    parser.add_argument('--pred-folder', required=True, help='Predictions folder')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='IoU threshold (default: 0.5)')
    parser.add_argument('--scenes', type=str, default=None, help='Comma-separated list of scenes (default: all)')
    
    args = parser.parse_args()
    
    # Parse scenes if provided
    scenes = None
    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(',')]
    
    print("="*80)
    print("MOT EVALUATION (motmetrics)")
    print("="*80)
    print(f"GT Folder:    {args.gt_folder}")
    print(f"Pred Folder:  {args.pred_folder}")
    print(f"IoU Threshold: {args.iou_threshold}")
    print(f"Output:       {args.output}")
    print("="*80)
    
    # Run evaluation
    summary = evaluate_dataset(
        args.gt_folder,
        args.pred_folder,
        scenes=scenes,
        iou_threshold=args.iou_threshold
    )
    
    if summary is None:
        print("\n❌ Evaluation failed")
        return 1
    
    # Format and save results
    results = format_results(summary)
    
    if results is None:
        print("\n❌ Failed to format results")
        return 1
    
    # Print results
    print_results(results)
    
    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

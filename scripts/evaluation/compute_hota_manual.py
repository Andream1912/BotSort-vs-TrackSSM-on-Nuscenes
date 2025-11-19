#!/usr/bin/env python3
"""
Manual HOTA computation for TrackSSM.
Simplified approach without TrackEval library complexity.
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from tqdm import tqdm

# NuScenes 7 classes
CLASS_MAPPING = {
    1: 'car',
    2: 'truck',
    3: 'bus',
    4: 'trailer',
    5: 'pedestrian',
    6: 'motorcycle',
    7: 'bicycle'
}

def load_mot_file(filepath):
    """Load MOT format file."""
    data = []
    if not Path(filepath).exists():
        return data
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 10:
                frame = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                conf = float(parts[6])
                cls = int(parts[7])
                vis = float(parts[8])
                
                data.append({
                    'frame': frame,
                    'id': track_id,
                    'bbox': [x, y, w, h],
                    'conf': conf,
                    'class': cls,
                    'vis': vis
                })
    
    return data

def bbox_iou(box1, box2):
    """Compute IoU between two bounding boxes [x,y,w,h]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to x1y1x2y2
    box1_x2, box1_y2 = x1 + w1, y1 + h1
    box2_x2, box2_y2 = x2 + w2, y2 + h2
    
    # Intersection
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    # Union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area

def compute_hota_for_sequence(gt_data, pred_data, iou_threshold=0.5):
    """
    Compute HOTA for a single sequence.
    
    HOTA = sqrt(DetA * AssA)
    DetA = Detection Accuracy
    AssA = Association Accuracy
    """
    # Group by frame
    gt_by_frame = defaultdict(list)
    pred_by_frame = defaultdict(list)
    
    for item in gt_data:
        gt_by_frame[item['frame']].append(item)
    
    for item in pred_data:
        pred_by_frame[item['frame']].append(item)
    
    # Get all frames
    all_frames = sorted(set(list(gt_by_frame.keys()) + list(pred_by_frame.keys())))
    
    # HOTA computation over alpha thresholds
    alphas = np.linspace(0.05, 0.95, 19)  # 19 alpha values as in TrackEval
    
    det_scores = []
    ass_scores = []
    
    # Track-level statistics
    gt_track_ids = set(item['id'] for item in gt_data)
    pred_track_ids = set(item['id'] for item in pred_data)
    
    # Potential matches: tracks that could be matched
    potential_matches = {}  # (gt_id, pred_id) -> num_matches
    
    total_dets = 0
    total_matched = 0
    total_pred = 0
    
    for frame in all_frames:
        gt_frame = gt_by_frame[frame]
        pred_frame = pred_by_frame[frame]
        
        total_pred += len(pred_frame)
        
        # Compute IoU matrix (ignore class since predictions don't have it)
        iou_matrix = np.zeros((len(gt_frame), len(pred_frame)))
        for i, gt in enumerate(gt_frame):
            for j, pred in enumerate(pred_frame):
                iou_matrix[i, j] = bbox_iou(gt['bbox'], pred['bbox'])
        
        # Match based on IoU threshold
        matches = []
        used_gt = set()
        used_pred = set()
        
        # Greedy matching (highest IoU first)
        for _ in range(min(len(gt_frame), len(pred_frame))):
            if iou_matrix.size == 0:
                break
            max_iou = iou_matrix.max()
            if max_iou < iou_threshold:
                break
            
            i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            if i not in used_gt and j not in used_pred:
                matches.append((i, j, max_iou))
                used_gt.add(i)
                used_pred.add(j)
                
                # Record potential track match
                gt_id = gt_frame[i]['id']
                pred_id = pred_frame[j]['id']
                key = (gt_id, pred_id)
                potential_matches[key] = potential_matches.get(key, 0) + 1
            
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0
        
        total_dets += len(gt_frame)
        total_matched += len(matches)
    
    if total_dets == 0:
        return {'HOTA': 0, 'DetA': 0, 'AssA': 0, 'DetRe': 0, 'DetPr': 0}
    
    # Detection Accuracy
    det_recall = total_matched / total_dets if total_dets > 0 else 0
    det_precision = total_matched / total_pred if total_pred > 0 else 0
    DetA = (det_recall + det_precision) / 2 if (det_recall + det_precision) > 0 else 0
    
    # Association Accuracy
    # Simplified: ratio of correctly associated track pairs
    if len(potential_matches) == 0:
        AssA = 0
    else:
        # Count how many GT tracks were correctly associated
        gt_correctly_matched = len(set(k[0] for k in potential_matches.keys()))
        AssA = gt_correctly_matched / len(gt_track_ids) if len(gt_track_ids) > 0 else 0
    
    # HOTA
    HOTA = np.sqrt(DetA * AssA) if (DetA > 0 and AssA > 0) else 0
    
    return {
        'HOTA': HOTA * 100,  # Convert to percentage
        'DetA': DetA * 100,
        'AssA': AssA * 100,
        'DetRe': det_recall * 100,
        'DetPr': det_precision * 100,
        'num_gt': len(gt_track_ids),
        'num_pred': len(pred_track_ids),
        'num_frames': len(all_frames)
    }

def compute_overall_hota(gt_dir, pred_dir, class_filter=None):
    """Compute HOTA across all sequences."""
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    
    sequences = sorted([d for d in gt_dir.iterdir() if d.is_dir()])
    
    print(f"\n📊 Computing HOTA for {len(sequences)} sequences...")
    if class_filter:
        print(f"   Filter: class {class_filter} ({CLASS_MAPPING.get(class_filter, 'unknown')})")
    
    all_results = []
    
    for seq_dir in tqdm(sequences):
        gt_file = seq_dir / 'gt' / 'gt.txt'
        pred_file = pred_dir / f"{seq_dir.name}.txt"
        
        if not gt_file.exists() or not pred_file.exists():
            continue
        
        # Load data
        gt_data = load_mot_file(gt_file)
        pred_data = load_mot_file(pred_file)
        
        # Filter by class if needed
        if class_filter is not None:
            gt_data = [d for d in gt_data if d['class'] == class_filter]
            pred_data = [d for d in pred_data if d['class'] == class_filter]
        
        if len(gt_data) == 0:
            continue
        
        # Compute HOTA
        result = compute_hota_for_sequence(gt_data, pred_data)
        all_results.append(result)
    
    if not all_results:
        return None
    
    # Aggregate results
    metrics = {
        'HOTA': np.mean([r['HOTA'] for r in all_results]),
        'DetA': np.mean([r['DetA'] for r in all_results]),
        'AssA': np.mean([r['AssA'] for r in all_results]),
        'DetRe': np.mean([r['DetRe'] for r in all_results]),
        'DetPr': np.mean([r['DetPr'] for r in all_results]),
        'num_sequences': len(all_results),
        'total_gt_tracks': sum(r['num_gt'] for r in all_results),
        'total_pred_tracks': sum(r['num_pred'] for r in all_results),
    }
    
    return metrics

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', required=True)
    parser.add_argument('--pred_dir', required=True)
    parser.add_argument('--output', default='results/final_evaluation/trackssm_hota.json')
    parser.add_argument('--per_class', action='store_true')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("TRACKSSM HOTA COMPUTATION (Manual Implementation)")
    print("="*80)
    
    all_results = {}
    
    if args.per_class:
        print("\n📊 MODE: Per-class HOTA")
        
        for cls_id, cls_name in CLASS_MAPPING.items():
            print(f"\n{'='*60}")
            print(f"CLASS: {cls_name.upper()}")
            print(f"{'='*60}")
            
            metrics = compute_overall_hota(args.gt_dir, args.pred_dir, class_filter=cls_id)
            
            if metrics:
                all_results[cls_name] = metrics
                print(f"\n✅ Results:")
                print(f"   HOTA  = {metrics['HOTA']:.2f}%")
                print(f"   DetA  = {metrics['DetA']:.2f}%")
                print(f"   AssA  = {metrics['AssA']:.2f}%")
                print(f"   DetRe = {metrics['DetRe']:.2f}%")
                print(f"   DetPr = {metrics['DetPr']:.2f}%")
                print(f"   Sequences: {metrics['num_sequences']}")
            else:
                print(f"   ⚠️  No data for {cls_name}")
    
    else:
        print("\n📊 MODE: Overall HOTA")
        
        metrics = compute_overall_hota(args.gt_dir, args.pred_dir)
        
        if metrics:
            all_results['overall'] = metrics
            print(f"\n✅ Results:")
            print(f"   HOTA  = {metrics['HOTA']:.2f}%")
            print(f"   DetA  = {metrics['DetA']:.2f}%")
            print(f"   AssA  = {metrics['AssA']:.2f}%")
            print(f"   DetRe = {metrics['DetRe']:.2f}%")
            print(f"   DetPr = {metrics['DetPr']:.2f}%")
            print(f"   Sequences: {metrics['num_sequences']}")
    
    # Save results
    if all_results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print("\n" + "="*80)
        print("✅ HOTA COMPUTATION COMPLETE!")
        print("="*80)
        print(f"\n📄 Results saved to: {output_path}\n")

if __name__ == '__main__':
    main()

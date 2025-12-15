#!/usr/bin/env python3
"""
Compare detection ranges between GT and detector predictions.

This script analyzes if detector and GT operate at the same distance range
by comparing bbox size distributions, positions, and overlap patterns.
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


def load_detections_by_frame(file_path, is_gt=False):
    """
    Load detections grouped by frame.
    
    Returns:
        dict: {frame_id: [list of detections]}
    """
    detections_by_frame = defaultdict(list)
    
    if not file_path.exists():
        return detections_by_frame
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            
            try:
                frame_id = int(parts[0])
                x, y, w, h = map(float, parts[2:6])
                conf = float(parts[6]) if len(parts) > 6 else 1.0
                class_id = int(parts[7]) if len(parts) > 7 else 1
                
                # Calculate bbox properties for range analysis
                cx = x + w / 2  # Center X
                cy = y + h / 2  # Center Y
                area = w * h
                
                detections_by_frame[frame_id].append({
                    'bbox': [x, y, w, h],
                    'center': [cx, cy],
                    'area': area,
                    'confidence': conf,
                    'class_id': class_id
                })
            except (ValueError, IndexError):
                continue
    
    return detections_by_frame


def analyze_spatial_overlap(gt_dets, pred_dets, iou_thresh=0.1):
    """
    Analyze spatial overlap between GT and predictions.
    
    Returns:
        dict with overlap statistics
    """
    def compute_iou(bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        x1_max, y1_max = x1 + w1, y1 + h1
        x2_max, y2_max = x2 + w2, y2 + h2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1_max, x2_max)
        yi2 = min(y1_max, y2_max)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union_area = w1 * h1 + w2 * h2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    # Track which GT detections are matched
    matched_gt = set()
    matched_pred = set()
    ious = []
    
    for i, gt in enumerate(gt_dets):
        best_iou = 0
        best_j = -1
        
        for j, pred in enumerate(pred_dets):
            iou = compute_iou(gt['bbox'], pred['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        
        if best_iou >= iou_thresh:
            matched_gt.add(i)
            matched_pred.add(best_j)
            ious.append(best_iou)
    
    # Unmatched detections
    unmatched_gt = [gt_dets[i] for i in range(len(gt_dets)) if i not in matched_gt]
    unmatched_pred = [pred_dets[j] for j in range(len(pred_dets)) if j not in matched_pred]
    
    return {
        'matched_gt': len(matched_gt),
        'matched_pred': len(matched_pred),
        'unmatched_gt': unmatched_gt,
        'unmatched_pred': unmatched_pred,
        'mean_iou': np.mean(ious) if ious else 0,
        'total_gt': len(gt_dets),
        'total_pred': len(pred_dets)
    }


def analyze_range_differences(scene_name, gt_folder, pred_folder):
    """
    Analyze if detector operates at different range than GT.
    """
    gt_file = gt_folder / scene_name / 'gt' / 'gt.txt'
    pred_file = pred_folder / f'{scene_name}.txt'
    
    if not gt_file.exists() or not pred_file.exists():
        return None
    
    gt_by_frame = load_detections_by_frame(gt_file, is_gt=True)
    pred_by_frame = load_detections_by_frame(pred_file, is_gt=False)
    
    # Analyze each frame
    frame_analyses = []
    
    all_frames = set(gt_by_frame.keys()) | set(pred_by_frame.keys())
    
    for frame_id in sorted(all_frames):
        gt_dets = gt_by_frame.get(frame_id, [])
        pred_dets = pred_by_frame.get(frame_id, [])
        
        if len(gt_dets) == 0 and len(pred_dets) == 0:
            continue
        
        overlap = analyze_spatial_overlap(gt_dets, pred_dets, iou_thresh=0.1)
        
        # Analyze unmatched detections by size
        unmatched_gt_tiny = sum(1 for d in overlap['unmatched_gt'] if d['area'] < 1000)
        unmatched_gt_small = sum(1 for d in overlap['unmatched_gt'] if 1000 <= d['area'] < 5000)
        unmatched_gt_medium = sum(1 for d in overlap['unmatched_gt'] if 5000 <= d['area'] < 20000)
        unmatched_gt_large = sum(1 for d in overlap['unmatched_gt'] if d['area'] >= 20000)
        
        unmatched_pred_tiny = sum(1 for d in overlap['unmatched_pred'] if d['area'] < 1000)
        unmatched_pred_small = sum(1 for d in overlap['unmatched_pred'] if 1000 <= d['area'] < 5000)
        unmatched_pred_medium = sum(1 for d in overlap['unmatched_pred'] if 5000 <= d['area'] < 20000)
        unmatched_pred_large = sum(1 for d in overlap['unmatched_pred'] if d['area'] >= 20000)
        
        frame_analyses.append({
            'frame_id': frame_id,
            'total_gt': overlap['total_gt'],
            'total_pred': overlap['total_pred'],
            'matched': overlap['matched_gt'],
            'unmatched_gt_by_size': {
                'tiny': unmatched_gt_tiny,
                'small': unmatched_gt_small,
                'medium': unmatched_gt_medium,
                'large': unmatched_gt_large
            },
            'unmatched_pred_by_size': {
                'tiny': unmatched_pred_tiny,
                'small': unmatched_pred_small,
                'medium': unmatched_pred_medium,
                'large': unmatched_pred_large
            }
        })
    
    return frame_analyses


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare detection ranges GT vs Detector')
    parser.add_argument('--gt-folder', type=str, default='data/nuscenes_mot_front/val',
                       help='Path to GT folder')
    parser.add_argument('--pred-folder', type=str, required=True,
                       help='Path to prediction folder')
    parser.add_argument('--num-scenes', type=int, default=30,
                       help='Number of scenes to analyze')
    parser.add_argument('--output', type=str, default='range_comparison.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    gt_path = Path(args.gt_folder)
    pred_path = Path(args.pred_folder)
    
    print("="*80)
    print("DETECTION RANGE COMPARISON: GT vs Detector")
    print("="*80)
    print(f"GT folder: {args.gt_folder}")
    print(f"Prediction folder: {args.pred_folder}")
    print(f"Analyzing {args.num_scenes} scenes")
    print()
    
    # Get scene names
    pred_files = sorted([f.stem for f in pred_path.glob('*.txt')])[:args.num_scenes]
    
    print(f"Processing {len(pred_files)} scenes...")
    
    all_results = []
    
    for scene_name in tqdm(pred_files, desc="Analyzing scenes"):
        result = analyze_range_differences(scene_name, gt_path, pred_path)
        if result:
            all_results.extend(result)
    
    # Aggregate statistics
    total_frames = len(all_results)
    total_gt = sum(r['total_gt'] for r in all_results)
    total_pred = sum(r['total_pred'] for r in all_results)
    total_matched = sum(r['matched'] for r in all_results)
    
    # Unmatched by size
    unmatched_gt_by_size = {
        'tiny': sum(r['unmatched_gt_by_size']['tiny'] for r in all_results),
        'small': sum(r['unmatched_gt_by_size']['small'] for r in all_results),
        'medium': sum(r['unmatched_gt_by_size']['medium'] for r in all_results),
        'large': sum(r['unmatched_gt_by_size']['large'] for r in all_results)
    }
    
    unmatched_pred_by_size = {
        'tiny': sum(r['unmatched_pred_by_size']['tiny'] for r in all_results),
        'small': sum(r['unmatched_pred_by_size']['small'] for r in all_results),
        'medium': sum(r['unmatched_pred_by_size']['medium'] for r in all_results),
        'large': sum(r['unmatched_pred_by_size']['large'] for r in all_results)
    }
    
    total_unmatched_gt = sum(unmatched_gt_by_size.values())
    total_unmatched_pred = sum(unmatched_pred_by_size.values())
    
    # Print results
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"Frames analyzed: {total_frames}")
    print(f"Total GT detections: {total_gt:,}")
    print(f"Total Detector predictions: {total_pred:,}")
    print(f"Matched (IoU >= 0.1): {total_matched:,} ({100*total_matched/total_gt:.1f}% of GT)")
    print(f"Unmatched GT: {total_unmatched_gt:,} ({100*total_unmatched_gt/total_gt:.1f}%)")
    print(f"Unmatched Predictions: {total_unmatched_pred:,} ({100*total_unmatched_pred/total_pred:.1f}%)")
    
    print("\n" + "="*80)
    print("UNMATCHED GT OBJECTS (Missing from Detector)")
    print("="*80)
    print("These are objects in GT that detector MISSED:")
    print()
    for size, count in unmatched_gt_by_size.items():
        pct = 100 * count / total_unmatched_gt if total_unmatched_gt > 0 else 0
        print(f"  {size.capitalize():10s}: {count:6,} ({pct:5.1f}%)")
    
    print("\n" + "="*80)
    print("UNMATCHED PREDICTIONS (Not in GT)")
    print("="*80)
    print("These are objects detector PREDICTED but NOT in GT:")
    print()
    for size, count in unmatched_pred_by_size.items():
        pct = 100 * count / total_unmatched_pred if total_unmatched_pred > 0 else 0
        print(f"  {size.capitalize():10s}: {count:6,} ({pct:5.1f}%)")
    
    print("\n" + "="*80)
    print("RANGE ANALYSIS")
    print("="*80)
    
    # Analyze if detector predicts more far or near objects
    gt_far_ratio = (unmatched_gt_by_size['tiny'] + unmatched_gt_by_size['small']) / total_unmatched_gt if total_unmatched_gt > 0 else 0
    pred_far_ratio = (unmatched_pred_by_size['tiny'] + unmatched_pred_by_size['small']) / total_unmatched_pred if total_unmatched_pred > 0 else 0
    
    gt_near_ratio = (unmatched_gt_by_size['large']) / total_unmatched_gt if total_unmatched_gt > 0 else 0
    pred_near_ratio = (unmatched_pred_by_size['large']) / total_unmatched_pred if total_unmatched_pred > 0 else 0
    
    print(f"\n📏 DISTANCE RANGE COMPARISON:")
    print(f"\nUnmatched GT objects (detector missed):")
    print(f"  Far objects (tiny+small): {100*gt_far_ratio:.1f}%")
    print(f"  Near objects (large):     {100*gt_near_ratio:.1f}%")
    
    print(f"\nUnmatched Predictions (not in GT):")
    print(f"  Far objects (tiny+small): {100*pred_far_ratio:.1f}%")
    print(f"  Near objects (large):     {100*pred_near_ratio:.1f}%")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if total_unmatched_pred < 0.1 * total_pred:
        print("✅ Detector has LOW false positive rate (<10%)")
        print("   → Detector does NOT predict objects outside GT range")
    else:
        print("⚠️  Detector has MODERATE-HIGH false positive rate (>10%)")
        print(f"   → {total_unmatched_pred:,} predictions not in GT")
    
    if pred_far_ratio > 0.7:
        print("\n⚠️  Most unmatched predictions are FAR objects (tiny+small)")
        print("   → Detector may be predicting objects beyond GT annotation range")
        print("   → This could indicate range mismatch")
    elif pred_near_ratio > 0.5:
        print("\n⚠️  Many unmatched predictions are NEAR objects (large)")
        print("   → Detector may be predicting partially occluded objects")
        print("   → Or GT has stricter visibility filtering")
    else:
        print("\n✅ Unmatched predictions are distributed across all sizes")
        print("   → No clear evidence of range mismatch")
    
    if gt_far_ratio > 0.7:
        print(f"\n⚠️  Detector misses MANY far objects ({100*gt_far_ratio:.1f}% of unmatched GT)")
        print("   → Detector has difficulty with small/distant objects")
        print("   → This is a RECALL issue, not a range mismatch")
    else:
        print(f"\n✅ Detector misses objects across all ranges")
        print("   → Not specifically missing far objects")
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT: GT vs Detector Range")
    print("="*80)
    
    if total_unmatched_pred < 0.15 * total_pred and pred_far_ratio < 0.6:
        print("✅ SAME RANGE: GT and Detector operate at similar distances")
        print("   - Detector does not over-predict far objects")
        print("   - False positive rate is low")
        print("   - No evidence of range mismatch")
    elif pred_far_ratio > 0.7 and total_unmatched_pred > 0.2 * total_pred:
        print("⚠️  DIFFERENT RANGE: Detector may predict FARTHER than GT")
        print("   - Many unmatched predictions are small (far) objects")
        print("   - Consider filtering small detections (<1000 px²)")
    else:
        print("⚙️  MIXED RESULTS: Some evidence of range differences")
        print("   - Review unmatched predictions manually")
        print("   - May need size-based filtering")
    
    # Save results
    summary = {
        'total_frames': total_frames,
        'total_gt': total_gt,
        'total_pred': total_pred,
        'matched': total_matched,
        'unmatched_gt': total_unmatched_gt,
        'unmatched_pred': total_unmatched_pred,
        'unmatched_gt_by_size': unmatched_gt_by_size,
        'unmatched_pred_by_size': unmatched_pred_by_size,
        'analysis': {
            'gt_far_ratio': float(gt_far_ratio),
            'pred_far_ratio': float(pred_far_ratio),
            'gt_near_ratio': float(gt_near_ratio),
            'pred_near_ratio': float(pred_near_ratio)
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output}")
    print("="*80)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Analyze visibility distribution in GT and compare with detector predictions.

This script helps answer the professor's question about how GT handles
distance/visibility vs detector predictions.
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


def analyze_gt_visibility(gt_folder: str):
    """
    Analyze visibility distribution in GT annotations.
    
    Returns:
        dict with visibility counts, bbox statistics per visibility level
    """
    gt_path = Path(gt_folder)
    
    visibility_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    bbox_areas_per_visibility = {1: [], 2: [], 3: [], 4: []}
    total_annotations = 0
    
    # Process all scene GT files
    scene_folders = sorted([d for d in gt_path.iterdir() if d.is_dir()])
    
    print(f"Analyzing {len(scene_folders)} scenes...")
    
    for scene_folder in tqdm(scene_folders, desc="Processing GT files"):
        gt_file = scene_folder / 'gt' / 'gt.txt'
        
        if not gt_file.exists():
            continue
        
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 9:
                    continue
                
                # Parse GT format: frame,id,x,y,w,h,conf,class,visibility
                try:
                    w = float(parts[4])
                    h = float(parts[5])
                    visibility = int(parts[8])
                    
                    if visibility not in [1, 2, 3, 4]:
                        continue
                    
                    bbox_area = w * h
                    
                    visibility_counts[visibility] += 1
                    bbox_areas_per_visibility[visibility].append(bbox_area)
                    total_annotations += 1
                
                except (ValueError, IndexError):
                    continue
    
    # Calculate statistics
    stats = {
        'total_annotations': total_annotations,
        'visibility_distribution': {},
        'bbox_area_stats': {}
    }
    
    for vis_level in [1, 2, 3, 4]:
        count = visibility_counts[vis_level]
        percentage = 100.0 * count / total_annotations if total_annotations > 0 else 0
        
        stats['visibility_distribution'][vis_level] = {
            'count': count,
            'percentage': percentage
        }
        
        if bbox_areas_per_visibility[vis_level]:
            areas = bbox_areas_per_visibility[vis_level]
            stats['bbox_area_stats'][vis_level] = {
                'mean': float(np.mean(areas)),
                'median': float(np.median(areas)),
                'std': float(np.std(areas)),
                'min': float(np.min(areas)),
                'max': float(np.max(areas)),
                'percentile_25': float(np.percentile(areas, 25)),
                'percentile_75': float(np.percentile(areas, 75))
            }
    
    return stats


def compare_detector_vs_gt(pred_folder: str, gt_folder: str, sample_scenes: int = 10):
    """
    Compare number of detections from detector vs GT per frame.
    
    This helps identify if detector predicts significantly more objects
    than GT (indicating predictions outside GT range).
    """
    pred_path = Path(pred_folder)
    gt_path = Path(gt_folder)
    
    comparisons = []
    
    # Sample random scenes
    pred_scenes = sorted([d.stem for d in pred_path.glob('*.txt')])[:sample_scenes]
    
    print(f"\nComparing detector vs GT for {len(pred_scenes)} scenes...")
    
    for scene_name in tqdm(pred_scenes, desc="Comparing"):
        pred_file = pred_path / f'{scene_name}.txt'
        gt_file = gt_path / scene_name / 'gt' / 'gt.txt'
        
        if not pred_file.exists() or not gt_file.exists():
            continue
        
        # Count detections per frame
        pred_counts = defaultdict(int)
        gt_counts = defaultdict(int)
        
        with open(pred_file, 'r') as f:
            for line in f:
                frame_id = int(line.split(',')[0])
                pred_counts[frame_id] += 1
        
        with open(gt_file, 'r') as f:
            for line in f:
                frame_id = int(line.split(',')[0])
                gt_counts[frame_id] += 1
        
        # Compare frame by frame
        all_frames = set(pred_counts.keys()) | set(gt_counts.keys())
        
        for frame_id in all_frames:
            pred_count = pred_counts.get(frame_id, 0)
            gt_count = gt_counts.get(frame_id, 0)
            
            comparisons.append({
                'scene': scene_name,
                'frame': frame_id,
                'pred_count': pred_count,
                'gt_count': gt_count,
                'diff': pred_count - gt_count
            })
    
    # Calculate statistics
    diffs = [c['diff'] for c in comparisons]
    pred_counts_all = [c['pred_count'] for c in comparisons]
    gt_counts_all = [c['gt_count'] for c in comparisons]
    
    comparison_stats = {
        'num_frames': len(comparisons),
        'avg_pred_per_frame': float(np.mean(pred_counts_all)),
        'avg_gt_per_frame': float(np.mean(gt_counts_all)),
        'avg_diff': float(np.mean(diffs)),
        'median_diff': float(np.median(diffs)),
        'std_diff': float(np.std(diffs)),
        'frames_with_more_preds': sum(1 for d in diffs if d > 0),
        'frames_with_more_gt': sum(1 for d in diffs if d < 0),
        'frames_equal': sum(1 for d in diffs if d == 0)
    }
    
    return comparison_stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze GT visibility distribution')
    parser.add_argument('--gt-folder', type=str, default='data/nuscenes_mot_front/val',
                       help='Path to GT folder')
    parser.add_argument('--pred-folder', type=str, default=None,
                       help='Path to prediction folder (optional, for comparison)')
    parser.add_argument('--output', type=str, default='visibility_analysis.json',
                       help='Output JSON file')
    parser.add_argument('--sample-scenes', type=int, default=10,
                       help='Number of scenes to sample for detector comparison')
    
    args = parser.parse_args()
    
    print("="*80)
    print("GT VISIBILITY DISTRIBUTION ANALYSIS")
    print("="*80)
    print(f"GT folder: {args.gt_folder}")
    if args.pred_folder:
        print(f"Prediction folder: {args.pred_folder}")
    print()
    
    # Analyze GT visibility
    gt_stats = analyze_gt_visibility(args.gt_folder)
    
    print("\n" + "="*80)
    print("VISIBILITY DISTRIBUTION IN GT")
    print("="*80)
    print(f"Total annotations: {gt_stats['total_annotations']}")
    print()
    
    for vis_level in [1, 2, 3, 4]:
        vis_info = gt_stats['visibility_distribution'][vis_level]
        vis_ranges = {
            1: "0-40% visible (very occluded)",
            2: "40-60% visible (medium occlusion)",
            3: "60-80% visible (low occlusion)",
            4: "80-100% visible (almost fully visible)"
        }
        
        print(f"Visibility {vis_level} ({vis_ranges[vis_level]}):")
        print(f"  Count: {vis_info['count']:,}")
        print(f"  Percentage: {vis_info['percentage']:.2f}%")
        
        if vis_level in gt_stats['bbox_area_stats']:
            bbox_stats = gt_stats['bbox_area_stats'][vis_level]
            print(f"  BBox area - Mean: {bbox_stats['mean']:.1f}, Median: {bbox_stats['median']:.1f}")
            print(f"  BBox area - Range: [{bbox_stats['min']:.1f}, {bbox_stats['max']:.1f}]")
        print()
    
    results = {'gt_stats': gt_stats}
    
    # Compare with detector predictions if provided
    if args.pred_folder:
        comparison_stats = compare_detector_vs_gt(
            args.pred_folder, args.gt_folder, args.sample_scenes
        )
        
        results['detector_comparison'] = comparison_stats
        
        print("="*80)
        print("DETECTOR vs GT COMPARISON")
        print("="*80)
        print(f"Frames analyzed: {comparison_stats['num_frames']}")
        print(f"Avg predictions per frame: {comparison_stats['avg_pred_per_frame']:.2f}")
        print(f"Avg GT per frame: {comparison_stats['avg_gt_per_frame']:.2f}")
        print(f"Avg difference (pred - GT): {comparison_stats['avg_diff']:.2f}")
        print(f"Median difference: {comparison_stats['median_diff']:.2f}")
        print()
        print(f"Frames with more predictions: {comparison_stats['frames_with_more_preds']} "
              f"({100*comparison_stats['frames_with_more_preds']/comparison_stats['num_frames']:.1f}%)")
        print(f"Frames with more GT: {comparison_stats['frames_with_more_gt']} "
              f"({100*comparison_stats['frames_with_more_gt']/comparison_stats['num_frames']:.1f}%)")
        print(f"Frames equal: {comparison_stats['frames_equal']}")
        print()
        
        if comparison_stats['avg_diff'] > 2:
            print("⚠️  WARNING: Detector predicts significantly MORE objects than GT!")
            print("   This suggests detector may predict objects outside GT range/visibility.")
            print("   Consider adding post-detection filtering.")
        elif comparison_stats['avg_diff'] < -2:
            print("⚠️  WARNING: GT has significantly MORE objects than detector predictions!")
            print("   This suggests detector misses many objects (low recall).")
        else:
            print("✓ Detector and GT have similar object counts per frame.")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print("="*80)
    print(f"✓ Analysis complete! Results saved to: {args.output}")
    print("="*80)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Analizza ID switches per classe per capire dove intervenire
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import json

# NuScenes 7 classes mapping
CLASS_NAMES = {
    1: 'car',
    2: 'truck', 
    3: 'bus',
    4: 'trailer',
    5: 'construction_vehicle',
    6: 'pedestrian',
    7: 'motorcycle',
    8: 'bicycle',
    9: 'traffic_cone',
    10: 'barrier'
}

def load_mot_file(filepath):
    """Load MOT format file"""
    if not os.path.exists(filepath):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(filepath, header=None, 
                        names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis', 'unused'])
        return df
    except:
        return pd.DataFrame()


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x,y,w,h]"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to corners
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2
    
    # Intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def detect_id_switches(gt_file, pred_file, iou_threshold=0.5):
    """
    Detect ID switches per class with detailed info
    Returns: list of switches with frame, GT class, etc.
    """
    gt_df = load_mot_file(gt_file)
    pred_df = load_mot_file(pred_file)
    
    if gt_df.empty or pred_df.empty:
        return []
    
    switches = []
    
    # Track GT_ID -> Pred_ID mapping per frame
    gt_to_pred_history = defaultdict(list)  # gt_id -> [(frame, pred_id, iou, class)]
    
    frames = sorted(gt_df['frame'].unique())
    
    for frame in frames:
        gt_frame = gt_df[gt_df['frame'] == frame]
        pred_frame = pred_df[pred_df['frame'] == frame]
        
        if gt_frame.empty or pred_frame.empty:
            continue
        
        # Match GT to predictions via IoU
        for _, gt_row in gt_frame.iterrows():
            gt_id = gt_row['id']
            gt_class = int(gt_row['class'])
            gt_box = [gt_row['x'], gt_row['y'], gt_row['w'], gt_row['h']]
            
            best_iou = 0
            best_pred_id = None
            
            for _, pred_row in pred_frame.iterrows():
                pred_box = [pred_row['x'], pred_row['y'], pred_row['w'], pred_row['h']]
                iou = compute_iou(gt_box, pred_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_pred_id = pred_row['id']
            
            if best_iou >= iou_threshold and best_pred_id is not None:
                gt_to_pred_history[gt_id].append((frame, best_pred_id, best_iou, gt_class))
    
    # Detect switches: quando stesso GT_ID è matchato a Pred_ID diversi
    for gt_id, history in gt_to_pred_history.items():
        if len(history) < 2:
            continue
        
        prev_pred_id = history[0][1]
        for i in range(1, len(history)):
            frame, pred_id, iou, gt_class = history[i]
            
            if pred_id != prev_pred_id:
                # ID SWITCH!
                switches.append({
                    'frame': frame,
                    'gt_id': gt_id,
                    'gt_class': gt_class,
                    'class_name': CLASS_NAMES.get(gt_class, f'class_{gt_class}'),
                    'old_pred_id': prev_pred_id,
                    'new_pred_id': pred_id,
                    'iou': iou,
                    'prev_frame': history[i-1][0],
                    'gap': frame - history[i-1][0]
                })
            
            prev_pred_id = pred_id
    
    return switches


def analyze_dataset(gt_folder, pred_folder):
    """Analyze entire dataset"""
    gt_path = Path(gt_folder)
    pred_path = Path(pred_folder)
    
    # Find all scenes
    pred_files = list(pred_path.glob('*.txt'))
    scene_names = [f.stem.replace('_CAM_FRONT', '') for f in pred_files]
    
    print(f"\n🔍 Analyzing {len(scene_names)} scenes...\n")
    
    all_switches = []
    scene_stats = []
    
    for scene_name in tqdm(scene_names, desc="Processing scenes"):
        # Find GT and prediction files
        scene_base = scene_name.replace('_CAM_FRONT', '')
        gt_file = gt_path / scene_base / 'gt' / 'gt.txt'
        pred_file = pred_path / f'{scene_name}_CAM_FRONT.txt'
        
        if not gt_file.exists():
            pred_file = pred_path / f'{scene_name}.txt'
        
        if not gt_file.exists() or not pred_file.exists():
            continue
        
        switches = detect_id_switches(gt_file, pred_file)
        all_switches.extend(switches)
        
        # Per-scene stats
        if switches:
            scene_stats.append({
                'scene': scene_name,
                'num_switches': len(switches),
                'classes': list(set([s['class_name'] for s in switches]))
            })
    
    return all_switches, scene_stats


def print_analysis(switches, scene_stats):
    """Print comprehensive analysis"""
    
    print("="*90)
    print("📊 ID SWITCHES ANALYSIS BY CLASS")
    print("="*90)
    
    # Overall stats
    total_switches = len(switches)
    print(f"\n🔢 Total ID Switches: {total_switches}")
    
    if total_switches == 0:
        print("\n✅ No ID switches detected!")
        return
    
    # By class
    print("\n📦 SWITCHES BY CLASS:")
    print("-"*90)
    class_counts = defaultdict(int)
    class_gaps = defaultdict(list)
    class_ious = defaultdict(list)
    
    for s in switches:
        class_name = s['class_name']
        class_counts[class_name] += 1
        class_gaps[class_name].append(s['gap'])
        class_ious[class_name].append(s['iou'])
    
    # Sort by count
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Class':<25} {'Count':>8} {'% Total':>10} {'Avg Gap':>10} {'Avg IoU':>10}")
    print("-"*90)
    
    for class_name, count in sorted_classes:
        pct = 100 * count / total_switches
        avg_gap = np.mean(class_gaps[class_name])
        avg_iou = np.mean(class_ious[class_name])
        print(f"{class_name:<25} {count:>8} {pct:>9.1f}% {avg_gap:>9.1f} {avg_iou:>9.3f}")
    
    # Gap distribution
    print("\n⏱️  FRAME GAP DISTRIBUTION:")
    print("-"*90)
    all_gaps = [s['gap'] for s in switches]
    gap_ranges = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 30), (31, float('inf'))]
    
    for min_gap, max_gap in gap_ranges:
        if max_gap == float('inf'):
            label = f"> {min_gap-1} frames"
            count = sum(1 for g in all_gaps if g >= min_gap)
        else:
            label = f"{min_gap}-{max_gap} frames" if min_gap != max_gap else f"{min_gap} frame"
            count = sum(1 for g in all_gaps if min_gap <= g <= max_gap)
        
        pct = 100 * count / total_switches
        print(f"  {label:<20} {count:>6} ({pct:>5.1f}%)")
    
    # IoU distribution
    print("\n🎯 IoU AT SWITCH POINT:")
    print("-"*90)
    all_ious = [s['iou'] for s in switches]
    iou_ranges = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    
    for min_iou, max_iou in iou_ranges:
        count = sum(1 for i in all_ious if min_iou <= i < max_iou)
        pct = 100 * count / total_switches
        print(f"  {min_iou:.1f}-{max_iou:.1f}:  {count:>6} ({pct:>5.1f}%)")
    
    # Worst scenes
    print("\n🔥 TOP 10 SCENES WITH MOST SWITCHES:")
    print("-"*90)
    sorted_scenes = sorted(scene_stats, key=lambda x: x['num_switches'], reverse=True)[:10]
    
    for i, scene in enumerate(sorted_scenes, 1):
        classes_str = ', '.join(scene['classes'][:3])
        if len(scene['classes']) > 3:
            classes_str += f" (+{len(scene['classes'])-3} more)"
        print(f"  {i:2d}. {scene['scene']:<30} {scene['num_switches']:>4} switches  [{classes_str}]")
    
    # Recommendations
    print("\n" + "="*90)
    print("💡 RECOMMENDATIONS:")
    print("="*90)
    
    # Check if one class dominates
    top_class, top_count = sorted_classes[0]
    top_pct = 100 * top_count / total_switches
    
    if top_pct > 60:
        print(f"\n⚠️  {top_class} causes {top_pct:.1f}% of switches!")
        print(f"   → Consider class-specific match_thresh for {top_class}")
        avg_iou = np.mean(class_ious[top_class])
        suggested_thresh = max(0.5, avg_iou - 0.05)
        print(f"   → Suggested match_thresh for {top_class}: {suggested_thresh:.2f}")
    
    # Check gap patterns
    immediate_switches = sum(1 for g in all_gaps if g == 1)
    immediate_pct = 100 * immediate_switches / total_switches
    
    if immediate_pct > 40:
        print(f"\n⚠️  {immediate_pct:.1f}% switches happen immediately (gap=1)!")
        print(f"   → Problem: association logic, not detection gaps")
        print(f"   → Consider: lower match_thresh or better re-ID")
    
    long_gaps = sum(1 for g in all_gaps if g > 10)
    long_pct = 100 * long_gaps / total_switches
    
    if long_pct > 30:
        print(f"\n⚠️  {long_pct:.1f}% switches after long gaps (>10 frames)!")
        print(f"   → Problem: detection gaps / track lost")
        print(f"   → Consider: increase max_age or improve detector")
    
    # IoU patterns
    low_iou = sum(1 for i in all_ious if i < 0.7)
    low_iou_pct = 100 * low_iou / total_switches
    
    if low_iou_pct > 40:
        print(f"\n⚠️  {low_iou_pct:.1f}% switches at low IoU (<0.7)!")
        print(f"   → Current match_thresh might be too low")
        print(f"   → Or: poor detection quality / object deformation")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze ID switches by class')
    parser.add_argument('--gt-folder', type=str, 
                       default='data/nuscenes_mot_front_7classes/val',
                       help='GT folder')
    parser.add_argument('--pred-folder', type=str,
                       default='results/POST_FINETUNED/trackssm_finetuned/data',
                       help='Predictions folder')
    parser.add_argument('--output', type=str,
                       default='results/idsw_analysis.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    # Run analysis
    switches, scene_stats = analyze_dataset(args.gt_folder, args.pred_folder)
    
    # Print results
    print_analysis(switches, scene_stats)
    
    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'total_switches': len(switches),
            'switches': switches,
            'scene_stats': scene_stats
        }, f, indent=2)
    
    print(f"\n✅ Detailed results saved to: {output_path}")


if __name__ == '__main__':
    main()

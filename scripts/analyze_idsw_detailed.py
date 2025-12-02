#!/usr/bin/env python3
"""
Analizza IDSW in dettaglio usando motmetrics per trovare pattern
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import motmetrics as mm
from tqdm import tqdm
import json

# Class mapping
CLASS_NAMES = {
    1: 'car', 2: 'truck', 3: 'bus', 4: 'trailer', 5: 'construction_vehicle',
    6: 'pedestrian', 7: 'motorcycle', 8: 'bicycle', 9: 'traffic_cone', 10: 'barrier'
}

def load_mot_file(filepath):
    if not os.path.exists(filepath):
        return pd.DataFrame()
    try:
        df = pd.read_csv(filepath, header=None, 
                        names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis', 'unused'])
        return df
    except:
        return pd.DataFrame()

def bbox_to_tlbr(bbox):
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

def compute_iou_matrix(bboxes1, bboxes2):
    n, m = len(bboxes1), len(bboxes2)
    if n == 0 or m == 0:
        return np.zeros((n, m))
    
    ious = np.zeros((n, m))
    for i, box1 in enumerate(bboxes1):
        for j, box2 in enumerate(bboxes2):
            x1, y1, x2, y2 = bbox_to_tlbr(box1)
            x3, y3, x4, y4 = bbox_to_tlbr(box2)
            
            xi1, yi1 = max(x1, x3), max(y1, y3)
            xi2, yi2 = min(x2, x4), min(y2, y4)
            
            if xi2 > xi1 and yi2 > yi1:
                inter = (xi2 - xi1) * (yi2 - yi1)
                union = (x2-x1)*(y2-y1) + (x4-x3)*(y4-y3) - inter
                ious[i, j] = inter / union if union > 0 else 0
    
    return ious

def evaluate_scene_detailed(gt_file, pred_file, iou_threshold=0.5):
    """Evaluate scene and track ID switches with class info"""
    gt_df = load_mot_file(gt_file)
    pred_df = load_mot_file(pred_file)
    
    if gt_df.empty or pred_df.empty:
        return None, None
    
    # Create accumulator (auto_id=False to allow manual frameid)
    acc = mm.MOTAccumulator(auto_id=False)
    
    # Track class info per GT id
    gt_class_map = {}
    
    frames = sorted(set(gt_df['frame'].unique()) | set(pred_df['frame'].unique()))
    
    for frame in frames:
        gt_frame = gt_df[gt_df['frame'] == frame]
        pred_frame = pred_df[pred_df['frame'] == frame]
        
        # Update class map
        for _, row in gt_frame.iterrows():
            gt_class_map[row['id']] = int(row['class'])
        
        if gt_frame.empty:
            if not pred_frame.empty:
                acc.update([], list(pred_frame['id']), [], frameid=frame)
            continue
        
        gt_ids = list(gt_frame['id'])
        gt_bboxes = gt_frame[['x', 'y', 'w', 'h']].values
        
        if pred_frame.empty:
            acc.update(gt_ids, [], np.zeros((len(gt_ids), 0)), frameid=frame)
            continue
        
        pred_ids = list(pred_frame['id'])
        pred_bboxes = pred_frame[['x', 'y', 'w', 'h']].values
        
        iou_matrix = compute_iou_matrix(gt_bboxes, pred_bboxes)
        distance_matrix = 1.0 - iou_matrix
        distance_matrix[iou_matrix < iou_threshold] = np.nan
        
        acc.update(gt_ids, pred_ids, distance_matrix, frameid=frame)
    
    return acc, gt_class_map

def analyze_dataset(gt_folder, pred_folder):
    """Analyze dataset and extract IDSW info by class"""
    gt_path = Path(gt_folder)
    pred_path = Path(pred_folder)
    
    pred_files = list(pred_path.glob('*.txt'))
    scene_names = [f.stem for f in pred_files]  # Keep full name like scene-0003_CAM_FRONT
    
    print(f"\n🔍 Analyzing {len(scene_names)} scenes with motmetrics...\n")
    
    accumulators = []
    class_maps = []
    scene_idsw = []
    
    for scene_name in tqdm(scene_names, desc="Processing"):
        # GT is stored as scene-0003_CAM_FRONT/gt/gt.txt
        gt_file = gt_path / scene_name / 'gt' / 'gt.txt'
        # Pred is stored as scene-0003_CAM_FRONT.txt
        pred_file = pred_path / f'{scene_name}.txt'
        
        if not gt_file.exists() or not pred_file.exists():
            continue
        
        acc, class_map = evaluate_scene_detailed(gt_file, pred_file)
        if acc is not None:
            accumulators.append(acc)
            class_maps.append(class_map)
            
            # Get IDSW for this scene
            mh = mm.metrics.create()
            summary = mh.compute(acc, metrics=['num_switches'], name=scene_name)
            idsw = summary['num_switches'].values[0]
            
            if idsw > 0:
                scene_idsw.append({
                    'scene': scene_name,
                    'idsw': int(idsw),
                    'class_map': class_map
                })
    
    # Compute overall metrics
    if not accumulators:
        print("\n⚠️  No valid scenes processed!")
        return {'total_idsw': 0, 'class_idsw': {}, 'class_scenes': {}, 'scene_idsw': []}
    
    mh = mm.metrics.create()
    summary = mh.compute_many(accumulators, metrics=['num_switches'], names=[f'scene_{i}' for i in range(len(accumulators))])
    
    total_idsw = summary['num_switches'].sum()
    
    # Aggregate by class
    class_idsw = defaultdict(int)
    class_scenes = defaultdict(list)
    
    for scene_info in scene_idsw:
        scene = scene_info['scene']
        idsw = scene_info['idsw']
        class_map = scene_info['class_map']
        
        # Distribute IDSW across classes present in scene (rough approximation)
        classes_in_scene = set(class_map.values())
        idsw_per_class = idsw / len(classes_in_scene) if classes_in_scene else 0
        
        for cls in classes_in_scene:
            class_name = CLASS_NAMES.get(cls, f'class_{cls}')
            class_idsw[class_name] += idsw_per_class
            class_scenes[class_name].append(scene)
    
    return {
        'total_idsw': int(total_idsw),
        'class_idsw': dict(class_idsw),
        'class_scenes': {k: len(set(v)) for k, v in class_scenes.items()},
        'scene_idsw': scene_idsw
    }

def print_results(results):
    """Print analysis results"""
    print("="*90)
    print("📊 ID SWITCHES ANALYSIS (motmetrics-based)")
    print("="*90)
    
    total = results['total_idsw']
    print(f"\n🔢 Total ID Switches: {total}")
    
    if total == 0:
        print("\n✅ No ID switches detected!")
        return
    
    print("\n📦 ESTIMATED IDSW BY CLASS:")
    print("-"*90)
    print(f"{'Class':<25} {'Est. IDSW':>12} {'% Total':>10} {'Scenes':>10}")
    print("-"*90)
    
    sorted_classes = sorted(results['class_idsw'].items(), key=lambda x: x[1], reverse=True)
    
    for class_name, idsw in sorted_classes:
        pct = 100 * idsw / total
        scenes = results['class_scenes'].get(class_name, 0)
        print(f"{class_name:<25} {idsw:>12.1f} {pct:>9.1f}% {scenes:>10}")
    
    print("\n💡 RECOMMENDATIONS:")
    print("-"*90)
    
    # Check if one class dominates
    if sorted_classes:
        top_class, top_idsw = sorted_classes[0]
        top_pct = 100 * top_idsw / total
        
        if top_pct > 50:
            print(f"\n⚠️  {top_class} is responsible for ~{top_pct:.1f}% of IDSW!")
            print(f"   → Class-specific optimization recommended")
            print(f"   → Consider lowering match_thresh for {top_class}")
        else:
            print(f"\n✅ IDSW is distributed across classes")
            print(f"   → Class-specific thresholds may NOT help much")
            print(f"   → Focus on general improvements (detector, max_age, etc.)")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-folder', default='data/nuscenes_mot_front/val')
    parser.add_argument('--pred-folder', default='results/POST_FINETUNED/trackssm_finetuned/data')
    parser.add_argument('--output', default='results/IDSW_ANALYSIS/detailed_analysis.json')
    args = parser.parse_args()
    
    results = analyze_dataset(args.gt_folder, args.pred_folder)
    print_results(results)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")

if __name__ == '__main__':
    main()

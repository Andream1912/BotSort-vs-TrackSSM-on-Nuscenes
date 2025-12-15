#!/usr/bin/env python3
"""
Compute validation metrics for YOLOX checkpoints
This simulates validation loss by evaluating detection quality on validation set
"""

import sys
sys.path.insert(0, 'external/YOLOX')
sys.path.insert(0, 'src')

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import cv2

from detectors.yolox_detector import YOLOXDetector

def load_gt_annotations(scene_path):
    """Load ground truth annotations from gt.txt"""
    gt_file = scene_path / 'gt' / 'gt.txt'
    if not gt_file.exists():
        return []
    
    annotations = []
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 10:
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                conf = float(parts[6])
                class_id = int(parts[7])
                vis = float(parts[8])
                
                annotations.append({
                    'frame_id': frame_id,
                    'bbox': [x, y, w, h],
                    'class_id': class_id,
                    'visibility': vis
                })
    return annotations

def compute_iou(box1, box2):
    """Compute IoU between two boxes [x,y,w,h]"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to corners
    x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
    x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2
    
    # Intersection
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    inter_w = max(0, xi_max - xi_min)
    inter_h = max(0, yi_max - yi_min)
    inter_area = inter_w * inter_h
    
    # Union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    
    return inter_area / union_area

def evaluate_checkpoint(checkpoint_path, val_scenes, conf_thresh=0.3, iou_thresh=0.5, max_scenes=20):
    """Evaluate checkpoint on validation scenes"""
    print(f"\nEvaluating: {checkpoint_path.name}")
    
    # Initialize detector
    detector = YOLOXDetector(
        model_path=str(checkpoint_path),
        test_size=(800, 1440),
        num_classes=7,
        conf_thresh=conf_thresh
    )
    
    all_precisions = []
    all_recalls = []
    
    for scene in tqdm(val_scenes[:max_scenes], desc="Scenes"):
        scene_path = Path(f"data/nuscenes_mot_front/val/{scene}")
        if not scene_path.exists():
            continue
        
        # Load GT
        gt_annotations = load_gt_annotations(scene_path)
        if not gt_annotations:
            continue
        
        # Group GT by frame
        gt_by_frame = {}
        for ann in gt_annotations:
            frame_id = ann['frame_id']
            if frame_id not in gt_by_frame:
                gt_by_frame[frame_id] = []
            gt_by_frame[frame_id].append(ann)
        
        # Evaluate each frame
        img_dir = scene_path / 'img1'
        for img_file in sorted(img_dir.glob('*.jpg')):
            frame_id = int(img_file.stem)
            
            # Load image
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            # Detect
            detections = detector.detect(img)
            
            # Get GT for this frame
            gt_boxes = gt_by_frame.get(frame_id, [])
            
            if len(gt_boxes) == 0:
                continue
            
            # Match detections to GT
            matched_gt = set()
            tp = 0
            
            for det in detections:
                det_box = [det['bbox'][0], det['bbox'][1], 
                          det['bbox'][2] - det['bbox'][0],
                          det['bbox'][3] - det['bbox'][1]]
                
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = compute_iou(det_box, gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_thresh:
                    matched_gt.add(best_gt_idx)
                    tp += 1
            
            # Compute precision and recall for this frame
            precision = tp / len(detections) if len(detections) > 0 else 0
            recall = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0
            
            all_precisions.append(precision)
            all_recalls.append(recall)
    
    # Compute average metrics
    avg_precision = np.mean(all_precisions) if all_precisions else 0
    avg_recall = np.mean(all_recalls) if all_recalls else 0
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    return {
        'precision': avg_precision * 100,
        'recall': avg_recall * 100,
        'f1': f1_score * 100,
        'num_frames': len(all_precisions)
    }

def plot_train_val_curves(train_losses, val_metrics, output_file):
    """Create train/val plot"""
    
    epochs = sorted(train_losses.keys())
    val_epochs = sorted(val_metrics.keys())
    
    train_loss_values = [train_losses[ep] for ep in epochs]
    val_f1_values = [val_metrics[ep]['f1'] for ep in val_epochs]
    val_precision_values = [val_metrics[ep]['precision'] for ep in val_epochs]
    val_recall_values = [val_metrics[ep]['recall'] for ep in val_epochs]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Training Loss
    ax1.plot(epochs, train_loss_values, 'o-', linewidth=2, markersize=8,
            color='#2E86AB', alpha=0.9, label='Training Loss')
    
    # Highlight saved checkpoints
    saved_epochs = [1, 5, 10, 15, 20, 25, 30]
    saved_losses = [train_losses[ep] for ep in saved_epochs if ep in train_losses]
    saved_epochs_actual = [ep for ep in saved_epochs if ep in train_losses]
    
    ax1.scatter(saved_epochs_actual, saved_losses, 
               s=200, marker='*', color='gold', 
               edgecolors='black', linewidths=1.5,
               label='Saved Checkpoints', zorder=5)
    
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_title('Training Loss Curve', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=11)
    
    # Plot 2: Validation Metrics
    ax2.plot(val_epochs, val_f1_values, 'o-', linewidth=2, markersize=8,
            color='#A23B72', alpha=0.9, label='F1 Score')
    ax2.plot(val_epochs, val_precision_values, 's-', linewidth=2, markersize=7,
            color='#F18F01', alpha=0.8, label='Precision')
    ax2.plot(val_epochs, val_recall_values, '^-', linewidth=2, markersize=7,
            color='#06A77D', alpha=0.8, label='Recall')
    
    ax2.scatter(val_epochs, val_f1_values, 
               s=200, marker='*', color='gold', 
               edgecolors='black', linewidths=1.5, zorder=5)
    
    ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Metric Value (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Validation Metrics (Detection Quality)', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='lower right', fontsize=11)
    ax2.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved: {output_file}")

if __name__ == '__main__':
    # Parse training log for losses
    print("Parsing training log...")
    from plot_yolox_training import parse_training_log
    train_losses = parse_training_log('yolox_finetuning/training_stable.log')
    
    # Get validation scenes
    val_dir = Path('data/nuscenes_mot_front/val')
    val_scenes = [d.name for d in val_dir.iterdir() if d.is_dir()]
    print(f"Found {len(val_scenes)} validation scenes")
    
    # Evaluate saved checkpoints
    checkpoints_dir = Path('yolox_finetuning/yolox_l_nuscenes_stable')
    saved_epochs = [1, 5, 10, 15, 20, 25, 30]
    
    val_metrics = {}
    
    for epoch in saved_epochs:
        ckpt_path = checkpoints_dir / f'epoch_{epoch}.pth'
        if not ckpt_path.exists():
            print(f"⚠️  Checkpoint not found: {ckpt_path}")
            continue
        
        metrics = evaluate_checkpoint(ckpt_path, val_scenes, max_scenes=20)
        val_metrics[epoch] = metrics
        
        print(f"Epoch {epoch:2d}: Precision={metrics['precision']:.1f}%, "
              f"Recall={metrics['recall']:.1f}%, F1={metrics['f1']:.1f}% "
              f"({metrics['num_frames']} frames)")
    
    # Save results
    results = {
        'train_losses': {int(k): float(v) for k, v in train_losses.items()},
        'val_metrics': {int(k): v for k, v in val_metrics.items()}
    }
    
    output_json = 'yolox_finetuning/train_val_metrics.json'
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved: {output_json}")
    
    # Plot
    plot_train_val_curves(train_losses, val_metrics, 'yolox_finetuning/train_val_curves.png')
    
    print("\n✓ Train/Val analysis complete!")

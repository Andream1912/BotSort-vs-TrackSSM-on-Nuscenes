#!/usr/bin/env python3
"""Compare multiple YOLOX checkpoints"""
import sys
sys.path.insert(0, 'external/YOLOX')
sys.path.insert(0, 'src')

import cv2
from detectors.yolox_detector import YOLOXDetector

def test_checkpoint(ckpt_path, name):
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"Checkpoint: {ckpt_path}")
    print('='*70)
    
    detector = YOLOXDetector(
        model_path=ckpt_path,
        test_size=(800, 1440),
        num_classes=7,
        conf_thresh=0.01
    )
    
    test_imgs = [
        'data/nuscenes_mot_front/val/scene-0003_CAM_FRONT/img1/000001.jpg',
        'data/nuscenes_mot_front/val/scene-0103_CAM_FRONT/img1/000001.jpg',
        'data/nuscenes_mot_front/val/scene-0655_CAM_FRONT/img1/000001.jpg',
    ]
    
    all_max_confs = []
    total_high = 0
    total_med = 0
    total_dets = 0
    
    for img_path in test_imgs:
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        detections = detector.detect(img)
        scene = img_path.split('/')[-3]
        num_dets = len(detections)
        total_dets += num_dets
        
        if num_dets > 0:
            confs = [d['confidence'] for d in detections]
            max_conf = max(confs)
            avg_conf = sum(confs) / len(confs)
            all_max_confs.append(max_conf)
            
            high = sum(1 for c in confs if c > 0.5)
            med = sum(1 for c in confs if 0.3 < c <= 0.5)
            total_high += high
            total_med += med
            
            print(f"\n{scene}:")
            print(f"  Dets: {num_dets:3d} | Max: {max_conf:.3f} | Avg: {avg_conf:.3f} | >0.5: {high:2d} | 0.3-0.5: {med:2d}")
        else:
            print(f"\n{scene}: 0 detections")
            all_max_confs.append(0.0)
    
    if all_max_confs:
        avg_max = sum(all_max_confs) / len(all_max_confs)
        print(f"\n{'-'*70}")
        print(f"SUMMARY - {name}:")
        print(f"  Total detections: {total_dets} (avg {total_dets/len(test_imgs):.1f} per image)")
        print(f"  High conf (>0.5): {total_high}")
        print(f"  Med conf (0.3-0.5): {total_med}")
        print(f"  Avg max conf: {avg_max:.3f}")
        
        if avg_max > 0.7:
            print(f"  ✅ EXCELLENT")
        elif avg_max > 0.5:
            print(f"  ✅ GOOD")
        elif avg_max > 0.3:
            print(f"  ⚠️  ACCEPTABLE")
        else:
            print(f"  ❌ POOR")
        
        return {
            'name': name,
            'avg_max_conf': avg_max,
            'total_high': total_high,
            'total_med': total_med,
            'total_dets': total_dets
        }
    return None


if __name__ == "__main__":
    results = []
    
    # Test all checkpoints
    checkpoints = [
        ('yolox_finetuning/yolox_l_nuscenes_clean/epoch_5.pth', 'Training 1 - Epoch 5 (LR 0.002)'),
        ('yolox_finetuning/yolox_l_nuscenes_clean/epoch_10.pth', 'Training 1 - Epoch 10 (LR 0.002)'),
        ('yolox_finetuning/yolox_l_nuscenes_clean_v2/epoch_5.pth', 'Training 2 - Epoch 5 (LR 0.001)'),
        ('yolox_finetuning/yolox_l_nuscenes_clean_v2/epoch_10.pth', 'Training 2 - Epoch 10 (LR 0.001)'),
    ]
    
    for ckpt_path, name in checkpoints:
        result = test_checkpoint(ckpt_path, name)
        if result:
            results.append(result)
    
    # Final comparison
    print(f"\n\n{'='*70}")
    print("FINAL COMPARISON:")
    print('='*70)
    print(f"{'Checkpoint':<35} {'Avg Max Conf':>12} {'High(>0.5)':>10} {'Med(0.3-0.5)':>12}")
    print('-'*70)
    
    for r in results:
        print(f"{r['name']:<35} {r['avg_max_conf']:>12.3f} {r['total_high']:>10d} {r['total_med']:>12d}")
    
    if results:
        best = max(results, key=lambda x: x['avg_max_conf'])
        print(f"\n{'='*70}")
        print(f"🏆 BEST: {best['name']}")
        print(f"   Avg max confidence: {best['avg_max_conf']:.3f}")
        print(f"   High confidence detections: {best['total_high']}")
        print('='*70)

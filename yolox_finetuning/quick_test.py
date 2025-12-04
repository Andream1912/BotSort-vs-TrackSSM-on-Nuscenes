#!/usr/bin/env python3
"""Quick test of finetuned YOLOX checkpoint"""
import sys
sys.path.insert(0, 'external/YOLOX')
sys.path.insert(0, 'src')

import cv2
from detectors.yolox_detector import YOLOXDetector

def test_checkpoint(ckpt_path):
    print(f"\nTesting: {ckpt_path}")
    print("=" * 70)
    
    # Initialize detector with finetuned checkpoint
    detector = YOLOXDetector(
        model_path=ckpt_path,
        test_size=(800, 1440),
        num_classes=7,
        conf_thresh=0.01  # Low threshold to see all detections
    )
    
    # Test images
    test_imgs = [
        'data/nuscenes_mot_front/val/scene-0003_CAM_FRONT/img1/000001.jpg',
        'data/nuscenes_mot_front/val/scene-0061_CAM_FRONT/img1/000001.jpg',
        'data/nuscenes_mot_front/val/scene-0103_CAM_FRONT/img1/000001.jpg',
    ]
    
    all_max_confs = []
    total_high = 0
    total_med = 0
    total_dets = 0
    
    for img_path in test_imgs:
        img = cv2.imread(img_path)
        if img is None:
            print(f"{img_path}: Not found")
            continue
        
        # Detect
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
            print(f"  Detections: {num_dets}")
            print(f"  Max conf: {max_conf:.3f}")
            print(f"  Avg conf: {avg_conf:.3f}")
            print(f"  >0.5: {high}, 0.3-0.5: {med}")
        else:
            print(f"\n{scene}: 0 detections")
            all_max_confs.append(0.0)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"  Images tested: {len(test_imgs)}")
    print(f"  Total detections: {total_dets}")
    print(f"  Avg per image: {total_dets/len(test_imgs):.1f}")
    print(f"  High conf (>0.5): {total_high}")
    print(f"  Med conf (0.3-0.5): {total_med}")
    if all_max_confs:
        avg_max = sum(all_max_confs) / len(all_max_confs)
        print(f"  Avg max conf: {avg_max:.3f}")
        
        if avg_max > 0.5:
            print("\n✅ SUCCESS: Avg max confidence > 0.5 (production ready!)")
        elif avg_max > 0.3:
            print("\n⚠️  ACCEPTABLE: Avg max confidence > 0.3 (usable with lower threshold)")
        else:
            print("\n❌ POOR: Avg max confidence < 0.3 (needs more training)")
    print("=" * 70)


if __name__ == "__main__":
    # Test epoch 5 from Training 1
    test_checkpoint('yolox_finetuning/yolox_l_nuscenes_clean/epoch_5.pth')

#!/usr/bin/env python3
import sys
import os
import cv2
import torch
import numpy as np

sys.path.insert(0, 'external/YOLOX')
from yolox.exp import get_exp
from yolox.utils import postprocess

def test_checkpoint(ckpt_path, config_path, test_imgs):
    """Test a YOLOX checkpoint on validation images"""
    
    print(f"\nTesting checkpoint: {ckpt_path}")
    print("=" * 70)
    
    # Load exp config
    exp = get_exp(config_path, None)
    model = exp.get_model()
    
    # Load checkpoint
    print(f"Loading checkpoint...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    model.cuda()
    
    print(f"Model loaded. Testing on {len(test_imgs)} images...\n")
    
    all_results = []
    
    for img_path in test_imgs:
        if not os.path.exists(img_path):
            print(f"{img_path}: Image not found")
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"{img_path}: Failed to load")
            continue
        
        # Preprocess
        img_resized = cv2.resize(img, (1440, 800))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).cuda()
        
        # Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)
        
        scene_name = img_path.split("/")[-3]
        
        if outputs[0] is not None:
            dets = outputs[0].cpu().numpy()
            confidences = dets[:, 4] * dets[:, 5]  # obj_conf * cls_conf
            max_conf = confidences.max()
            num_dets = len(dets)
            avg_conf = confidences.mean()
            
            # Count by confidence threshold
            high_conf = (confidences > 0.5).sum()
            med_conf = ((confidences > 0.3) & (confidences <= 0.5)).sum()
            low_conf = (confidences <= 0.3).sum()
            
            result = {
                'scene': scene_name,
                'num_dets': num_dets,
                'max_conf': max_conf,
                'avg_conf': avg_conf,
                'high_conf': high_conf,
                'med_conf': med_conf,
                'low_conf': low_conf
            }
            all_results.append(result)
            
            print(f"{scene_name}:")
            print(f"  Total detections: {num_dets}")
            print(f"  Max confidence: {max_conf:.3f}")
            print(f"  Avg confidence: {avg_conf:.3f}")
            print(f"  High conf (>0.5): {high_conf}")
            print(f"  Med conf (0.3-0.5): {med_conf}")
            print(f"  Low conf (<0.3): {low_conf}")
        else:
            print(f"{scene_name}: 0 detections")
            all_results.append({
                'scene': scene_name,
                'num_dets': 0,
                'max_conf': 0.0,
                'avg_conf': 0.0,
                'high_conf': 0,
                'med_conf': 0,
                'low_conf': 0
            })
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY:")
    if all_results:
        total_dets = sum(r['num_dets'] for r in all_results)
        total_high = sum(r['high_conf'] for r in all_results)
        total_med = sum(r['med_conf'] for r in all_results)
        avg_max_conf = np.mean([r['max_conf'] for r in all_results])
        
        print(f"  Total images tested: {len(all_results)}")
        print(f"  Total detections: {total_dets}")
        print(f"  Avg detections per image: {total_dets/len(all_results):.1f}")
        print(f"  Total high conf (>0.5): {total_high}")
        print(f"  Total med conf (0.3-0.5): {total_med}")
        print(f"  Average max confidence: {avg_max_conf:.3f}")
        print()
        
        # Success criteria
        if avg_max_conf > 0.5:
            print("✅ EXCELLENT: Average max confidence > 0.5")
        elif avg_max_conf > 0.3:
            print("⚠️  ACCEPTABLE: Average max confidence > 0.3")
        else:
            print("❌ POOR: Average max confidence < 0.3")
    
    print("=" * 70)
    return all_results


if __name__ == "__main__":
    # Test images
    test_imgs = [
        'data/nuscenes_mot_front/val/scene-0003_CAM_FRONT/img1/000001.jpg',
        'data/nuscenes_mot_front/val/scene-0061_CAM_FRONT/img1/000001.jpg',
        'data/nuscenes_mot_front/val/scene-0103_CAM_FRONT/img1/000001.jpg',
        'data/nuscenes_mot_front/val/scene-0655_CAM_FRONT/img1/000001.jpg',
    ]
    
    # Test Training 1 epoch 5
    results = test_checkpoint(
        'yolox_finetuning/yolox_l_nuscenes_clean/epoch_5.pth',
        'yolox_finetuning/configs/yolox_l_nuscenes_clean.py',
        test_imgs
    )

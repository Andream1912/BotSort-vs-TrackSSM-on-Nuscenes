#!/usr/bin/env python3
"""
Run validation on YOLOX fine-tuned checkpoint to get mAP and other metrics.
This will generate the missing validation metrics for thesis documentation.
"""

import sys
import os
from pathlib import Path

# Add YOLOX to path
yolox_path = Path(__file__).parent.parent / "external/YOLOX"
sys.path.insert(0, str(yolox_path))

import torch
from yolox.exp import get_exp
from yolox.data import COCODataset, ValTransform
from yolox.evaluators import COCOEvaluator
from yolox.utils import get_model_info, postprocess

def main():
    print("="*80)
    print("YOLOX VALIDATION - Computing mAP on Validation Set")
    print("="*80)
    
    # Paths
    checkpoint_path = yolox_path / "YOLOX_outputs/yolox_x_nuscenes_7class/latest_ckpt.pth"
    val_ann = Path(__file__).parent.parent / "data/nuscenes_mot_front/annotations/val_7class.json"
    val_img_dir = Path(__file__).parent.parent / "data/nuscenes_mot_front/val"
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 1
    
    if not val_ann.exists():
        print(f"❌ Validation annotations not found: {val_ann}")
        return 1
    
    print(f"✅ Checkpoint: {checkpoint_path}")
    print(f"✅ Validation annotations: {val_ann}")
    print(f"✅ Validation images: {val_img_dir}")
    
    # Load experiment config
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    
    # Get experiment (use the same config as training)
    exp = get_exp(None, "yolox-x")  # Base YOLOX-X config
    exp.num_classes = 7  # NuScenes 7 classes
    
    # Load model
    model = exp.get_model()
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {get_model_info(model, exp.test_size)}")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    print(f"✅ Loaded checkpoint from epoch {ckpt.get('start_epoch', 'unknown')}")
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model.to(device)
    model.eval()
    
    # Prepare dataset
    print("\n" + "="*80)
    print("PREPARING VALIDATION DATASET")
    print("="*80)
    
    valdataset = COCODataset(
        data_dir=str(val_img_dir.parent),
        json_file=str(val_ann.name),
        name="val",
        img_size=exp.test_size,
        preproc=ValTransform(legacy=False),
    )
    
    print(f"✅ Validation dataset loaded: {len(valdataset)} images")
    
    # Create evaluator
    evaluator = COCOEvaluator(
        dataloader=torch.utils.data.DataLoader(
            valdataset,
            batch_size=exp.test_conf.get("batch_size", 8),
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        ),
        img_size=exp.test_size,
        confthre=exp.test_conf.get("conf_thresh", 0.01),
        nmsthre=exp.test_conf.get("nms_thresh", 0.65),
        num_classes=exp.num_classes,
        testdev=False,
    )
    
    # Run evaluation
    print("\n" + "="*80)
    print("RUNNING EVALUATION")
    print("="*80)
    
    with torch.no_grad():
        stats, ap_class = evaluator.evaluate(
            model, False, False, None
        )
    
    # Print results
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    
    if stats is not None:
        print(f"\n📊 COCO Evaluation Metrics:")
        print(f"   mAP (IoU=0.50:0.95): {stats[0]*100:.2f}%")
        print(f"   mAP (IoU=0.50):      {stats[1]*100:.2f}%")
        print(f"   mAP (IoU=0.75):      {stats[2]*100:.2f}%")
        print(f"   mAP (small):         {stats[3]*100:.2f}%")
        print(f"   mAP (medium):        {stats[4]*100:.2f}%")
        print(f"   mAP (large):         {stats[5]*100:.2f}%")
        
        # Save to file
        output_file = Path(__file__).parent / "validation_metrics.txt"
        with open(output_file, 'w') as f:
            f.write("YOLOX Fine-tuned Validation Metrics\n")
            f.write("="*50 + "\n\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Epoch: {ckpt.get('start_epoch', 'unknown')}\n\n")
            f.write("COCO Metrics:\n")
            f.write(f"  mAP (IoU=0.50:0.95): {stats[0]*100:.2f}%\n")
            f.write(f"  mAP (IoU=0.50):      {stats[1]*100:.2f}%\n")
            f.write(f"  mAP (IoU=0.75):      {stats[2]*100:.2f}%\n")
            f.write(f"  mAP (small):         {stats[3]*100:.2f}%\n")
            f.write(f"  mAP (medium):        {stats[4]*100:.2f}%\n")
            f.write(f"  mAP (large):         {stats[5]*100:.2f}%\n")
        
        print(f"\n✅ Validation metrics saved to: {output_file}")
    else:
        print("❌ Evaluation failed!")
        return 1
    
    print("\n" + "="*80)
    print("✅ VALIDATION COMPLETE!")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    exit(main())

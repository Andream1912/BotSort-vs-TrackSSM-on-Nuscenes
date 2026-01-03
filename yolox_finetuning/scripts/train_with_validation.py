#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
YOLOX Training with Validation
Includes validation every N epochs to track mAP and other metrics.
"""

import os
import sys
import torch
import torch.nn as nn
from loguru import logger
import json
from pathlib import Path

# Add YOLOX to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../external/YOLOX'))

from yolox.exp import get_exp
from yolox.data import COCODataset, ValTransform
from yolox.evaluators import COCOEvaluator
from yolox.utils import configure_nccl, get_num_devices


def validate(model, exp, device, epoch):
    """Run validation and return metrics."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Running validation at epoch {epoch}...")
    logger.info(f"{'='*50}")
    
    model.eval()
    
    # Create validation dataset
    val_dataset = COCODataset(
        data_dir=str(Path(exp.data_dir).parent),  # Parent of train/val dirs
        json_file=os.path.basename(exp.val_ann),
        name="val",
        img_size=exp.test_size,
        preproc=ValTransform(legacy=False),
    )
    
    logger.info(f"Validation dataset: {len(val_dataset)} images")
    
    # Create dataloader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=exp.test_conf.get("batch_size", 8),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    
    # Create evaluator
    evaluator = COCOEvaluator(
        dataloader=val_loader,
        img_size=exp.test_size,
        confthre=exp.test_conf.get("conf_thresh", 0.01),
        nmsthre=exp.test_conf.get("nms_thresh", 0.65),
        num_classes=exp.num_classes,
        testdev=False,
    )
    
    # Run evaluation
    with torch.no_grad():
        try:
            stats, ap_class = evaluator.evaluate(model, False, False, None)
            
            if stats is not None:
                metrics = {
                    'epoch': epoch,
                    'mAP_50_95': float(stats[0]),
                    'mAP_50': float(stats[1]),
                    'mAP_75': float(stats[2]),
                    'mAP_small': float(stats[3]),
                    'mAP_medium': float(stats[4]),
                    'mAP_large': float(stats[5]),
                }
                
                logger.info(f"\n📊 Validation Results (Epoch {epoch}):")
                logger.info(f"  mAP@0.50:0.95: {metrics['mAP_50_95']*100:.2f}%")
                logger.info(f"  mAP@0.50:      {metrics['mAP_50']*100:.2f}%")
                logger.info(f"  mAP@0.75:      {metrics['mAP_75']*100:.2f}%")
                logger.info(f"  mAP (small):   {metrics['mAP_small']*100:.2f}%")
                logger.info(f"  mAP (medium):  {metrics['mAP_medium']*100:.2f}%")
                logger.info(f"  mAP (large):   {metrics['mAP_large']*100:.2f}%")
                
                model.train()
                return metrics
            else:
                logger.warning("Validation returned no metrics")
                model.train()
                return None
                
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            import traceback
            traceback.print_exc()
            model.train()
            return None


def main():
    # Config - ULTRA STABLE V3 with Validation
    exp_file = "yolox_finetuning/configs/yolox_l_nuscenes_stable.py"
    ckpt_file = "weights/detectors/yolox_l.pth"
    
    # Validation settings
    VAL_INTERVAL = 5  # Validate every N epochs
    VAL_EPOCHS = [1, 5, 10, 15, 20, 25, 30]  # Also validate at these specific epochs
    
    logger.info(f"Loading config: {exp_file}")
    exp = get_exp(exp_file, None)
    
    # Add validation annotation path to exp
    exp.val_ann = "data/nuscenes_yolox_6cams/annotations/val.json"
    exp.test_conf = {
        'batch_size': 8,
        'conf_thresh': 0.01,
        'nms_thresh': 0.65
    }
    
    logger.info(f"Creating model...")
    model = exp.get_model()
    logger.info(f"Model: {exp.num_classes} classes")
    
    # Load COCO weights
    logger.info(f"Loading COCO checkpoint: {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]
    
    # Filter checkpoint to match our class count
    logger.info(f"Filtering checkpoint for class mismatch...")
    model_state_dict = model.state_dict()
    filtered_ckpt = {}
    skipped = []
    
    for k, v in ckpt.items():
        if k in model_state_dict:
            if v.shape == model_state_dict[k].shape:
                filtered_ckpt[k] = v
            else:
                skipped.append(k)
        else:
            skipped.append(k)
    
    logger.info(f"Loaded {len(filtered_ckpt)} params, skipped {len(skipped)}")
    
    model.load_state_dict(filtered_ckpt, strict=False)
    logger.info(f"✓ Checkpoint loaded")
    
    # Freeze backbone (CSPDarknet + FPN)
    logger.info(f"\n🔒 Freezing backbone (CSPDarknet + FPN)...")
    frozen_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        if 'head' not in name:
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            trainable_params += param.numel()
    
    total_params = frozen_params + trainable_params
    logger.info(f"  ✓ Frozen backbone params: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    logger.info(f"  ✓ Trainable head params: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    logger.info(f"  ✓ Total params: {total_params:,}")
    
    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create dataloader
    logger.info(f"Creating dataloader...")
    train_loader = exp.get_data_loader(
        batch_size=exp.batch_size,
        is_distributed=False,
        no_aug=False,
    )
    
    logger.info(f"Dataset: {len(exp.dataset)} images")
    logger.info(f"Batch size: {exp.batch_size}")
    max_iter = len(exp.dataset) // exp.batch_size
    logger.info(f"Iterations per epoch: {max_iter}")
    
    # Create optimizer (only trainable params)
    logger.info(f"Creating optimizer...")
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter) and v.bias.requires_grad:
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)  # no decay
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter) and v.weight.requires_grad:
            pg1.append(v.weight)  # apply decay
    
    optimizer = torch.optim.SGD(
        pg0, lr=exp.basic_lr_per_img * exp.batch_size, momentum=exp.momentum, nesterov=True
    )
    optimizer.add_param_group({"params": pg1, "weight_decay": exp.weight_decay})
    optimizer.add_param_group({"params": pg2})
    
    logger.info(f"Optimizer: SGD with {len(pg0)} + {len(pg1)} + {len(pg2)} param groups")
    
    # Create scheduler
    lr_scheduler = exp.get_lr_scheduler(exp.basic_lr_per_img * exp.batch_size, max_iter)
    
    # Track validation history
    validation_history = []
    best_map = 0.0
    best_epoch = 0
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Starting ULTRA STABLE training with VALIDATION")
    logger.info(f"Config: V3 - Maximum stability + validation tracking")
    logger.info(f"Epochs: {exp.max_epoch}")
    logger.info(f"Batch size: {exp.batch_size}")
    logger.info(f"LR: {exp.basic_lr_per_img * exp.batch_size:.6f}")
    logger.info(f"Validation: Every {VAL_INTERVAL} epochs + {VAL_EPOCHS}")
    logger.info(f"{'='*50}\n")
    
    model.train()
    
    # Training loop
    for epoch in range(1, exp.max_epoch + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{exp.max_epoch}")
        logger.info(f"{'='*60}")
        
        for iter_idx, (imgs, targets, _, _) in enumerate(train_loader):
            # Stop at max_iter to match epoch definition
            if iter_idx >= max_iter:
                break
                
            iter_num = (epoch - 1) * max_iter + iter_idx
            
            # Update learning rate
            lr = lr_scheduler.update_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            
            # Forward
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            outputs = model(imgs, targets)
            loss = outputs["total_loss"]
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log every 50 iterations
            if (iter_idx + 1) % 50 == 0 or iter_idx == 0:
                loss_iou = outputs["iou_loss"]
                loss_l1 = outputs["l1_loss"]
                loss_conf = outputs["conf_loss"]
                loss_cls = outputs["cls_loss"]
                
                # Convert to float if tensor
                if hasattr(loss_iou, 'item'):
                    loss_iou = loss_iou.item()
                if hasattr(loss_l1, 'item'):
                    loss_l1 = loss_l1.item()
                if hasattr(loss_conf, 'item'):
                    loss_conf = loss_conf.item()
                if hasattr(loss_cls, 'item'):
                    loss_cls = loss_cls.item()
                
                logger.info(
                    f"  iter {iter_idx + 1}/{max_iter} - "
                    f"loss: total: {loss.item():.2f}, "
                    f"iou: {loss_iou:.2f}, "
                    f"l1: {loss_l1:.2f}, "
                    f"conf: {loss_conf:.2f}, "
                    f"cls: {loss_cls:.2f}, "
                    f"lr: {lr:.6f}"
                )
        
        # Save checkpoint
        logger.info(f"\n💾 Saving checkpoint epoch {epoch}...")
        save_dir = os.path.join(exp.output_dir, exp.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        
        ckpt_name = os.path.join(save_dir, f"epoch_{epoch}.pth")
        
        save_dict = {
            "start_epoch": epoch,
            "model": model.state_dict(),
        }
        
        torch.save(save_dict, ckpt_name)
        size_mb = os.path.getsize(ckpt_name) / (1024 * 1024)
        logger.info(f"✓ Saved to {ckpt_name} ({size_mb:.1f} MB)")
        
        # Run validation
        should_validate = (epoch % VAL_INTERVAL == 0) or (epoch in VAL_EPOCHS)
        
        if should_validate:
            metrics = validate(model, exp, device, epoch)
            
            if metrics:
                validation_history.append(metrics)
                
                # Save validation history
                history_file = os.path.join(save_dir, "validation_history.json")
                with open(history_file, 'w') as f:
                    json.dump(validation_history, f, indent=2)
                logger.info(f"✓ Validation history saved to {history_file}")
                
                # Check if best model
                current_map = metrics['mAP_50_95']
                if current_map > best_map:
                    best_map = current_map
                    best_epoch = epoch
                    
                    # Save as best model
                    best_ckpt = os.path.join(save_dir, "best_model.pth")
                    torch.save(save_dict, best_ckpt)
                    logger.info(f"🌟 New best model! mAP: {best_map*100:.2f}% (epoch {best_epoch})")
                    logger.info(f"   Saved to {best_ckpt}")
        
        # Cleanup intermediate checkpoints
        epochs_to_keep = [1, 5, 10, 15, 20, 25, 30]
        if epoch not in epochs_to_keep and epoch != best_epoch:
            try:
                os.remove(ckpt_name)
                logger.info(f"  Removed intermediate checkpoint to save space")
            except:
                pass
    
    logger.info(f"\n{'='*50}")
    logger.info(f"✅ Training completed!")
    logger.info(f"{'='*50}")
    logger.info(f"🌟 Best model: Epoch {best_epoch}, mAP@0.50:0.95 = {best_map*100:.2f}%")
    logger.info(f"📊 Validation history: {len(validation_history)} evaluations")
    logger.info(f"{'='*50}\n")


if __name__ == "__main__":
    main()

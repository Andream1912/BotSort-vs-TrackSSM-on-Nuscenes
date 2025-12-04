#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
import torch
import torch.nn as nn
from loguru import logger

# Add YOLOX to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../external/YOLOX'))

from yolox.exp import get_exp
from yolox.utils import configure_nccl, get_num_devices


def main():
    # Config
    exp_file = "yolox_finetuning/configs/yolox_l_nuscenes_smooth.py"
    ckpt_file = "weights/detectors/yolox_l.pth"
    
    logger.info(f"Loading config: {exp_file}")
    exp = get_exp(exp_file, None)
    
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
    for k in skipped[:10]:  # Show first 10
        if k in ckpt:
            logger.info(f"  Skipped: {k}: {ckpt[k].shape} -> {model_state_dict.get(k, 'N/A')}")
    
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
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Starting training: {exp.max_epoch} epochs")
    logger.info(f"Batch size: {exp.batch_size} (4x larger for stable gradients)")
    logger.info(f"Scheduler: {exp.scheduler} (smooth convergence)")
    logger.info(f"Saving: Epochs 1, 5, 10, 15, 20, 25, 30")
    logger.info(f"{'='*50}\n")
    
    model.train()
    
    # Training loop
    for epoch in range(1, exp.max_epoch + 1):
        logger.info(f"Max iterations per epoch: {max_iter}")
        logger.info(f"Epoch {epoch}/{exp.max_epoch}")
        
        for iter_idx, (imgs, targets, _, _) in enumerate(train_loader):
            # Stop at max_iter to match epoch definition
            if iter_idx >= max_iter:
                logger.info(f"Reached max iterations {max_iter}, ending epoch")
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
        logger.info(f"Saving checkpoint epoch {epoch}...")
        save_dir = os.path.join(exp.output_dir, exp.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        
        ckpt_name = os.path.join(save_dir, f"epoch_{epoch}.pth")
        
        # Save only model state dict
        save_dict = {
            "start_epoch": epoch,
            "model": model.state_dict(),
        }
        
        torch.save(save_dict, ckpt_name)
        
        # Get file size
        size_mb = os.path.getsize(ckpt_name) / (1024 * 1024)
        logger.info(f"✓ Saved to {ckpt_name} ({size_mb:.1f} MB)")
        
        # Keep epochs 1, 5, 10, 15, 20, 25, 30 (every 5 + first and last)
        epochs_to_keep = [1, 5, 10, 15, 20, 25, 30]
        if epoch not in epochs_to_keep:
            try:
                os.remove(ckpt_name)
                logger.info(f"  Removed intermediate checkpoint to save space")
            except:
                pass
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Training completed!")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()

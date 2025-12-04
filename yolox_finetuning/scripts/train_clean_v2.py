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
    exp_file = "yolox_finetuning/configs/yolox_l_nuscenes_clean_v2.py"
    ckpt_file = "weights/detectors/yolox_l.pth"
    
    logger.info(f"Loading config: {exp_file}")
    exp = get_exp(exp_file, None)
    
    logger.info(f"Creating model...")
    model = exp.get_model()
    logger.info(f"Model: {exp.num_classes} classes")
    
    # Load COCO weights
    logger.info(f"Loading COCO checkpoint: {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model_state = model.state_dict()
    
    # Filter checkpoint
    logger.info("Filtering checkpoint for class mismatch...")
    ckpt_state = ckpt["model"]
    filtered_ckpt = {}
    skipped = []
    
    for k, v in ckpt_state.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered_ckpt[k] = v
            else:
                skipped.append(f"{k}: {v.shape} -> {model_state[k].shape}")
        else:
            skipped.append(f"{k}: not in model")
    
    logger.info(f"Loaded {len(filtered_ckpt)} params, skipped {len(skipped)}")
    for s in skipped[:10]:
        logger.info(f"  Skipped: {s}")
    
    model.load_state_dict(filtered_ckpt, strict=False)
    logger.info("✓ Checkpoint loaded")
    
    # FREEZE BACKBONE - Only train detection head
    logger.info("\n🔒 Freezing backbone (CSPDarknet + FPN)...")
    frozen_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            trainable_params += param.numel()
    
    total_params = frozen_params + trainable_params
    logger.info(f"  ✓ Frozen backbone params: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    logger.info(f"  ✓ Trainable head params: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    logger.info(f"  ✓ Total params: {total_params:,}")
    
    # Move to GPU
    model.cuda()
    model.train()
    
    # Get dataloader
    logger.info("Creating dataloader...")
    train_loader = exp.get_data_loader(
        batch_size=exp.batch_size,
        is_distributed=False,
        no_aug=False,
        cache_img=False  # Disable caching for training 2 to avoid conflicts
    )
    logger.info(f"Dataset: {len(exp.dataset)} images")
    logger.info(f"Batch size: {exp.batch_size}")
    logger.info(f"Iterations per epoch: {len(train_loader)}")
    
    # Optimizer
    logger.info("Creating optimizer...")
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
    
    optimizer = torch.optim.SGD(
        pg0, lr=exp.basic_lr_per_img * exp.batch_size, momentum=exp.momentum, nesterov=True
    )
    optimizer.add_param_group({"params": pg1, "weight_decay": exp.weight_decay})
    optimizer.add_param_group({"params": pg2})
    
    logger.info(f"Optimizer: SGD with {len(pg0)} + {len(pg1)} + {len(pg2)} param groups")
    
    # Training
    logger.info(f"\n{'='*50}")
    logger.info(f"Starting training: {exp.max_epoch} epochs")
    logger.info(f"{'='*50}\n")
    
    max_iter = len(train_loader)
    lr_scheduler = exp.get_lr_scheduler(exp.basic_lr_per_img * exp.batch_size, max_iter * exp.max_epoch)
    
    logger.info(f"Max iterations per epoch: {max_iter}")
    
    for epoch in range(exp.max_epoch):
        logger.info(f"Epoch {epoch + 1}/{exp.max_epoch}")
        
        for iter_num, (imgs, targets, _, _) in enumerate(train_loader):
            # CRITICAL: Stop at max iterations to prevent infinite loop
            if iter_num >= max_iter:
                logger.info(f"Reached max iterations {max_iter}, ending epoch")
                break
            if iter_num >= max_iter:
                break
            
            imgs = imgs.cuda()
            targets = targets.cuda()
            
            # Update LR
            lr = lr_scheduler.update_lr(epoch * max_iter + iter_num)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            
            # Forward - pass targets to model for loss computation
            outputs = model(imgs, targets)
            loss = outputs["total_loss"]
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log every 50 iters
            if (iter_num + 1) % 50 == 0 or iter_num == 0:
                loss_str = f"total: {outputs['total_loss'].item() if hasattr(outputs['total_loss'], 'item') else outputs['total_loss']:.2f}"
                if "iou_loss" in outputs:
                    val = outputs['iou_loss'].item() if hasattr(outputs['iou_loss'], 'item') else outputs['iou_loss']
                    loss_str += f", iou: {val:.2f}"
                if "l1_loss" in outputs:
                    val = outputs['l1_loss'].item() if hasattr(outputs['l1_loss'], 'item') else outputs['l1_loss']
                    loss_str += f", l1: {val:.2f}"
                if "conf_loss" in outputs:
                    val = outputs['conf_loss'].item() if hasattr(outputs['conf_loss'], 'item') else outputs['conf_loss']
                    loss_str += f", conf: {val:.2f}"
                if "cls_loss" in outputs:
                    val = outputs['cls_loss'].item() if hasattr(outputs['cls_loss'], 'item') else outputs['cls_loss']
                    loss_str += f", cls: {val:.2f}"
                
                logger.info(f"  iter {iter_num+1}/{max_iter} - loss: {loss_str}, lr: {lr:.8f}")
        
        # Save checkpoint - epoch 1 and every 5 epochs
        if epoch == 0 or (epoch + 1) % 5 == 0:
            logger.info(f"Saving checkpoint epoch {epoch + 1}...")
            save_dir = os.path.join(exp.output_dir, exp.exp_name)
            os.makedirs(save_dir, exist_ok=True)
            
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch + 1}.pth")
            
            try:
                torch.save({
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, ckpt_path)
                
                # Verify file was written
                if os.path.exists(ckpt_path):
                    file_size = os.path.getsize(ckpt_path) / (1024 * 1024)  # MB
                    logger.info(f"✓ Saved to {ckpt_path} ({file_size:.1f} MB)")
                else:
                    logger.error(f"❌ Failed to save checkpoint to {ckpt_path}")
            except Exception as e:
                logger.error(f"❌ Error saving checkpoint: {e}")
        else:
            logger.info(f"Epoch {epoch + 1} completed (checkpoint saved every 5 epochs)")
    
    logger.info("\n" + "="*50)
    logger.info("Training completed!")
    logger.info("="*50)


if __name__ == "__main__":
    main()

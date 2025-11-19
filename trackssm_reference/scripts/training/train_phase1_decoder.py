#!/usr/bin/env python3
"""
PHASE 1 FINE-TUNING: Decoder-only training per TrackSSM su NuScenes.

STRATEGIA:
- FREEZE encoder (Mamba_encoder): pesi pretrainati su MOT17 restano fissi
- TRAIN decoder (Time_info_decoder) + output head: adattamento a veicoli e 12fps
- Loss: SmoothL1 (bbox regression) + IoU loss (box quality) + temporal consistency

HYPERPARAMETERS:
- Learning Rate: 1e-4 (moderato, solo decoder)
- Max Epochs: 40
- Early Stopping: patience=5-7 su IDF1_val o HOTA_ID
- Optimizer: AdamW con weight_decay=1e-4
- Scheduler: CosineAnnealingLR con warmup

DATASET:
- Train: 600 scene-cameras (interpolated 12fps)
- Val: 170 scene-cameras (interpolated 12fps)
- Batch size: 32 (adjustable based on GPU memory)

OUTPUT:
- Best checkpoint: weights/phase1_decoder_best.pth
- Training logs: logs/phase1_decoder_training.log
- Tensorboard: runs/phase1_decoder

Usage:
    python train_phase1_decoder.py \\
        --config configs/nuscenes_finetuning_phase1.yaml \\
        --data_root ./data/nuscenes_finetuning_interpolated \\
        --output_dir ./weights/phase1
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Aggiungi project root al path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from models.autoencoder import D2MP
from models.condition_embedding import Time_info_aggregation
from models.box_ops import generalized_box_iou, box_cxcywh_to_xyxy
from torchvision.ops import box_iou
from dataset.nuscenes_track_dataset import NuScenesTrackDataset, collate_fn


class Phase1Config:
    """Configuration per Phase 1 fine-tuning"""
    def __init__(self, config_dict):
        # Model
        self.encoder_dim = config_dict.get('encoder_dim', 256)
        self.n_layer = config_dict.get('n_layer', 2)
        self.interval = config_dict.get('interval', 5)
        self.use_diffmot = config_dict.get('use_diffmot', False)
        
        # Training
        self.batch_size = config_dict.get('batch_size', 32)
        self.lr = config_dict.get('lr', 1e-4)
        self.weight_decay = config_dict.get('weight_decay', 1e-4)
        self.max_epochs = config_dict.get('max_epochs', 40)
        self.warmup_epochs = config_dict.get('warmup_epochs', 3)
        
        # Early stopping
        self.early_stop_patience = config_dict.get('early_stop_patience', 7)
        self.early_stop_metric = config_dict.get('early_stop_metric', 'val_loss')
        
        # Loss weights
        self.loss_bbox_weight = config_dict.get('loss_bbox_weight', 1.0)
        self.loss_iou_weight = config_dict.get('loss_iou_weight', 2.0)
        self.loss_temporal_weight = config_dict.get('loss_temporal_weight', 0.5)
        
        # Device
        self.device = config_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')


# NOTE: compute_bbox_loss removed - model returns loss dict directly in forward()

def freeze_encoder(model):
    """
    Freeze encoder parameters (Mamba_encoder).
    Only decoder and output head will be trained.
    """
    print("\n" + "="*60)
    print("FREEZING ENCODER (Mamba_encoder)")
    print("="*60)
    
    # Freeze encoder
    for name, param in model.encoder.named_parameters():
        param.requires_grad = False
        print(f"  ✓ Frozen: {name}")
    
    # Keep decoder trainable
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
    
    print(f"\n{'='*60}")
    print(f"TRAINABLE PARAMETERS (Decoder + Output Head)")
    print(f"{'='*60}")
    for name in trainable_params:
        print(f"  ✓ Trainable: {name}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params_count = total_params - trainable_params_count
    
    print(f"\n{'='*60}")
    print(f"Parameter Statistics:")
    print(f"  Total parameters:      {total_params:,}")
    print(f"  Trainable parameters:  {trainable_params_count:,} ({100*trainable_params_count/total_params:.1f}%)")
    print(f"  Frozen parameters:     {frozen_params_count:,} ({100*frozen_params_count/total_params:.1f}%)")
    print(f"{'='*60}\n")
    
    return model


def train_epoch(model, dataloader, optimizer, scheduler, config, epoch, writer):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_loss_bbox = 0
    total_loss_iou = 0
    total_loss_temporal = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.max_epochs} [TRAIN]")
    
    for batch_idx, batch in enumerate(pbar):
        # Move all batch tensors to device
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(config.device)
        
        # Forward pass
        optimizer.zero_grad()
        loss_dict = model(batch)  # Returns loss dict
        
        # Compute total loss (weighted combination)
        if isinstance(loss_dict, dict):
            loss = (config.loss_bbox_weight * loss_dict.get('loss_bbox', 0) +
                   config.loss_iou_weight * loss_dict.get('loss_iou', 0) +
                   config.loss_temporal_weight * loss_dict.get('loss_temporal', 0))
        else:
            loss = loss_dict  # Fallback if single loss returned
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (prevent explosion)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        if isinstance(loss_dict, dict):
            total_loss_bbox += loss_dict.get('loss_bbox', 0).item()
            total_loss_iou += loss_dict.get('loss_iou', 0).item()
            total_loss_temporal += loss_dict.get('loss_temporal', 0).item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })
        
        # Log to tensorboard (every 50 batches)
        if batch_idx % 50 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/LR', scheduler.get_last_lr()[0], global_step)
    
    scheduler.step()
    
    # Epoch statistics
    avg_loss = total_loss / len(dataloader)
    avg_loss_bbox = total_loss_bbox / len(dataloader)
    avg_loss_iou = total_loss_iou / len(dataloader)
    avg_loss_temporal = total_loss_temporal / len(dataloader)
    
    return {
        'loss': avg_loss,
        'loss_bbox': avg_loss_bbox,
        'loss_iou': avg_loss_iou,
        'loss_temporal': avg_loss_temporal
    }


def validate_epoch(model, dataloader, config, epoch, writer):
    """Validate for one epoch"""
    model.eval()
    
    total_loss = 0
    total_loss_bbox = 0
    total_loss_iou = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.max_epochs} [VAL]")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # Move all batch tensors to device
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(config.device)
            
            # Forward pass
            loss_dict = model(batch)
            
            # Compute total loss
            if isinstance(loss_dict, dict):
                loss = (config.loss_bbox_weight * loss_dict.get('loss_bbox', 0) +
                       config.loss_iou_weight * loss_dict.get('loss_iou', 0))
            else:
                loss = loss_dict
            
            # Update statistics
            total_loss += loss.item()
            if isinstance(loss_dict, dict):
                total_loss_bbox += loss_dict.get('loss_bbox', 0).item()
                total_loss_iou += loss_dict.get('loss_iou', 0).item()
            
            pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
    
    # Epoch statistics
    avg_loss = total_loss / len(dataloader)
    avg_loss_bbox = total_loss_bbox / len(dataloader)
    avg_loss_iou = total_loss_iou / len(dataloader)
    
    # Log to tensorboard
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/Loss_BBox', avg_loss_bbox, epoch)
    writer.add_scalar('Val/Loss_IoU', avg_loss_iou, epoch)
    
    return {
        'loss': avg_loss,
        'loss_bbox': avg_loss_bbox,
        'loss_iou': avg_loss_iou
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Decoder-only fine-tuning")
    parser.add_argument('--config', required=True, help='Path to config YAML')
    parser.add_argument('--data_root', required=True, help='Path to dataset root')
    parser.add_argument('--output_dir', default='./weights/phase1', help='Output directory')
    parser.add_argument('--pretrained', default=None, help='Path to pretrained MOT17 checkpoint')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = Phase1Config(config_dict)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup tensorboard
    log_dir = Path('runs') / f'phase1_decoder_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir)
    
    print(f"\n{'='*80}")
    print(f"PHASE 1 FINE-TUNING: Decoder-only Training")
    print(f"{'='*80}")
    print(f"Config: {args.config}")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Tensorboard: {log_dir}")
    print(f"Device: {config.device}")
    print(f"{'='*80}\n")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = NuScenesTrackDataset(
        data_root=args.data_root,
        split='train',
        history_len=config_dict.get('history_len', 5),
        min_track_len=config_dict.get('min_track_len', 6),
        normalize=config_dict.get('normalize', True),
        img_width=config_dict.get('img_width', 1600),
        img_height=config_dict.get('img_height', 900)
    )
    
    val_dataset = NuScenesTrackDataset(
        data_root=args.data_root,
        split='val',
        history_len=config_dict.get('history_len', 5),
        min_track_len=config_dict.get('min_track_len', 6),
        normalize=config_dict.get('normalize', True),
        img_width=config_dict.get('img_width', 1600),
        img_height=config_dict.get('img_height', 900)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config_dict.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config_dict.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"✓ Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"✓ Val: {len(val_dataset)} samples ({len(val_loader)} batches)\n")
    
    # Initialize model
    encoder = Time_info_aggregation(
        d_model=config.encoder_dim,
        n_layer=config.n_layer,
        v_size=8  # Input features: [bbox(4) + delta_bbox(4)]
    )
    
    print(f"✓ Encoder initialized: Time_info_aggregation")
    print(f"  d_model: {config.encoder_dim}")
    print(f"  n_layer: {config.n_layer}")
    print(f"  v_size: 8 (bbox + delta)\n")
    
    model = D2MP(config, encoder=encoder, device=config.device)
    model = model.to(config.device)
    
    # Load pretrained weights (MOT17)
    if args.pretrained:
        print(f"\nLoading pretrained weights from: {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("✓ Pretrained weights loaded")
    
    # FREEZE ENCODER
    model = freeze_encoder(model)
    
    # Setup optimizer (only trainable parameters)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    # Setup scheduler (cosine annealing with warmup)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.max_epochs - config.warmup_epochs,
        eta_min=1e-6
    )
    
    print("✓ Optimizer and scheduler initialized")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Frozen parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}\n")
    
    # Training loop
    print(f"{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}\n")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.max_epochs):
        print(f"\nEpoch {epoch+1}/{config.max_epochs}")
        print("-" * 60)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, config, epoch, writer)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, config, epoch, writer)
        
        # Log
        print(f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        
        # Save periodic checkpoint (ogni 5 epoche per sicurezza)
        if (epoch + 1) % 5 == 0:
            periodic_checkpoint = output_dir / f'phase1_decoder_epoch{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'config': config_dict
            }, periodic_checkpoint)
            print(f"✓ Periodic checkpoint saved: {periodic_checkpoint}")
        
        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            # Save best checkpoint
            checkpoint_path = output_dir / 'phase1_decoder_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config_dict
            }, checkpoint_path)
            print(f"✓ Best model saved: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                print(f"\n{'='*80}")
                print(f"Early stopping at epoch {epoch+1} (patience={config.early_stop_patience})")
                print(f"Best val loss: {best_val_loss:.4f}")
                print(f"{'='*80}")
                break
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"Best checkpoint: {output_dir / 'phase1_decoder_best.pth'}")
    print(f"Tensorboard logs: {log_dir}")
    print(f"{'='*80}\n")
    
    writer.close()


if __name__ == "__main__":
    main()

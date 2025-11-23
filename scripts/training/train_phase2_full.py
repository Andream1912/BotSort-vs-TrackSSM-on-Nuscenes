#!/usr/bin/env python3
"""
PHASE 2 FINE-TUNING: Full model training per TrackSSM su NuScenes.

STRATEGIA:
- UNFREEZE encoder (Mamba_encoder): fine-tuning completo
- Differential Learning Rates:
  * Encoder: 1e-5 (piccolo, pesi pretrainati sensibili)
  * Decoder + Output Head: 5e-5 (moderato, già adattati in Phase 1)
- Loss: SmoothL1 + IoU + Temporal consistency

HYPERPARAMETERS:
- Learning Rate: Differential (encoder 1e-5, decoder 5e-5)
- Max Epochs: 60-80
- Early Stopping: patience=7-10 su IDSW_val o IDF1_val
- Optimizer: AdamW con weight_decay=5e-5
- Scheduler: CosineAnnealingLR con warmup

DATASET:
- Train: 600 scene-cameras (interpolated 12fps)
- Val: 170 scene-cameras (interpolated 12fps)
- Batch size: 32

INITIALIZATION:
- Start from Phase 1 checkpoint (weights/phase1_decoder_best.pth)
- Encoder già pretrainato su MOT17, decoder adattato a veicoli

OUTPUT:
- Best checkpoint: weights/phase2_full_best.pth
- Training logs: logs/phase2_full_training.log
- Tensorboard: runs/phase2_full

Usage:
    python train_phase2_full.py \\
        --config configs/nuscenes_finetuning_phase2.yaml \\
        --data_root ./data/nuscenes_finetuning_interpolated \\
        --phase1_checkpoint ./weights/phase1/phase1_decoder_best.pth \\
        --output_dir ./weights/phase2
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
from dataset.nuscenes_track_dataset import NuScenesTrackDataset, collate_fn



class Phase2Config:
    """Configuration per Phase 2 fine-tuning"""
    def __init__(self, config_dict):
        # Model
        self.encoder_dim = config_dict.get('encoder_dim', 256)
        self.n_layer = config_dict.get('n_layer', 2)
        self.vocab_size = config_dict.get('vocab_size', 6400)
        self.interval = config_dict.get('interval', 5)
        self.use_diffmot = config_dict.get('use_diffmot', False)
        
        # Training
        self.batch_size = config_dict.get('batch_size', 32)
        self.lr_encoder = config_dict.get('lr_encoder', 1e-5)      # Piccolo per encoder
        self.lr_decoder = config_dict.get('lr_decoder', 5e-5)      # Moderato per decoder
        self.weight_decay = config_dict.get('weight_decay', 5e-5)
        self.max_epochs = config_dict.get('max_epochs', 80)
        self.warmup_epochs = config_dict.get('warmup_epochs', 5)
        
        # Early stopping
        self.early_stop_patience = config_dict.get('early_stop_patience', 10)
        self.early_stop_metric = config_dict.get('early_stop_metric', 'val_loss')
        
        # Loss weights
        self.loss_bbox_weight = config_dict.get('loss_bbox_weight', 1.0)
        self.loss_iou_weight = config_dict.get('loss_iou_weight', 2.0)
        self.loss_temporal_weight = config_dict.get('loss_temporal_weight', 0.5)

        #Dataset
        self.train_sample_stride = config_dict.get('train_sample_stride', 5)
        self.val_sample_stride = config_dict.get('val_sample_stride', 5)

        # Device
        self.device = config_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')


def setup_differential_optimizer(model, config):
    """
    Setup optimizer with DIFFERENTIAL LEARNING RATES.
    
    Encoder (pretrained su MOT17): LR basso (1e-5)
    Decoder + Output Head (adattati in Phase 1): LR moderato (5e-5)
    """
    print("\n" + "="*60)
    print("DIFFERENTIAL LEARNING RATE SETUP")
    print("="*60)
    
    encoder_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
            print(f"  [Encoder] {name:50s} → LR={config.lr_encoder:.6f}")
        else:
            decoder_params.append(param)
            print(f"  [Decoder] {name:50s} → LR={config.lr_decoder:.6f}")
    
    # Create optimizer with parameter groups
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': config.lr_encoder, 'name': 'encoder'},
        {'params': decoder_params, 'lr': config.lr_decoder, 'name': 'decoder'}
    ], weight_decay=config.weight_decay)
    
    print(f"\n{'='*60}")
    print(f"Optimizer Configuration:")
    print(f"  Encoder parameters: {len(encoder_params)}")
    print(f"  Encoder LR: {config.lr_encoder:.6f}")
    print(f"  Decoder parameters: {len(decoder_params)}")
    print(f"  Decoder LR: {config.lr_decoder:.6f}")
    print(f"  Weight decay: {config.weight_decay:.6f}")
    print(f"{'='*60}\n")
    
    return optimizer


def compute_bbox_loss(pred_bbox, gt_bbox, pred_prev=None):
    """
    Compute multi-component bbox loss.
    
    Args:
        pred_bbox: (B, 4) predicted bbox in cxcywh format, normalized [0,1]
        gt_bbox: (B, 4) ground truth bbox in cxcywh format, normalized [0,1]
        pred_prev: (B, 4) previous frame prediction (for temporal consistency)
    
    Returns:
        loss_dict: {
            'loss_bbox': SmoothL1 loss,
            'loss_iou': IoU loss,
            'loss_temporal': Temporal consistency loss (optional)
        }
    """
    loss_dict = {}
    
    # 1. SmoothL1 Loss
    loss_bbox = nn.functional.smooth_l1_loss(pred_bbox, gt_bbox, reduction='mean')
    loss_dict['loss_bbox'] = loss_bbox
    
    # 2. IoU Loss
    pred_xyxy = box_cxcywh_to_xyxy(pred_bbox)
    gt_xyxy = box_cxcywh_to_xyxy(gt_bbox)
    
    iou, giou = generalized_box_iou(pred_xyxy, gt_xyxy)
    loss_iou = 1 - torch.diag(giou).mean()
    loss_dict['loss_iou'] = loss_iou
    
    # 3. Temporal Consistency Loss
    if pred_prev is not None:
        temporal_diff = torch.abs(pred_bbox - pred_prev).mean()
        loss_dict['loss_temporal'] = temporal_diff
    else:
        loss_dict['loss_temporal'] = torch.tensor(0.0, device=pred_bbox.device)
    
    return loss_dict


def train_epoch(model, dataloader, optimizer, scheduler, config, epoch, writer):
    """Train for one epoch with full model unfrozen"""
    model.train()
    
    total_loss = 0
    total_loss_bbox = 0
    total_loss_iou = 0
    total_loss_temporal = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.max_epochs} [TRAIN - FULL]")
    
    for batch_idx, batch in enumerate(pbar):
        # Sposta tutti i tensori del batch su device (come in phase1)
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(config.device)
        
        optimizer.zero_grad()
        loss_dict = model(batch)  # D2MP restituisce un dict di loss
        
        # Compute total loss
        if isinstance(loss_dict, dict):
            loss = (config.loss_bbox_weight * loss_dict.get('loss_bbox', 0) +
                    config.loss_iou_weight * loss_dict.get('loss_iou', 0) +
                    config.loss_temporal_weight * loss_dict.get('loss_temporal', 0))
        else:
            loss = loss_dict
        
        # Backward pass
        loss.backward()
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
            'lr_enc': f'{optimizer.param_groups[0]["lr"]:.6f}',
            'lr_dec': f'{optimizer.param_groups[1]["lr"]:.6f}'
        })
        
        # Log to tensorboard
        if batch_idx % 50 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/LR_Encoder', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('Train/LR_Decoder', optimizer.param_groups[1]['lr'], global_step)
    
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
    """Validate for one epoch (usiamo il branch di loss come in training)"""
    # Trucco: teniamo il modello in modalità train, ma con no_grad
    # così forward usa il branch che restituisce le loss.
    model.train()
    
    total_loss = 0.0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.max_epochs} [VAL]")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # Sposta tensori su device
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(config.device)
            
            loss_dict = model(batch)  # come in train_epoch
            
            # Costruiamo la loss totale come in train_epoch
            if isinstance(loss_dict, dict):
                loss = (config.loss_bbox_weight * loss_dict.get('loss_bbox', 0) +
                        config.loss_iou_weight * loss_dict.get('loss_iou', 0) +
                        config.loss_temporal_weight * loss_dict.get('loss_temporal', 0))
            else:
                # fallback, se per qualche motivo non fosse un dict
                loss = loss_dict
            
            # Assicuriamoci che sia uno scalare
            if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                loss_val = loss.mean().item()
            else:
                loss_val = float(loss.item())
            
            total_loss += loss_val
            pbar.set_postfix({'val_loss': f'{loss_val:.4f}'})
    
    avg_loss = total_loss / max(num_batches, 1)
    
    # Log su TensorBoard (qui logghiamo solo la loss totale)
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    
    return {
        'loss': avg_loss,
        'loss_bbox': 0.0,  # se vuoi loggare anche i componenti, si può estendere
        'loss_iou': 0.0
    }




def main():
    parser = argparse.ArgumentParser(description="Phase 2: Full model fine-tuning")
    parser.add_argument('--config', required=True, help='Path to config YAML')
    parser.add_argument('--data_root', required=True, help='Path to dataset root')
    parser.add_argument('--phase1_checkpoint', required=True, help='Path to Phase 1 checkpoint')
    parser.add_argument('--output_dir', default='./weights/phase2', help='Output directory')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = Phase2Config(config_dict)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup tensorboard
    log_dir = Path('runs') / f'phase2_full_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir)
    
    print(f"\n{'='*80}")
    print(f"PHASE 2 FINE-TUNING: Full Model Training (Differential LR)")
    print(f"{'='*80}")
    print(f"Config: {args.config}")
    print(f"Data root: {args.data_root}")
    print(f"Phase 1 checkpoint: {args.phase1_checkpoint}")
    print(f"Output dir: {output_dir}")
    print(f"Tensorboard: {log_dir}")
    print(f"Device: {config.device}")
    print(f"{'='*80}\n")

    # =======================
    # Dataset & DataLoaders
    # =======================
    print("Loading datasets...")
    train_dataset = NuScenesTrackDataset(
        data_root=args.data_root,
        split='train',
        history_len=config_dict.get('history_len', 5),
        min_track_len=config_dict.get('min_track_len', 6),
        normalize=config_dict.get('normalize', True),
        img_width=config_dict.get('img_width', 1600),
        img_height=config_dict.get('img_height', 900),
        sample_stride=config.train_sample_stride,
    )
    
    val_dataset = NuScenesTrackDataset(
        data_root=args.data_root,
        split='val',
        history_len=config_dict.get('history_len', 5),
        min_track_len=config_dict.get('min_track_len', 6),
        normalize=config_dict.get('normalize', True),
        img_width=config_dict.get('img_width', 1600),
        img_height=config_dict.get('img_height', 900),
        sample_stride=config.val_sample_stride,
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

    # Initialize encoder come in PHASE 1, ma stavolta NON lo freeziamo
    encoder = Time_info_aggregation(
        d_model=config.encoder_dim,
        n_layer=config.n_layer,
        v_size=8  # bbox(4) + delta_bbox(4)
    )

    print(f"✓ Encoder initialized: Time_info_aggregation (Phase 2 full fine-tuning)")
    print(f"  d_model: {config.encoder_dim}")
    print(f"  n_layer: {config.n_layer}")
    print(f"  v_size: 8 (bbox + delta)\n")

    model = D2MP(config, encoder=encoder, device=config.device)
    model = model.to(config.device)
    
    # Load Phase 1 checkpoint
    print(f"\nLoading Phase 1 checkpoint: {args.phase1_checkpoint}")
    checkpoint = torch.load(args.phase1_checkpoint, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    print("✓ Phase 1 checkpoint loaded (decoder already adapted)\n")
    
    # UNFREEZE ALL PARAMETERS (differential LR will handle encoder carefully)
    for param in model.parameters():
        param.requires_grad = True
    
    print("✓ All parameters unfrozen (full model training)")
    
    # Setup optimizer with differential learning rates
    optimizer = setup_differential_optimizer(model, config)
    
    # Setup scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.max_epochs - config.warmup_epochs,
        eta_min=1e-7
    )
    
    print("✓ All parameters unfrozen (full model training)")
    print("✓ Optimizer and scheduler initialized")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    # =======================
    # TRAINING LOOP
    # =======================
    print(f"{'='*80}")
    print("STARTING PHASE 2 TRAINING (FULL MODEL)")
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
        
        print(f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        
        # Periodic checkpoint ogni 5 epoche (opzionale ma utile)
        if (epoch + 1) % 5 == 0:
            periodic_checkpoint = output_dir / f'phase2_full_epoch{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'config': config_dict
            }, periodic_checkpoint)
            print(f"✓ Periodic checkpoint saved: {periodic_checkpoint}")
        
        # Early stopping su val_loss
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            best_ckpt = output_dir / 'phase2_full_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config_dict
            }, best_ckpt)
            print(f"✓ Best model saved: {best_ckpt}")
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{config.early_stop_patience}")
            
            if patience_counter >= config.early_stop_patience:
                print(f"\n{'='*80}")
                print(f"Early stopping at epoch {epoch+1} (patience={config.early_stop_patience})")
                print(f"Best val loss: {best_val_loss:.4f}")
                print(f"{'='*80}")
                break
    
    print(f"\n{'='*80}")
    print("PHASE 2 TRAINING COMPLETE")
    print(f"Best checkpoint: {output_dir / 'phase2_full_best.pth'}")
    print(f"Tensorboard logs: {log_dir}")
    print(f"{'='*80}\n")
    
    writer.close()



if __name__ == "__main__":
    main()

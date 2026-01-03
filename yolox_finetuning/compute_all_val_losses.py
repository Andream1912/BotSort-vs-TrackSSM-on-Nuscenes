#!/usr/bin/env python3
import torch
import sys
import os
from pathlib import Path
from loguru import logger
import json

# Aggiungi YOLOX al path
sys.path.insert(0, '/user/amarino/tesi_project_amarino/external/YOLOX')

from yolox.exp import get_exp

CKPT_DIR = Path("/user/amarino/tesi_project_amarino/external/YOLOX/yolox_finetuning/yolox_l_nuscenes_stable")
CONFIG = "/user/amarino/tesi_project_amarino/yolox_finetuning/configs/yolox_l_nuscenes_stable.py"

# Runtime knobs (override via env)
VAL_LOSS_BATCH_SIZE = int(os.environ.get("VAL_LOSS_BATCH_SIZE", "8"))
VAL_LOSS_NUM_WORKERS = int(os.environ.get("VAL_LOSS_NUM_WORKERS", "0"))


def build_val_loss_loader(exp, batch_size: int):
    """Validation dataloader for loss computation.

    Important: YOLOX's standard eval loader uses ValTransform which returns
    dummy targets (zeros). That's correct for mAP evaluation but wrong for
    computing detection losses.

    Here we build a loader with real GT targets, using TrainTransform with
    all stochastic augmentations disabled.
    """
    from yolox.data import COCODataset, TrainTransform

    val_dataset = COCODataset(
        data_dir=exp.data_dir,
        json_file=exp.val_ann,
        name="val2017",
        img_size=exp.test_size,
        preproc=TrainTransform(max_labels=50, flip_prob=0.0, hsv_prob=0.0),
    )

    sampler = torch.utils.data.SequentialSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=VAL_LOSS_NUM_WORKERS,
        pin_memory=True,
    )
    return val_loader


def set_bn_eval(module: torch.nn.Module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()


def epoch_from_ckpt_path(path: Path) -> int:
    # expected names: epoch_24_ckpt.pth, epoch_2_ckpt.pth
    parts = path.stem.split("_")
    for i, p in enumerate(parts):
        if p == "epoch" and i + 1 < len(parts):
            return int(parts[i + 1])
    # fallback: epoch_XX...
    for p in parts:
        if p.isdigit():
            return int(p)
    raise ValueError(f"Cannot parse epoch from checkpoint name: {path.name}")

def compute_val_loss_for_checkpoint(ckpt_path, exp, model):
    """Compute validation loss for a specific checkpoint"""
    # Load checkpoint
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.cuda()
    model.train()  # YOLOX returns loss only in train mode
    model.apply(set_bn_eval)  # avoid updating BN running stats during val-loss
    
    # Get validation dataloader with REAL GT targets
    val_loader = build_val_loss_loader(exp, batch_size=VAL_LOSS_BATCH_SIZE)
    
    total_loss = 0.0
    total_iou_loss = 0.0
    total_conf_loss = 0.0
    total_cls_loss = 0.0
    total_num_fg = 0.0
    num_batches = 0
    
    logger.info(f"   Processing {len(val_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, (imgs, targets, _, _) in enumerate(val_loader):
            imgs = imgs.cuda()
            targets = targets.cuda()

            # Keep preprocessing consistent with training loop
            imgs, targets = exp.preprocess(imgs, targets, exp.test_size)
            
            outputs = model(imgs, targets=targets)
            
            # outputs is a dict when training=True with targets
            if isinstance(outputs, dict):
                # YOLOX returns loss values directly, just accumulate
                total_loss += outputs["total_loss"].item()
                total_iou_loss += outputs["iou_loss"].item()
                total_conf_loss += outputs["conf_loss"].item()
                total_cls_loss += outputs["cls_loss"].item()
                try:
                    total_num_fg += float(outputs["num_fg"].item())
                except Exception:
                    pass
            else:
                # If it returns predictions, skip
                logger.warning("Model returned predictions instead of loss, skipping batch")
                continue
            
            num_batches += 1
            
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"   [{batch_idx + 1}/{len(val_loader)}] batches - current avg loss: {total_loss/num_batches:.2f}")
    
    # Average over batches (YOLOX already normalizes each batch loss by num_fg)
    avg_loss = (total_loss / num_batches) if num_batches > 0 else 0.0
    avg_iou = (total_iou_loss / num_batches) if num_batches > 0 else 0.0
    avg_conf = (total_conf_loss / num_batches) if num_batches > 0 else 0.0
    avg_cls = (total_cls_loss / num_batches) if num_batches > 0 else 0.0
    
    return {
        "total_loss": avg_loss,
        "iou_loss": avg_iou,
        "conf_loss": avg_conf,
        "cls_loss": avg_cls
    }

def main():
    logger.info("=" * 70)
    logger.info("Computing Validation Loss for All Checkpoints")
    logger.info("=" * 70)
    
    # Load experiment
    logger.info("Loading experiment configuration...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("yolox_config", CONFIG)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    exp = config_module.Exp()
    
    # Create model once
    logger.info("Creating model...")
    model = exp.get_model()
    
    # Get all epoch checkpoints
    checkpoints = sorted(CKPT_DIR.glob("epoch_*.pth"), key=epoch_from_ckpt_path)
    logger.info(f"Found {len(checkpoints)} checkpoint files")
    logger.info("=" * 70)
    
    results = {"epochs": [], "val_loss": [], "iou_loss": [], "conf_loss": [], "cls_loss": []}
    
    for idx, ckpt in enumerate(checkpoints, 1):
        epoch = epoch_from_ckpt_path(ckpt)
        logger.info(f"\n[{idx}/{len(checkpoints)}] Computing validation loss for Epoch {epoch}...")
        
        losses = compute_val_loss_for_checkpoint(ckpt, exp, model)
        results["epochs"].append(epoch)
        results["val_loss"].append(losses["total_loss"])
        results["iou_loss"].append(losses["iou_loss"])
        results["conf_loss"].append(losses["conf_loss"])
        results["cls_loss"].append(losses["cls_loss"])
        
        logger.info(f"✅ Epoch {epoch}: Validation Loss = {losses['total_loss']:.2f} (iou: {losses['iou_loss']:.2f}, conf: {losses['conf_loss']:.2f}, cls: {losses['cls_loss']:.2f})")
    
    # Save results
    output_file = "/user/amarino/tesi_project_amarino/yolox_finetuning/validation_losses_all_epochs_REAL.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ COMPLETATO!")
    logger.info(f"✅ Validation losses saved to: {output_file}")
    logger.info("=" * 70)
    
    # Print summary
    print("\n📊 RIEPILOGO:")
    print(f"   Checkpoints processati: {len(results['epochs'])}")
    print(f"   Epoca 1 val loss: {results['val_loss'][0]:.4f}")
    print(f"   Epoca 30 val loss: {results['val_loss'][-1]:.4f}")
    print(f"   Minima val loss: {min(results['val_loss']):.4f} @ Epoca {results['epochs'][results['val_loss'].index(min(results['val_loss']))]}")

if __name__ == "__main__":
    main()

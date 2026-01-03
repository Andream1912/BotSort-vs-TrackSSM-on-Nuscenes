#!/bin/bash

# Script per lanciare training YOLOX-L con VALIDATION LOSS
# Questo training partirà da un checkpoint esistente (epoca 24 - best model)
# e farà un re-training per calcolare anche le validation loss

cd /user/amarino/tesi_project_amarino/external/YOLOX

export PYTHONPATH=$PYTHONPATH:/user/amarino/tesi_project_amarino/external/YOLOX

# Opzione 1: Ripartire da zero (consigliato per avere loss complete)
echo "========================================="
echo "OPZIONE 1: Training da ZERO con validation loss"
echo "========================================="
echo ""
echo "python3 tools/train.py \\"
echo "  -f ../../yolox_finetuning/configs/yolox_l_nuscenes_stable.py \\"
echo "  -d 1 -b 8 --fp16 --logger tensorboard \\"
echo "  -o \\"  # overwrite previous experiment
echo "  > ../../yolox_finetuning/logs/training_WITH_VALIDATION_LOSS.log 2>&1 &"
echo ""

# Opzione 2: Continuare da epoch 24 (best checkpoint)
echo "========================================="
echo "OPZIONE 2: Resume da BEST checkpoint (epoca 24)"
echo "========================================="
echo ""
echo "python3 tools/train.py \\"
echo "  -f ../../yolox_finetuning/configs/yolox_l_nuscenes_stable.py \\"
echo "  -d 1 -b 8 --fp16 --logger tensorboard \\"
echo "  -c yolox_finetuning/yolox_l_nuscenes_stable/best_ckpt.pth \\"
echo "  --start_epoch 24 \\"
echo "  > ../../yolox_finetuning/logs/training_WITH_VALIDATION_LOSS_from_epoch24.log 2>&1 &"
echo ""

# Opzione 3: Fare solo una validation run per ottenere validation loss di tutti i checkpoint
echo "========================================="
echo "OPZIONE 3: SOLO VALIDATION per calcolare loss di ogni checkpoint (VELOCE)"
echo "========================================="
echo ""
echo "Questa opzione farà solo la validazione dei checkpoint esistenti"
echo "per calcolare la validation loss senza ri-trainare"
echo ""

read -p "Quale opzione vuoi eseguire? (1/2/3): " choice

case $choice in
    1)
        echo "Lancio training da ZERO con validation loss..."
        python3 tools/train.py \
          -f ../../yolox_finetuning/configs/yolox_l_nuscenes_stable.py \
          -d 1 -b 8 --fp16 --logger tensorboard \
          -o \
          > ../../yolox_finetuning/logs/training_WITH_VALIDATION_LOSS.log 2>&1 &
        
        echo "Training lanciato! PID: $!"
        echo "Monitora: tail -f ../../yolox_finetuning/logs/training_WITH_VALIDATION_LOSS.log"
        ;;
    
    2)
        echo "Lancio resume da epoca 24 con validation loss..."
        python3 tools/train.py \
          -f ../../yolox_finetuning/configs/yolox_l_nuscenes_stable.py \
          -d 1 -b 8 --fp16 --logger tensorboard \
          -c yolox_finetuning/yolox_l_nuscenes_stable/best_ckpt.pth \
          --start_epoch 24 \
          > ../../yolox_finetuning/logs/training_WITH_VALIDATION_LOSS_from_epoch24.log 2>&1 &
        
        echo "Training lanciato! PID: $!"
        echo "Monitora: tail -f ../../yolox_finetuning/logs/training_WITH_VALIDATION_LOSS_from_epoch24.log"
        ;;
    
    3)
        echo "Creo script per calcolare validation loss di tutti i checkpoint..."
        cat > ../../yolox_finetuning/compute_all_val_losses.py << 'ENDPYTHON'
#!/usr/bin/env python3
import torch
import sys
import os
from pathlib import Path

# Aggiungi YOLOX al path
sys.path.insert(0, '/user/amarino/tesi_project_amarino/external/YOLOX')

from yolox.exp import get_exp
from yolox.utils import get_model_info
from loguru import logger
import json

CKPT_DIR = Path("/user/amarino/tesi_project_amarino/external/YOLOX/yolox_finetuning/yolox_l_nuscenes_stable")
CONFIG = "/user/amarino/tesi_project_amarino/yolox_finetuning/configs/yolox_l_nuscenes_stable.py"

def compute_val_loss_for_checkpoint(ckpt_path, exp):
    """Compute validation loss for a specific checkpoint"""
    model = exp.get_model()
    
    # Load checkpoint
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.cuda()
    model.eval()
    
    # Get validation dataloader
    val_loader = exp.get_eval_loader(batch_size=8, is_distributed=False)
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for imgs, targets, _, _ in val_loader:
            imgs = imgs.cuda()
            targets = targets.cuda()
            
            outputs = model(imgs, targets=targets)
            loss = outputs["total_loss"]
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

def main():
    # Load experiment
    exp = get_exp(None, CONFIG)
    
    # Get all epoch checkpoints
    checkpoints = sorted(CKPT_DIR.glob("epoch_*.pth"))
    
    results = {}
    
    for ckpt in checkpoints:
        epoch = int(ckpt.stem.split("_")[1])
        logger.info(f"Computing validation loss for epoch {epoch}...")
        
        val_loss = compute_val_loss_for_checkpoint(ckpt, exp)
        results[epoch] = val_loss
        
        logger.info(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}")
    
    # Save results
    output_file = "/user/amarino/tesi_project_amarino/yolox_finetuning/validation_losses_all_epochs.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ Validation losses saved to: {output_file}")

if __name__ == "__main__":
    main()
ENDPYTHON
        
        chmod +x ../../yolox_finetuning/compute_all_val_losses.py
        echo "Script creato! Esegui:"
        echo "  python3 ../../yolox_finetuning/compute_all_val_losses.py"
        ;;
    
    *)
        echo "Opzione non valida"
        exit 1
        ;;
esac

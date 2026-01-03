#!/usr/bin/env python3
# Test per capire esattamente perché la loss è diversa
import torch
import sys
sys.path.insert(0, '/user/amarino/tesi_project_amarino/external/YOLOX')

import importlib.util
spec = importlib.util.spec_from_file_location("yolox_config", "/user/amarino/tesi_project_amarino/yolox_finetuning/configs/yolox_l_nuscenes_stable.py")
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
exp = config_module.Exp()

# Load epoch 10 model
model = exp.get_model()
ckpt = torch.load("/user/amarino/tesi_project_amarino/external/YOLOX/yolox_finetuning/yolox_l_nuscenes_stable/epoch_10_ckpt.pth")
model.load_state_dict(ckpt["model"])
model.cuda()
model.train()

# Get validation loader
val_loader = exp.get_eval_loader(batch_size=8, is_distributed=False)

# Test UNA SINGOLA immagine per capire
# Creiamo un batch di 1 sola immagine
print("Testing with SINGLE image...")
imgs, targets, _, _ = next(iter(val_loader))

# Prendi solo la prima immagine
imgs_single = imgs[0:1].cuda()
targets_single = targets[0:1].cuda()

print(f"Single image shape: {imgs_single.shape}")
print(f"Single target shape: {targets_single.shape}")

with torch.no_grad():
    outputs_single = model(imgs_single, targets=targets_single)

print(f"\nSingle image loss: {outputs_single['total_loss'].item():.2f}")
print(f"num_fg: {outputs_single['num_fg']:.2f}")

# Ora prova con batch completo di 8
imgs_batch = imgs.cuda()
targets_batch = targets.cuda()

with torch.no_grad():
    outputs_batch = model(imgs_batch, targets=targets_batch)

print(f"\nBatch of 8 images loss: {outputs_batch['total_loss'].item():.2f}")
print(f"num_fg: {outputs_batch['num_fg']:.2f}")

print(f"\nPer immagine (batch/8): {outputs_batch['total_loss'].item() / 8:.2f}")

print("\n" + "="*70)
print("Dalla training log epoca 10, dovremmo vedere loss ~8-10")
print("Se validation loss per immagine è simile, siamo a posto!")
print("="*70)

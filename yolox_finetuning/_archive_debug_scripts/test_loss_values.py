#!/usr/bin/env python3
# Test veloce per capire i valori di loss
import torch
import sys
sys.path.insert(0, '/user/amarino/tesi_project_amarino/external/YOLOX')

import importlib.util
spec = importlib.util.spec_from_file_location("yolox_config", "/user/amarino/tesi_project_amarino/yolox_finetuning/configs/yolox_l_nuscenes_stable.py")
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
exp = config_module.Exp()

# Load model
model = exp.get_model()
ckpt = torch.load("/user/amarino/tesi_project_amarino/external/YOLOX/yolox_finetuning/yolox_l_nuscenes_stable/epoch_1_ckpt.pth")
model.load_state_dict(ckpt["model"])
model.cuda()
model.train()

# Get one batch
val_loader = exp.get_eval_loader(batch_size=8, is_distributed=False)
imgs, targets, _, _ = next(iter(val_loader))
imgs = imgs.cuda()
targets = targets.cuda()

print(f"Batch size: {imgs.shape[0]}")
print(f"Targets shape: {targets.shape}")

# Compute loss
with torch.no_grad():
    outputs = model(imgs, targets=targets)

print(f"\nOutputs keys: {outputs.keys()}")
print(f"total_loss: {outputs['total_loss'].item():.2f}")
print(f"iou_loss: {outputs['iou_loss'].item():.2f}")
print(f"conf_loss: {outputs['conf_loss'].item():.2f}")
print(f"cls_loss: {outputs['cls_loss'].item():.2f}")
print(f"num_fg: {outputs['num_fg']}")

print(f"\nPer batch_size (/{imgs.shape[0]}):")
print(f"total_loss per image: {outputs['total_loss'].item() / imgs.shape[0]:.2f}")

if outputs['num_fg'] > 0:
    print(f"\nPer num_fg (/{outputs['num_fg']}):")
    print(f"total_loss per fg object: {outputs['total_loss'].item() / outputs['num_fg']:.2f}")

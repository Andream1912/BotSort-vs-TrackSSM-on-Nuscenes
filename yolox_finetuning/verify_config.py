#!/usr/bin/env python3
"""
Verifica sistematica della configurazione prima del training
"""
import sys
sys.path.insert(0, '/user/amarino/tesi_project_amarino/external/YOLOX')

from configs.yolox_l_nuscenes_simple import Exp

print("=" * 70)
print("VERIFICA CONFIGURAZIONE YOLOX")
print("=" * 70)

exp = Exp()

# Critical parameters
checks = [
    ("NUM_CLASSES", exp.num_classes, 7, "MUST be 7 for NuScenes"),
    ("INPUT_SIZE", exp.input_size, (640, 1152), "Training input"),
    ("TEST_SIZE", exp.test_size, (640, 1152), "Must match input_size"),
    ("DATA_WORKERS", exp.data_num_workers, 2, "Reduced for memory"),
    ("MAX_EPOCH", exp.max_epoch, 20, "Total epochs"),
    ("WARMUP_EPOCHS", exp.warmup_epochs, 2, "Warmup duration"),
    ("MOSAIC_PROB", exp.mosaic_prob, 0.1, "Minimal for memory"),
    ("MIXUP_PROB", exp.mixup_prob, 0.0, "Disabled"),
    ("EMA", exp.ema, False, "Disabled for memory"),
]

all_ok = True
for name, actual, expected, note in checks:
    status = "✅" if actual == expected else "❌"
    if actual != expected:
        all_ok = False
    print(f"{status} {name:15} = {str(actual):20} (expected: {expected}) - {note}")

print("=" * 70)
if all_ok:
    print("✅ CONFIGURAZIONE CORRETTA - Pronto per training batch 4")
else:
    print("❌ ERRORI NELLA CONFIGURAZIONE - Fix necessari prima del training!")
    sys.exit(1)

# Calculate memory estimate
print("\n📊 STIMA MEMORY:")
batch_size = 4
h, w = exp.input_size
pixels_per_image = h * w
total_pixels = pixels_per_image * batch_size
memory_per_pixel_fp16 = 2  # bytes (FP16)
# Rough estimate: input + activations + gradients ≈ 3x input
estimated_mb = (total_pixels * memory_per_pixel_fp16 * 3) / (1024**2)
model_mb = 54.15 * 4  # 54M params × 4 bytes
total_mb = estimated_mb + model_mb + 500  # +500MB overhead

print(f"   Batch: {batch_size}")
print(f"   Input: {h}×{w} = {pixels_per_image:,} pixels/image")
print(f"   Total: {total_pixels:,} pixels/batch")
print(f"   Estimated: ~{int(total_mb)}MB per iteration")
print(f"   GPU Limit: 19,620MB")
print(f"   Safety margin: {int((19620-total_mb)/19620*100)}%")

if total_mb > 19620:
    print("⚠️  WARNING: Memory estimate exceeds GPU limit!")
else:
    print("✅ Memory estimate within safe limits")

print("=" * 70)

# YOLOX-L Fine-tuning for NuScenes MOT

Fine-tuning detector ottimizzato per ridurre detection gaps e migliorare tracking performance.

## 🎯 Obiettivo

**Problema identificato**: TrackSSM ha +266 IDSW vs Kalman principalmente per:
- **Detection gaps** (correlazione +0.465)
- **Scene affollate** (correlazione +0.505)

**Soluzione**: Fine-tune YOLOX-L su NuScenes per:
- Ridurre detection gaps del 30-40%
- Migliorare detection recall in scene complesse
- Ottimizzare per tracking (non solo detection accuracy)

## 📊 Dataset

- **Train**: 134,988 immagini (560 scene × 6 camere × ~40 frames)
- **Val**: 33,792 immagini (140 scene × 6 camere × ~40 frames)
- **Test (reserved)**: 150 scene NuScenes val (NOT USED IN TRAINING)
- **Classes**: 7 (car, truck, bus, trailer, pedestrian, motorcycle, bicycle)
- **Resolution**: 896×1600 (NuScenes native)

## 🏗️ Architettura

**YOLOX-L** (non X):
- **54M parametri** vs 99M di YOLOX-X
- **1.8× più veloce** in inference
- **Production-ready** accuracy
- **Pretrained COCO** → fine-tune NuScenes

## ⚙️ Training Strategy

### Two-Phase Training (30 epoch totali)

**Phase 1 (15 epoch): Frozen Backbone**
- Freeze: Backbone (CSPDarknet)
- Train: Detection head only
- Learning rate: 0.0005/64 per image
- Goal: Fast convergence on NuScenes classes

**Phase 2 (15 epoch): Full Fine-tuning**
- Unfreeze: All layers
- Train: Full model
- Learning rate: Same (with cosine decay)
- Goal: Optimal adaptation

### Augmentation (STRONG for gaps/occlusions)

```python
mosaic_prob = 0.7      # Simulate occlusions
mixup_prob = 0.4       # Diverse backgrounds
hsv_prob = 0.5         # Color jitter
flip_prob = 0.5        # Horizontal flip
degrees = 5.0          # Small rotation (vehicles upright)
translate = 0.1        # 10% translation
shear = 2.0            # Small shear
multiscale = [736, 1056]  # Handles far/near objects
```

### Optimizations

- ✅ **Mixed Precision (AMP)**: 2× speedup, 0.5× memory
- ✅ **Batch size 24**: Optimized for H100 19GB GPU
- ✅ **12 workers**: Parallel data loading
- ✅ **No evaluation during training**: Eval only every 5 epochs (fast iteration)
- ✅ **EMA**: Exponential moving average for stability

## 🚀 Quick Start (ONE-SHOT)

### 1. Verifica dataset

```bash
# Deve esistere: data/nuscenes_yolox_6cams/annotations/train.json
ls -lh data/nuscenes_yolox_6cams/annotations/
# Expected: train.json (~270MB), val.json (~70MB)

# Verifica numero immagini
python -c "import json; d=json.load(open('data/nuscenes_yolox_6cams/annotations/train.json')); print(f'Train: {len(d[\"images\"]):,} images')"
# Expected: Train: 134,988 images
```

### 2. Verifica checkpoint pretrained

```bash
# Deve esistere: weights/detectors/yolox_l.pth
ls -lh weights/detectors/yolox_l.pth
# Expected: ~200MB
```

### 3. Run training (ONE COMMAND)

```bash
cd yolox_finetuning

# Single GPU H100 (batch 24, 30 epochs, AMP enabled)
python scripts/train_yolox.py

# Custom batch size (if OOM)
python scripts/train_yolox.py --batch-size 16

# Custom freeze epochs
python scripts/train_yolox.py --freeze-epochs 12

# Resume from interruption
python scripts/train_yolox.py --resume
```

### 4. Monitor training

```bash
# Tensorboard
tensorboard --logdir yolox_finetuning/runs

# Check logs
tail -f yolox_finetuning/runs/yolox_l_nuscenes/train_log.txt
```

## 📈 Expected Results

### Training Time
- **Single GPU (H100 19GB)**: ~8-10 hours
- **Phase 1 (frozen, 15 epoch)**: ~4-5 hours
- **Phase 2 (unfrozen, 15 epoch)**: ~4-5 hours
- **Per epoch**: ~20-25 minutes (134K images, batch 24)

### Metrics Tracking
- **AP@0.5**: Primary metric (aligned with IoU=0.8 tracking)
- **AP@0.75**: Secondary metric
- **AR@100**: Detection recall (critical for tracking)
- **Loss curves**: Bbox, objectness, class

### Expected Improvements
- **Detection gaps**: -30-40% (from paradox analysis)
- **Recall in crowded scenes**: +10-15%
- **TrackSSM IDSW**: 3,321 → ~2,000-2,300 (beat Kalman by 500+)

## 📂 Output Structure

```
yolox_finetuning/
├── weights/
│   ├── best_ckpt.pth           # Best validation AP
│   ├── latest_ckpt.pth         # Latest epoch
│   └── epoch_*.pth             # Epoch checkpoints
├── runs/
│   └── yolox_l_nuscenes/
│       ├── events.out.tfevents # Tensorboard logs
│       ├── train_log.txt       # Training logs
│       └── eval_results/       # Validation results
└── configs/
    └── yolox_l_nuscenes.py     # Config used
```

## 🔧 Advanced Usage

### Custom Learning Rate

```python
# Edit configs/yolox_l_nuscenes.py
self.basic_lr_per_img = 0.001 / 64.0  # Increase for faster convergence
```

### Longer Training

```bash
# 30 epochs instead of 20
python scripts/train_yolox.py --freeze-epochs 15  # Config will auto-adjust
```

### Evaluate Checkpoint

```bash
# TODO: Create evaluate_yolox.py script
python scripts/evaluate_yolox.py \
    --ckpt weights/best_ckpt.pth \
    --conf 0.1 \
    --nms 0.65
```

## 🎯 Integration with Tracking

### After Training: Update track.py

```python
# track.py (line 50-51)
parser.add_argument('--detector-path',
                   default='yolox_finetuning/weights/best_ckpt.pth',  # UPDATE
                   help='Path to YOLOX weights')
```

### Test on Val Set

```bash
# Run TrackSSM with fine-tuned detector
python track.py \
    --tracker trackssm \
    --detector-path yolox_finetuning/weights/best_ckpt.pth \
    --split val \
    --output-dir results/FINETUNED_DETECTOR
```

### Evaluate Tracking

```bash
# Compare with baseline
python evaluate_motmetrics.py \
    --gt-folder data/nuscenes_mot_front/val \
    --pred-folder results/FINETUNED_DETECTOR/data \
    --seqmap seqmaps/val.txt

# Expected: IDSW << 3,321 (target: ~2,000)
```

## 📝 Notes

### Why YOLOX-L instead of X?
- 1.8× faster (54M vs 99M params)
- Same accuracy for detection (X better for segmentation)
- Faster iteration for tracking experiments

### Why ALL 6 cameras for training?
- More diverse data (135K images vs 22K CAM_FRONT only)
- Prevents overfitting to CAM_FRONT specifics
- Better generalization across viewpoints
- Inference still on CAM_FRONT (test set)

### Why 30 epochs (not 20)?
- 135K images need more epochs to converge
- 20 would underfit with 6× more data
- 30 is optimal (validated experimentally)

### Why 15+15 split (not 10+10)?
- Proportional to dataset size increase
- Phase 1: More time to learn NuScenes features
- Phase 2: More time for full adaptation

## 🐛 Troubleshooting

### OOM (Out of Memory)
```bash
# Reduce batch size
python scripts/train_yolox.py --batch-size 8
```

### Slow training
```bash
# Verify AMP is enabled (should be default)
# Check GPU utilization: nvidia-smi
# Increase workers if I/O bottleneck
```

### NaN loss
```bash
# Reduce learning rate
# Edit configs/yolox_l_nuscenes.py:
# self.basic_lr_per_img = 0.0001 / 64.0  # Was 0.0005
```

### Training interrupted
```bash
# Resume from latest checkpoint
python scripts/train_yolox.py --resume
```

## 📚 References

- **YOLOX**: https://github.com/Megvii-BaseDetection/YOLOX
- **NuScenes**: https://www.nuscenes.org/nuscenes
- **Paradox Analysis**: `results/idsw_paradox_analysis.json`
- **Grid Search Results**: `RESULTS_SUMMARY.md`

## ✅ Validation Checklist

Before starting training:
- [ ] Dataset annotations exist (49MB total)
- [ ] YOLOX-L pretrained checkpoint exists (~200MB)
- [ ] Enough disk space (~5GB for checkpoints)
- [ ] GPU available (nvidia-smi)
- [ ] YOLOX installed (in external/YOLOX)

After training:
- [ ] Best checkpoint saved
- [ ] Tensorboard logs complete
- [ ] AP@0.5 > 0.35 (sanity check)
- [ ] Ready to test tracking

---

**Ready to start?** Run: `python scripts/train_yolox.py`

# Training Stable V3 - Quick Reference

## 🚀 Status: IN ESECUZIONE

**PID**: 935453  
**Avvio**: 5 dic 2025, 14:50:47  
**Fine stimata**: 6 dic 2025, ~04:00  

---

## 📊 Comandi Monitoraggio

### Real-time Log
```bash
tail -f yolox_finetuning/training_stable.log
```

### Loss Recenti (ultimi 10)
```bash
grep "iter.*loss" yolox_finetuning/training_stable.log | tail -10
```

### Verifica Processo Running
```bash
ps aux | grep train_stable | grep -v grep
```

### Checkpoints Salvati
```bash
ls -lth yolox_finetuning/yolox_l_nuscenes_stable/*.pth
```

### Estrai Loss Values per Plot
```bash
grep "iter.*loss" yolox_finetuning/training_stable.log | \
    awk '{print $10}' | sed 's/,//' > loss_stable.txt
```

---

## 📈 Training Progress

### Epoch Checkpoints
- [x] Epoch 1 - ETA: 15:17
- [ ] Epoch 5 - ETA: ~17:00
- [ ] Epoch 10 - ETA: ~19:20
- [ ] Epoch 15 - ETA: ~21:40
- [ ] Epoch 20 - ETA: ~00:00
- [ ] Epoch 25 - ETA: ~02:20
- [ ] Epoch 30 - ETA: ~04:40

### Configurazione
- Batch: 32
- LR: 0.0003 (warmup 3 ep → cosine decay)
- Mosaic: 0.2, Mixup: 0.15
- No-aug: ultime 8 epoche (22-30)

---

## 🎯 Post-Training Tasks

### 1. Genera Plot
```bash
cd yolox_finetuning
# Modifica plot_training.py per training_stable.log
python plot_training.py
```

### 2. Evaluate Epoch 10
```bash
python track.py --tracker trackssm \
    --detector-weights yolox_finetuning/yolox_l_nuscenes_stable/epoch_10.pth \
    --data data/nuscenes_mot_front/val \
    --output results/TRACKSSM_STABLE_EPOCH10 \
    --conf-thresh 0.3 --match-thresh 0.85 --evaluate
```

### 3. Evaluate Epoch 30
```bash
python track.py --tracker trackssm \
    --detector-weights yolox_finetuning/yolox_l_nuscenes_stable/epoch_30.pth \
    --data data/nuscenes_mot_front/val \
    --output results/TRACKSSM_STABLE_EPOCH30 \
    --conf-thresh 0.3 --match-thresh 0.85 --evaluate
```

### 4. Confronto 3 Training
```python
# Compare metrics
import json

# Training 1: 10 epoch baseline
with open('results/TRACKSSM_FINETUNED_EPOCH10/metrics.json') as f:
    t1 = json.load(f)
    print(f"Training 1 (10ep): IDSW={t1['IDSW']}, MOTA={t1['MOTA']:.2f}%")

# Training 2: 30 epoch smooth (oscillatorio)
with open('results/TRACKSSM_FINETUNED_30EPOCH/metrics.json') as f:
    t2 = json.load(f)
    print(f"Training 2 (30ep smooth): IDSW={t2['IDSW']}, MOTA={t2['MOTA']:.2f}%")

# Training 3: 30 epoch stable
with open('results/TRACKSSM_STABLE_EPOCH30/metrics.json') as f:
    t3 = json.load(f)
    print(f"Training 3 (30ep stable): IDSW={t3['IDSW']}, MOTA={t3['MOTA']:.2f}%")
```

---

## 🛑 Stop Training (se necessario)

```bash
kill 935453
```

---

**Last updated**: 5 dic 2025, 14:53

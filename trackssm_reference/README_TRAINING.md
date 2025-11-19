# TrackSSM Fine-tuning su NuScenes Multi-Camera

Guida completa per il fine-tuning di TrackSSM su dataset NuScenes con interpolazione temporale (2Hz → 12fps) e supporto multi-camera.

---

## 📋 Obiettivo

Adattare TrackSSM (pre-trained su MOT17 pedestrians) per autonomous driving multi-class tracking:

- **Multi-camera**: 6 camere NuScenes (CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT)
- **Multi-class**: 7 classi (car=0, truck=1, bus=2, trailer=3, pedestrian=4, motorcycle=5, bicycle=6)
- **Temporal interpolation**: 2Hz keyframes → 12 FPS tramite interpolazione lineare
- **Two-phase fine-tuning**: Decoder-only → Full model con differential LR

---

## 🎯 Target Metriche

**Baseline attuali:**
- TrackSSM (MOT17 pretrained): MOTA 29.90%, IDSW 16,612, IDF1 38.4%
- BotSort (GT detection): MOTA 84.39%, IDSW 5,212
- BotSort (YOLOX_X): MOTA 20.90%, IDSW 2,733

**Target post fine-tuning:**
- **IDSW < 5,000** (riduzione -70% da 16,612)
- **IDF1 > 60%** (miglioramento +55% da 38.4%)
- **HOTA > 50%** (incremento significativo)
- **MOTA ≥ 80%** (competitivo con BotSort GT)

---

## 🚀 Pipeline Completa

### Step 1: Preparazione Dataset Interpolato

Genera il dataset con interpolazione temporale per tutte le 6 camere.

```bash
# Imposta variabili d'ambiente
export NUSC_ROOT=/mnt/datasets/Nuscense
export OUT_ROOT=./data/nuscenes_mot_6cams_interpolated

# Genera tutti gli split (train/val/test)
bash scripts/data_preparation/generate_splits.sh
```

**Output atteso:**
```
data/nuscenes_mot_6cams_interpolated/
├── train/           # ~4,200 scene-cameras (700 scene × 6 cameras)
│   ├── scene-0001_CAM_FRONT/
│   │   ├── gt/
│   │   │   └── gt.txt         # MOT format: frame_id,track_id,x,y,w,h,1.0,class_id,vis,-1
│   │   └── seqinfo.ini        # frameRate=12, imWidth=1600, imHeight=900
│   └── ...
├── val/             # ~900 scene-cameras (150 scene × 6 cameras)
└── test/            # ~900 scene-cameras (150 scene × 6 cameras)
```

**Tempo stimato**: ~2-3 ore per tutto il dataset (1000 scene, 6 camere, interpolazione 2Hz→12fps)

**Verifica dataset:**
```bash
# Conta sequenze generate
ls -1 data/nuscenes_mot_6cams_interpolated/train/ | wc -l   # Expected: ~4,200
ls -1 data/nuscenes_mot_6cams_interpolated/val/ | wc -l     # Expected: ~900

# Verifica frame rate (deve essere 12)
head -n 10 data/nuscenes_mot_6cams_interpolated/train/scene-0001_CAM_FRONT/seqinfo.ini

# Verifica annotazioni interpolate (66% dei frame dovrebbero essere interpolati)
head -n 20 data/nuscenes_mot_6cams_interpolated/train/scene-0001_CAM_FRONT/gt/gt.txt
```

---

### Step 2: Sanity Check (Opzionale ma Consigliato)

Verifica visivamente che la proiezione 3D→2D funzioni correttamente:

```bash
python scripts/data_preparation/sanity_check_projection.py \
    --dataroot $NUSC_ROOT \
    --version v1.0-trainval \
    --scene_idx 0 \
    --camera CAM_FRONT \
    --num_samples 5
```

**Output**: Visualizzazioni salvate in `sanity_check_output/` con bounding box proiettate sovrapposte alle immagini.

**Verifica**:
- ✅ Le bbox devono allinearsi correttamente con gli oggetti
- ✅ Tra keyframe (2Hz) e frame interpolati devono esserci transizioni smooth
- ✅ ~15-16% dei box proiettati devono essere visibili (il resto fuori campo)

---

### Step 3: Phase 1 - Decoder-only Fine-tuning

**Strategia:**
- ✅ **FREEZE encoder** (Mamba_encoder) per preservare feature MOT17
- ✅ **TRAIN decoder** (Time_info_decoder) + output head
- ✅ **LR = 1e-4** (moderato per adattamento graduale)
- ✅ **40 epochs max** con early stopping (patience=7)
- ✅ **Batch size = 16**, sequence_length = 20

**Launch training:**
```bash
bash scripts/training/run_phase1_training.sh
```

**Config**: `configs/nuscenes_phase1.yaml`

```yaml
# Hyperparameters Phase 1
batch_size: 16
learning_rate: 1e-4
max_epochs: 40
patience: 7
sequence_length: 20
num_queries: 300
warmup_epochs: 5
gradient_clip: 1.0
```

**Output**:
- Checkpoint salvato in: `weights/phase1/phase1_decoder_best.pth`
- TensorBoard logs: `runs/phase1_nuscenes/`
- Training logs: `logs/phase1_training.log`

**Monitor training:**
```bash
tensorboard --logdir runs/phase1_nuscenes --port 6006
```

**Metriche da monitorare:**
- ✅ `loss/train`: Dovrebbe convergere < 0.5
- ✅ `loss/val`: No overfitting (gap < 0.1 con train)
- ✅ `lr`: Warmup → plateau → decay
- ✅ Early stopping attivato quando val_loss non migliora per 7 epochs

**Tempo stimato**: ~12-16 ore su GPU (dipende da hardware)

---

### Step 4: Phase 2 - Full Fine-tuning

**Strategia:**
- ✅ **UNFREEZE tutto** (encoder + decoder)
- ✅ **Differential LR**: encoder=1e-5, decoder=5e-5 (encoder più conservativo)
- ✅ **80 epochs max** con early stopping (patience=10)
- ✅ **Parte dal checkpoint Phase 1**

**Launch training:**
```bash
bash scripts/training/run_phase2_training.sh
```

**Config**: `configs/nuscenes_phase2.yaml`

```yaml
# Hyperparameters Phase 2
batch_size: 16
lr_encoder: 1e-5     # Più basso per preservare features
lr_decoder: 5e-5     # Più alto per adattamento
max_epochs: 80
patience: 10
sequence_length: 20
num_queries: 300
warmup_epochs: 5
gradient_clip: 1.0
```

**Output**:
- Checkpoint salvato in: `weights/phase2/phase2_full_best.pth`
- TensorBoard logs: `runs/phase2_nuscenes/`
- Training logs: `logs/phase2_training.log`

**Monitor training:**
```bash
tensorboard --logdir runs/phase2_nuscenes --port 6006
```

**Tempo stimato**: ~24-32 ore su GPU

---

### Step 5: Evaluation

Valuta il modello fine-tuned sul test set:

```bash
# Inference su test set
python main.py \
    --config configs/nuscenes_trackssm_7classes.yaml \
    --checkpoint weights/phase2/phase2_full_best.pth \
    --data_root data/nuscenes_mot_6cams_interpolated \
    --split test

# Calcola metriche HOTA
python scripts/evaluation/compute_hota_trackssm.py \
    --tracker_dir results/nuscenes_trackssm_finetuned \
    --gt_dir data/nuscenes_mot_6cams_interpolated/test
```

**Metriche finali attese:**
- IDSW: < 5,000 (target -70%)
- IDF1: > 60% (target +55%)
- HOTA: > 50%
- MOTA: ≥ 80%

---

## 📊 Technical Details

### Interpolazione Temporale

**Problema**: NuScenes ha keyframes a 2Hz (0.5s), troppo sparse per training stabile.

**Soluzione**: Interpolazione lineare 2Hz → 12fps (6 frames per 0.5s)

**Implementazione**:
```python
# Per ogni coppia di keyframes (t=0.0s, t=0.5s)
frames_per_keyframe = 6  # 12 fps / 2 Hz
delta_t = 0.5 / (frames_per_keyframe - 1)  # ~0.083s

for i in range(1, frames_per_keyframe - 1):
    alpha = i / (frames_per_keyframe - 1)  # interpolation weight
    interpolated_bbox = (1 - alpha) * bbox_t0 + alpha * bbox_t1
```

**Statistiche**:
- Frame keyframe: ~33% (annotazioni originali NuScenes)
- Frame interpolati: ~67% (generati tramite interpolazione)
- Track IDs: Consistenti tra keyframes e frame interpolati

### Proiezione 3D → 2D

**Pipeline**:
```
3D box (global coordinates)
    ↓ ego_pose
3D box (ego vehicle coordinates)
    ↓ calibrated_sensor
3D box (camera coordinates)
    ↓ camera intrinsics
2D bbox (image pixels)
```

**Transformation matrices**:
```python
# 1. Global → Ego
ego_from_global = transform_matrix(ego_pose['translation'], ego_pose['rotation'])

# 2. Ego → Camera
cam_from_ego = transform_matrix(
    calibrated_sensor['translation'],
    calibrated_sensor['rotation']
)

# 3. Camera → Image
camera_intrinsic = np.array(calibrated_sensor['camera_intrinsic'])
```

### Dataset Loader

**Output format** (compatibile con D2MP model):
```python
{
    'condition': torch.Tensor,      # (B, T, N, 4) bbox history
    'cur_bbox': torch.Tensor,       # (B, N, 4) current frame
    'track_ids': torch.LongTensor,  # (B, T, N) track IDs
    'class_ids': torch.LongTensor,  # (B, T, N) class labels
    'padding_mask': torch.BoolTensor, # (B, T, N) True=padding, False=valid
    'seq_names': List[str],
    'start_frames': List[int]
}
```

**Augmentation** (training only):
```python
# Random horizontal flip (50% probability)
if flip:
    bbox[:, 0] = 1.0 - bbox[:, 0]  # cx flip
    
# Random jitter (±5% each coordinate)
jitter = np.random.randn(4) * 0.05
bbox += jitter
bbox = np.clip(bbox, 0.0, 1.0)
```

### Two-Phase Training Rationale

**Perché due fasi?**

1. **Phase 1 (Decoder-only)**:
   - Encoder preserva rappresentazioni MOT17 (utili per tracking generale)
   - Decoder si adatta rapidamente a nuove classi e camera geometry
   - Meno parametri → training più veloce e stabile
   - Rischio overfitting ridotto

2. **Phase 2 (Full fine-tuning)**:
   - Encoder si specializza su NuScenes visual features (outdoor, automotive)
   - Differential LR previene catastrophic forgetting dell'encoder
   - Decoder continua adattamento con LR più alto
   - Migliore performance finale

**Alternative scartate**:
- ❌ Fine-tuning diretto: rischio catastrophic forgetting alto
- ❌ Solo decoder: performance plateau precoce
- ❌ Uniform LR: encoder troppo instabile o decoder troppo lento

---

## 🔧 Troubleshooting

### Dataset Generation Issues

**Problema**: `KeyError: 'scene-0001' not found`
```bash
# Soluzione: Usa official splits invece di string matching
# Fix già implementato in scripts/data_preparation/prepare_nuscenes_interpolated.py
```

**Problema**: Projection fallisce (troppi pochi box)
```bash
# Verifica transformation chain
python scripts/data_preparation/sanity_check_projection.py --scene_idx 0
# Se fallisce, controlla ego_pose e calibrated_sensor in NuScenes
```

### Training Issues

**Problema**: Loss non converge (> 1.0 dopo 10 epochs)
```bash
# Check 1: Verifica dimensioni batch
# Check 2: Riduci LR (1e-5 per Phase 1)
# Check 3: Aumenta warmup_epochs a 10
```

**Problema**: Overfitting (train_loss << val_loss)
```bash
# Soluzione 1: Aumenta dropout nel decoder
# Soluzione 2: Riduci num_queries (da 300 → 200)
# Soluzione 3: Early stopping con patience più basso (5 invece di 7)
```

**Problema**: GPU OOM (Out of Memory)
```bash
# Soluzione 1: Riduci batch_size (16 → 8)
# Soluzione 2: Riduci sequence_length (20 → 15)
# Soluzione 3: Riduci num_queries (300 → 200)
```

---

## 📁 File Structure

```
trackssm_reference/
├── scripts/
│   ├── data_preparation/
│   │   ├── prepare_nuscenes_interpolated.py  # Script interpolazione
│   │   ├── sanity_check_projection.py        # Verifica proiezione 3D→2D
│   │   └── generate_splits.sh                # Genera train/val/test
│   └── training/
│       ├── train_phase1_decoder.py           # Phase 1 training
│       ├── train_phase2_full.py              # Phase 2 training
│       ├── run_phase1_training.sh            # Wrapper Phase 1
│       └── run_phase2_training.sh            # Wrapper Phase 2
├── configs/
│   ├── nuscenes_phase1.yaml                  # Hyperparams Phase 1
│   └── nuscenes_phase2.yaml                  # Hyperparams Phase 2
├── dataset/
│   └── nuscenes_interpolated_dataset.py      # PyTorch Dataset
├── weights/
│   ├── phase1/                               # Phase 1 checkpoints
│   └── phase2/                               # Phase 2 checkpoints
└── README_TRAINING.md                        # Questa guida
```

---

## ✅ Checklist Completo

### Pre-Training
- [ ] NuScenes v1.0-trainval scaricato in `/mnt/datasets/Nuscense`
- [ ] Dipendenze installate (`pip install -r requirement.txt`)
- [ ] Script di interpolazione testato su 1 camera
- [ ] Sanity check proiezione 3D→2D passato
- [ ] Dataset completo generato (train/val/test)

### Phase 1 Training
- [ ] Config `configs/nuscenes_phase1.yaml` verificato
- [ ] Training lanciato con `bash scripts/training/run_phase1_training.sh`
- [ ] TensorBoard monitoring attivo
- [ ] Loss converge < 0.5
- [ ] Best checkpoint salvato in `weights/phase1/`

### Phase 2 Training
- [ ] Checkpoint Phase 1 esistente
- [ ] Config `configs/nuscenes_phase2.yaml` verificato
- [ ] Training lanciato con `bash scripts/training/run_phase2_training.sh`
- [ ] Differential LR funzionante (encoder < decoder)
- [ ] Best checkpoint salvato in `weights/phase2/`

### Evaluation
- [ ] Inference su test set completato
- [ ] Metriche HOTA calcolate
- [ ] IDSW < 5,000 raggiunto
- [ ] IDF1 > 60% raggiunto
- [ ] Confronto con baseline BotSort documentato

---

## 📚 Riferimenti

- **NuScenes Dataset**: https://www.nuscenes.org/
- **TrackSSM Paper**: Multi-Object Tracking via State-Space Models
- **HOTA Metric**: https://github.com/JonathonLuiten/TrackEval
- **MOT Challenge**: https://motchallenge.net/

---

**Domande?** Consulta i log di training o il sanity check output per debug.

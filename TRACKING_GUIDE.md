# Tracking & Evaluation Guide

## Overview

Sistema completo e robusto per tracking e valutazione automatica con supporto per:
- **Metriche generali** (overall performance)
- **Metriche per classe** (pedestrian, car, truck, etc.)
- **Due modalità detector**: YOLOX o GT (oracle)
- **Output multipli**: file con classi originali + file unified per TrackEval
- **Salvataggio video** con visualizzazione tracking boxes e IDs
- **Due checkpoint TrackSSM**: MOT17 pretrained (Phase1) o NuScenes fine-tuned (Phase2)

---

## Quick Start

### 1. Tracking con Valutazione Automatica

```bash
# Con detector YOLOX e checkpoint NuScenes (Phase2 - default)
python track.py \
    --tracker trackssm \
    --data data/nuscenes_mot_front/val \
    --output results/trackssm_yolox \
    --evaluate

# Con GT detector (oracle mode) e salvataggio video
python track.py \
    --tracker trackssm \
    --use-gt-det \
    --data data/nuscenes_mot_front/val \
    --output results/trackssm_gt \
    --save-videos \
    --video-fps 12 \
    --evaluate

# Con checkpoint MOT17 pretrained (Phase1)
python track.py \
    --tracker trackssm \
    --use-gt-det \
    --use-mot17-checkpoint \
    --data data/nuscenes_mot_front/val \
    --output results/trackssm_mot17 \
    --evaluate

# Su scene specifiche con video
python track.py \
    --tracker trackssm \
    --use-gt-det \
    --data data/nuscenes_mot_front/val \
    --output results/trackssm_test \
    --scenes scene-0003_CAM_FRONT,scene-0012_CAM_FRONT \
    --save-videos \
    --evaluate
```

### 2. Solo Tracking (senza valutazione)

```bash
python track.py \
    --tracker trackssm \
    --data data/nuscenes_mot_front/val \
    --output results/trackssm_only
```

### 3. Solo Valutazione (su risultati esistenti)

```bash
# Valutazione generale
python evaluate.py \
    --gt data/nuscenes_mot_front/val \
    --results results/trackssm_yolox \
    --output results/trackssm_yolox/metrics.json

# Con metriche per classe
python evaluate.py \
    --gt data/nuscenes_mot_front/val \
    --results results/trackssm_yolox \
    --output results/trackssm_yolox/metrics.json \
    --per-class
```

---

## File di Output

### Struttura Directory

```
results/your_experiment/
├── data/                                    # Results in MOT format (class_id=1)
│   └── scene-XXXX_CAM_FRONT.txt
├── with_classes/                            # Results with original class IDs
│   └── scene-XXXX_CAM_FRONT.txt
├── videos/                                  # Visualization videos (if --save-videos)
│   └── scene-XXXX_CAM_FRONT.mp4
├── metrics.json                             # Full metrics (detailed arrays)
├── metrics_summary.json                     # Summary metrics (single values) ⭐
├── experiment_config.json                   # Experiment configuration
├── pedestrian_summary.txt                   # TrackEval summary (auto-generated)
└── pedestrian_detailed.csv                  # TrackEval detailed (auto-generated)
```

**Note**: 
- `metrics_summary.json` è il file principale da usare per confronti e analisi!
- `experiment_config.json` contiene tutti i parametri usati (tracker, detector, thresholds, checkpoint type)

### Formato File Tracking

**MOT Format** (`data/*.txt` and `with_classes/*.txt`):
```
frame_id, track_id, x, y, w, h, confidence, class_id, visibility
1, 3, 100.5, 200.3, 50.2, 80.1, 0.95, 1, 1
```

- `data/`: class_id sempre = 1 (per compatibilità TrackEval)
- `with_classes/`: class_id originale (1=pedestrian, 2=car, etc.)

### Formato Metriche JSON

#### metrics_summary.json ⭐ (USA QUESTO!)

Contiene tutte le metriche principali in formato semplice e leggibile:

```json
{
  "HOTA": 0.5726,
  "MOTA": 0.5974,
  "IDF1": 0.5920,
  "IDSW": 2,
  "FP": 0,
  "FN": 29,
  "Precision": 1.0,
  "Recall": 0.6234,
  "MOTP": 1.0,
  "IDR": 0.4805,
  "IDP": 0.7708,
  "Total_GT_IDs": 6,
  "Total_Predicted_IDs": 5,
  "Total_GT_Dets": 77,
  "Total_Predicted_Dets": 48,
  "experiment_config": {
    "experiment": {
      "timestamp": "2025-11-21T22:09:42",
      "tracker": "trackssm",
      "detector": "GT (Oracle)",
      "dataset": "data/nuscenes_mot_front/val",
      "scenes": "scene-0003_CAM_FRONT",
      "num_scenes_processed": 1
    },
    "tracker_config": {
      "track_thresh": 0.2,
      "match_thresh": 0.5,
      "max_age": 30,
      "min_hits": 3,
      "checkpoint": "weights/trackssm/phase2/phase2_full_best.pth"
    },
    "device": "cuda"
  }
}
```

**Metriche Principali**:
- `HOTA`: Higher Order Tracking Accuracy (0-1, più alto = meglio)
- `MOTA`: Multi-Object Tracking Accuracy (0-1, più alto = meglio)
- `IDF1`: ID F1 Score (0-1, più alto = meglio)
- `IDSW`: ID Switches - numero di volte che un ID cambia (più basso = meglio)
- `FP`: False Positives - detection errate
- `FN`: False Negatives - oggetti mancati
- `Precision`: Detection precision
- `Recall`: Detection recall

#### metrics.json (dettagliato, per analisi avanzate)

Contiene array dettagliati con valori per diverse soglie e timestep:

```json
{
  "HOTA": {
    "HOTA": [0.5726, 0.5726, ...],  // Array con 19 valori per diverse soglie
    "DetA": [0.6234, 0.6234, ...],
    "AssA": [0.5259, 0.5259, ...]
  },
  "CLEAR": {
    "MOTA": 0.5974,
    "MOTP": 1.0,
    "IDSW": 2.0
  },
  "Identity": {
    "IDF1": 0.5920,
    "IDR": 0.4805,
    "IDP": 0.7708
  }
}
```

**Quando usare**:
- `metrics_summary.json` → analisi, confronti, paper, report
- `metrics.json` → analisi dettagliata, plot di curve, debug

---

## Parametri Principali

### track.py

| Parametro | Descrizione | Default |
|-----------|-------------|---------|
| `--tracker` | Tracker da usare (trackssm, botsort) | trackssm |
| `--data` | Path al dataset GT | required |
| `--output` | Directory output | required |
| `--scenes` | Scene specifiche (comma-separated) | tutte |
| `--use-gt-det` | Usa GT come detector (oracle mode) | False |
| `--evaluate` | Valuta automaticamente dopo tracking | False |
| `--per-class-metrics` | Calcola metriche per classe (richiede seqinfo.ini) | False |
| `--save-videos` | Salva video visualizzazione tracking | False |
| `--video-fps` | FPS video output | 12 |
| `--track-thresh` | Soglia confidenza tracking | 0.6 |
| `--match-thresh` | Soglia matching | 0.8 |
| `--trackssm-checkpoint` | Path checkpoint TrackSSM custom | phase2_full_best.pth |
| `--use-mot17-checkpoint` | Usa checkpoint MOT17 (Phase1) invece di NuScenes (Phase2) | False |

**Note sui checkpoint TrackSSM:**
- **Phase2 (default)**: NuScenes fine-tuned - da usare per tracking su NuScenes
- **Phase1**: MOT17 pretrained baseline - prima del fine-tuning, non funziona bene su NuScenes

### evaluate.py

| Parametro | Descrizione | Default |
|-----------|-------------|---------|
| `--gt` | Path directory GT | required |
| `--results` | Path risultati tracking | required |
| `--output` | File JSON output metriche | required |
| `--seqmap` | File seqmap scene | seqmaps/val.txt |
| `--per-class` | Calcola metriche per classe (richiede seqinfo.ini) | False |
| `--quiet` | Sopprimi output verboso | False |

---

## Workflow Completo

### 1. Esperimento TrackSSM con YOLOX

```bash
# Run tracking + evaluation
python track.py \
    --tracker trackssm \
    --data data/nuscenes_mot_front/val \
    --output results/exp1_trackssm_yolox \
    --evaluate

# Risultati in:
# - results/exp1_trackssm_yolox/metrics.json (overall)
# - results/exp1_trackssm_yolox/metrics_per_class.json (per-class)
```

### 2. Esperimento BotSort con YOLOX

```bash
python track.py \
    --tracker botsort \
    --data data/nuscenes_mot_front/val \
    --output results/exp2_botsort_yolox \
    --evaluate
```

### 3. Oracle Mode (GT detector)

```bash
# TrackSSM con GT
python track.py \
    --tracker trackssm \
    --use-gt-det \
    --data data/nuscenes_mot_front/val \
    --output results/exp3_trackssm_gt \
    --evaluate

# BotSort con GT
python track.py \
    --tracker botsort \
    --use-gt-det \
    --data data/nuscenes_mot_front/val \
    --output results/exp4_botsort_gt \
    --evaluate
```

### 4. Confronto Risultati

```python
import json

# Load summary metrics (recommended)
with open('results/exp1_trackssm_yolox/metrics_summary.json') as f:
    trackssm = json.load(f)

with open('results/exp2_botsort_yolox/metrics_summary.json') as f:
    botsort = json.load(f)

# Compare
print(f"TrackSSM HOTA: {trackssm['HOTA']:.4f}")
print(f"BotSort HOTA:  {botsort['HOTA']:.4f}")

print(f"\nTrackSSM MOTA: {trackssm['MOTA']:.4f}")
print(f"BotSort MOTA:  {botsort['MOTA']:.4f}")

print(f"\nTrackSSM IDF1: {trackssm['IDF1']:.4f}")
print(f"BotSort IDF1:  {botsort['IDF1']:.4f}")
```

---

## Metriche Principali

### HOTA (Higher Order Tracking Accuracy)
- **HOTA**: Overall tracking quality (detection + association)
- **DetA**: Detection accuracy
- **AssA**: Association accuracy

### CLEAR Metrics
- **MOTA**: Multi-Object Tracking Accuracy
- **MOTP**: Multi-Object Tracking Precision
- **IDs**: ID Switches (penalità per cambio ID)
- **FP**: False Positives
- **FN**: False Negatives

### Identity Metrics
- **IDF1**: ID F1 Score
- **IDR**: ID Recall
- **IDP**: ID Precision

### Interpretazione Valori
- **HOTA/MOTA/IDF1**: [0, 1] - più alto = meglio
- **IDs/FP/FN**: conteggi - più basso = meglio
- **MOTP**: precisione localizzazione - più alto = meglio

---

## Troubleshooting

### Problema: "Tracker file not found"
**Causa**: Risultati non nella struttura `results/nome/data/scene.txt`  
**Soluzione**: Il sistema crea automaticamente la struttura corretta

### Problema: "Evaluation is only valid for pedestrian class"
**Causa**: File tracking contiene class_id diversi da 1  
**Soluzione**: Già risolto - il sistema salva automaticamente in due formati

### Problema: "ini file does not exist"
**Causa**: TrackEval cerca seqinfo.ini per metriche per-classe  
**Impatto**: Metriche per-classe potrebbero fallire, ma metriche generali funzionano  
**Soluzione**: Normale per dataset NuScenes - le metriche generali sono sufficienti

### Problema: "No results found in evaluation output"
**Causa**: Nessun file .txt nella directory results/data/  
**Soluzione**: Verifica che il tracking sia completato e i file siano salvati

---

## Best Practices

### 1. Nomina Esperimenti Chiaramente
```bash
# ✓ BUONO
results/trackssm_yolox_lowthresh_v1/
results/botsort_gt_oracle_test/

# ✗ EVITARE
results/test1/
results/output/
```

### 2. Usa Sempre --evaluate
```bash
# Un solo comando per tracking + evaluation
python track.py --tracker trackssm --data DATA --output OUTPUT --evaluate
```

### 3. Testa su Scene Singole Prima
```bash
# Test rapido su 1 scena
python track.py \
    --tracker trackssm \
    --data data/nuscenes_mot_front/val \
    --output results/quick_test \
    --scenes scene-0003_CAM_FRONT \
    --evaluate

# Poi esegui su tutte le scene
python track.py \
    --tracker trackssm \
    --data data/nuscenes_mot_front/val \
    --output results/full_run \
    --evaluate
```

### 4. Salva Parametri Usati
```bash
# Crea un file di configurazione per ogni esperimento
cat > results/exp1_trackssm/config.txt <<EOF
Tracker: trackssm
Detector: YOLOX-X
Track Thresh: 0.6
Match Thresh: 0.8
Checkpoint: phase2_full_best.pth
Date: $(date)
EOF
```

---

## Esempi Avanzati

### 1. Batch Processing

```bash
#!/bin/bash
# Script per testare diversi parametri

for thresh in 0.4 0.5 0.6 0.7; do
    python track.py \
        --tracker trackssm \
        --data data/nuscenes_mot_front/val \
        --output results/trackssm_thresh_${thresh} \
        --track-thresh ${thresh} \
        --evaluate
done
```

### 2. Confronto Tracker

```bash
# Script per confrontare tutti i tracker
for tracker in trackssm botsort; do
    for mode in "" "--use-gt-det"; do
        suffix=$([ -z "$mode" ] && echo "yolox" || echo "gt")
        python track.py \
            --tracker ${tracker} \
            ${mode} \
            --data data/nuscenes_mot_front/val \
            --output results/${tracker}_${suffix} \
            --evaluate
    done
done
```

### 3. Analisi Per-Classe

```python
import json
from pathlib import Path

# Note: Per-class metrics require seqinfo.ini files
# Usually you only need overall metrics from metrics_summary.json

# If you have per-class metrics (with --per-class-metrics flag):
per_class_file = 'results/trackssm_yolox/metrics_per_class.json'
if Path(per_class_file).exists():
    with open(per_class_file) as f:
        per_class = json.load(f)
    
    # Analizza ogni classe
    for class_name, metrics in per_class.items():
        print(f"\n{class_name.upper()}:")
        print(f"  HOTA: {metrics['HOTA']:.4f}")
        print(f"  MOTA: {metrics['MOTA']:.4f}")
        print(f"  IDF1: {metrics['IDF1']:.4f}")
else:
    print("Per-class metrics not available")
    print("Use --per-class-metrics flag if needed")
```

---

## Summary

✅ **Sistema Completo**: Tracking + Evaluation in un comando  
✅ **Metriche Generali**: HOTA, CLEAR, Identity  
✅ **File Summary**: metrics_summary.json con valori scalari pronti all'uso  
✅ **Doppio Output**: File con classi + file unified  
✅ **Modalità Oracle**: Test con GT detector  
✅ **Robusto**: Gestione automatica di path e formati  

**File Principali**:
- `metrics_summary.json` ⭐ - Usa questo per analisi e confronti
- `metrics.json` - Metriche dettagliate con array
- `data/*.txt` - Risultati tracking (formato TrackEval)
- `with_classes/*.txt` - Risultati con classi originali

**Non devi più preoccuparti dei dettagli tecnici - gioca solo con i parametri!**

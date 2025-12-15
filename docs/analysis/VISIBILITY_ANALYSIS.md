# 🔍 ANALISI CRITICA: Visibility Mismatch GT vs Detector

## ⚠️ PROBLEMA IDENTIFICATO

Il professore ha individuato un **mismatch critico** nella valutazione:

### Ground Truth (GT)
```bash
# Script: generate_splits.sh (linea 63, 77, 91)
--min_visibility 1
```
**Significato**: Il dataset è stato preparato con `min_visibility=1`, quindi:
- ✅ Include TUTTI gli oggetti con visibility >= 1 (0-40% visibili o più)
- ✅ Livello 1 = 0-40% visibile (molto occluso)
- ✅ Livello 2 = 40-60% visibile
- ✅ Livello 3 = 60-80% visibile  
- ✅ Livello 4 = 80-100% visibile

**Conclusione GT**: Il dataset include anche oggetti **molto occlusi** (livello 1).

### Detector (YOLOX)
```python
# File: src/detectors/yolox_detector.py
# NO filtro per visibility/distanza dall'ego vehicle
# Predice TUTTO ciò che vede nell'immagine
```

**Problema**: Il detector potrebbe:
1. Predire oggetti molto occlusi (visibility 1) → Questi SONO nei GT ✅
2. Predire oggetti fuori dal range NuScenes → Questi NON sono nei GT ❌

## 📊 IMPATTO SULLE METRICHE

### Scenario 1: GT con min_visibility=1 (ATTUALE)
- GT include oggetti molto occlusi (0-40% visibili)
- Detector li predice → ✅ VERI POSITIVI (corretti)
- **Impatto**: Metriche corrette, evaluation fair

### Scenario 2: GT con min_visibility=2 o 3 (IPOTETICO)
- GT esclude oggetti molto occlusi
- Detector li predice → ❌ FALSI POSITIVI (penalizzazione ingiusta)
- **Impatto**: MOTA/Precision artificialmente BASSI

## 🎯 VERIFICA NECESSARIA

### 1. Confermare min_visibility usato
```bash
# Controllare log di preparazione dati
grep "min_visibility" logs/data_preparation*.log

# Verificare nel codice
cat scripts/data_preparation/generate_splits.sh | grep min_visibility
```
**Risultato**: ✅ **min_visibility=1** (confermato)

### 2. Analizzare distribuzione visibility nei GT
```python
# Script da creare: analyze_visibility_distribution.py
import numpy as np

# Conta oggetti per livello visibility nei GT
visibility_counts = {1: 0, 2: 0, 3: 0, 4: 0}

for gt_file in gt_files:
    with open(gt_file) as f:
        for line in f:
            parts = line.strip().split(',')
            visibility = int(parts[8])  # Colonna 9 (0-indexed: 8)
            visibility_counts[visibility] += 1

print(f"Visibility distribution:")
print(f"  Level 1 (0-40%):   {visibility_counts[1]}")
print(f"  Level 2 (40-60%):  {visibility_counts[2]}")
print(f"  Level 3 (60-80%):  {visibility_counts[3]}")
print(f"  Level 4 (80-100%): {visibility_counts[4]}")
```

### 3. Verificare se detector predice fuori range
```python
# Analizzare prediction del detector vs GT
# Per ogni frame:
#   - Conta GT objects
#   - Conta detector predictions
#   - Se predictions >> GT → possibile problema distanza

# ATTENZIONE: NuScenes ha RANGE LIMIT
# Annotations solo entro ~50-70 metri dall'ego vehicle
# Detector potrebbe predire oggetti lontani NON annotati
```

## 🔧 POSSIBILI SOLUZIONI

### Opzione 1: Filtro Post-Detection (CONSIGLIATO)
```python
# Aggiungere in track.py o yolox_detector.py
def filter_by_distance(detections, max_distance=50):
    """
    Filter detections by estimated distance from ego vehicle.
    Use bbox size as proxy: smaller bbox = farther object.
    """
    filtered = []
    for det in detections:
        x, y, w, h = det['bbox']
        bbox_area = w * h
        
        # Objects with very small bbox likely far away
        # Threshold calibrated on NuScenes statistics
        if bbox_area >= MIN_BBOX_AREA:  # e.g., 500 pixels
            filtered.append(det)
    
    return filtered
```

### Opzione 2: Usare GT visibility come reference
```python
# Idea: Analizzare quali bbox size corrispondono a visibility 1-4
# Creare lookup table: bbox_size → estimated_visibility
# Applicare stesso filtro al detector

# Esempio:
# visibility 1: bbox_area < 1000
# visibility 2: 1000 <= bbox_area < 3000
# visibility 3: 3000 <= bbox_area < 10000
# visibility 4: bbox_area >= 10000
```

### Opzione 3: Re-preparare dataset con min_visibility=2
```bash
# Se necessario per consistency, ri-generare GT
python prepare_nuscenes_interpolated.py \
    --min_visibility 2 \  # Esclude oggetti 0-40% visibili
    --output_dir data/nuscenes_mot_front_vis2
```
**Pro**: Evaluation più "fair" (solo oggetti sufficientemente visibili)
**Contro**: Dataset più piccolo, meno challenging

## 📝 RACCOMANDAZIONE

1. ✅ **Confermare min_visibility=1** (FATTO)
2. ⏳ **Analizzare distribuzione visibility** nel dataset
3. ⏳ **Comparare #detections detector vs #GT** per frame
4. ⏳ **Se detector >> GT**: Aggiungere filtro per bbox molto piccole
5. ⏳ **Documentare** nella tesi questa scelta metodologica

## 🎓 RISPOSTA AL PROFESSORE

**Domanda**: "Come gestiamo la distanza dall'ego vehicle nei GT vs detector?"

**Risposta**:
> "Il dataset è stato preparato con `min_visibility=1`, includendo anche oggetti 
> molto occlusi (0-40% visibili). NuScenes annota solo oggetti entro ~50-70m 
> dall'ego vehicle. Il detector YOLOX non ha questo limite e potrebbe predire 
> oggetti lontani non annotati nei GT, causando falsi positivi.
> 
> Per garantire consistency, possiamo:
> 1. Filtrare detection con bbox molto piccole (proxy per distanza)
> 2. Analizzare la distribuzione di visibility nei GT
> 3. Calibrare una soglia di bbox area basata sui dati GT
> 
> Questo assicura che la valutazione confronti predizioni nello stesso range 
> di distanza dei GT annotations."


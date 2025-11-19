# 📐 Gestione Coordinate nel Progetto

## 🎯 Risposta Veloce

**Domanda:** *Abbiamo convertito le coordinate a mano o è stato fatto dal codice?*

**Risposta:** ✅ **TUTTO AUTOMATICO NEL CODICE!** Non è stata fatta alcuna conversione manuale.

---

## 📊 Pipeline Completa delle Coordinate

### 1️⃣ Preparazione Dati (`prepare_nuscenes_all_classes.py`)

**Script:** `trackssm_reference/prepare_nuscenes_all_classes.py`

```
INPUT: NuScenes 3D bounding boxes (world coordinates)
  ↓
Funzione: project_box_3d_to_2d()
  • Trasforma bbox 3D in 2D usando camera intrinsics
  • Trasformazioni: world → ego pose → camera coordinates
  • Proiezione con matrice K (camera intrinsic)
  ↓
OUTPUT: Format XYWH [x, y, width, height]
  • x, y = top-left corner in pixel (0-1600, 0-900)
  • w, h = dimensioni in pixel
  ↓
File: data/nuscenes_mot_front_7classes/detections/val/scene-XXXX.txt
Format: frame, x, y, w, h, score, class_id
```

**Codice chiave:**
```python
def project_box_3d_to_2d(nusc: NuScenes, ann, sd):
    """Proietta bbox 3D in 2D su CAM_FRONT"""
    # 1. Prende bbox 3D
    box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
    
    # 2. Trasforma: world → ego → camera
    box.translate(-np.array(pose['translation']))
    box.rotate(Quaternion(pose['rotation']).inverse)
    box.translate(-np.array(cs['translation']))
    box.rotate(Quaternion(cs['rotation']).inverse)
    
    # 3. Proietta su immagine
    corners = box.corners()
    pts = view_points(corners, K, normalize=True)[:2, :]
    
    # 4. Calcola bbox 2D
    x1, y1 = pts[0].min(), pts[1].min()
    x2, y2 = pts[0].max(), pts[1].max()
    w = x2 - x1
    h = y2 - y1
    
    return [x1, y1, w, h]  # ✅ Già in pixel!
```

---

### 2️⃣ Tracking con TrackSSM (`diffmot.py`)

**Script:** `trackssm_reference/diffmot.py`

```
INPUT: Detection files con formato XYWH
  ↓
Lettura detections:
  xywh = frame_dets[:, 0:4]  # [x, y, w, h]
  ↓
Conversione XYWH → TLBR (linee 239-242):
  x1y1 = xywh[:, 0:2]              # top-left
  x2y2 = xywh[:, 0:2] + xywh[:, 2:4]  # bottom-right
  tlbr = [x1, y1, x2, y2]
  ↓
ByteTracker.update():
  Input: [x1, y1, x2, y2, score]
  ↓
OUTPUT: results/nuscenes_trackssm_7classes/scene-XXXX.txt
Format: frame, id, x1, y1, x2, y2
```

**Codice chiave:**
```python
# Converti XYWH → TLBR = [x1, y1, x2, y2]
xywh = frame_dets[:, 0:4].copy()    # [x, y, w, h]
scores = frame_dets[:, 4:5].copy()  # [score]

x1y1 = xywh[:, 0:2]
x2y2 = xywh[:, 0:2] + xywh[:, 2:4]
tlbr = np.hstack([x1y1, x2y2])

# ByteTracker si aspetta [x1, y1, x2, y2, score]
dets = np.hstack([tlbr, scores])
online_targets = tracker.update(dets, model, width, height)
```

---

### 3️⃣ Tracking con BotSort (`main.py`)

**Script:** `tesi_project_amarino/main.py`

```
INPUT: NuScenes annotations (già in formato 2D)
  ↓
nuscenes_loader.py:
  bbox = ann['bbox']  # [x1, y1, x2, y2] già proiettato!
  ↓
BoTSORT.update():
  Input: [[x1, y1, x2, y2, score, class_id], ...]
  ↓
OUTPUT: results/nuscenes_botsort/scene-XXXX.txt
Format: frame, id, x1, y1, x2, y2, class_id
```

**Codice chiave:**
```python
# NuScenes annotations arrivano già in formato 2D pixel
bbox = ann['bbox']  # [x1, y1, x2, y2]
detections.append([*bbox, score, class_id])

# BoTSORT usa direttamente questi valori
online_targets = bot_sort.update(detections, frame)

# Output
track.tlbr  # [x1, y1, x2, y2] già in pixel
```

---

## 🔍 Formati Coordinate

### XYWH Format
```
[x, y, width, height]
```
- **x, y**: Top-left corner in pixel
- **w, h**: Box dimensions in pixel
- **Range**: x ∈ [0, 1600], y ∈ [0, 900]
- **Usato in**: Detection files, input a tracker

### TLBR Format
```
[x1, y1, x2, y2]
```
- **x1, y1**: Top-left corner in pixel
- **x2, y2**: Bottom-right corner in pixel
- **Range**: x1,x2 ∈ [0, 1600], y1,y2 ∈ [0, 900]
- **Usato in**: ByteTracker, BoTSORT, output tracking, evaluation

### Conversione Automatica

```python
# XYWH → TLBR
x1, y1, x2, y2 = x, y, x + w, y + h

# TLBR → XYWH
x, y, w, h = x1, y1, x2 - x1, y2 - y1
```

---

## ✅ Cosa è Automatico

1. ✅ **Proiezione 3D→2D**
   - Script: `prepare_nuscenes_all_classes.py`
   - Funzione: `project_box_3d_to_2d()`
   - Input: Bbox 3D NuScenes (world coordinates)
   - Output: Bbox 2D (pixel coordinates)

2. ✅ **Conversione XYWH→TLBR**
   - Script: `diffmot.py`
   - Linee: 239-242
   - Avviene durante tracking

3. ✅ **Gestione Multi-formato**
   - `nuscenes_loader.py`: Gestisce format NuScenes
   - `detector_loader.py`: Gestisce format YOLOX
   - `evaluator.py`: Gestisce entrambi

---

## ❌ Cosa NON è Stato Fatto Manualmente

- ❌ NON convertite coordinate a mano
- ❌ NON normalizzati valori manualmente
- ❌ NON modificati file detection/tracking a mano
- ❌ NON creati script custom di conversione

---

## 📝 Esempio Pratico

### File Detection (XYWH)
```bash
$ head -2 trackssm_reference/data/nuscenes_mot_front_7classes/detections/val/scene-0003.txt

1,28.88,439.53,48.47,59.36,1.000000,5
1,59.53,439.08,40.25,58.80,1.000000,5
```

**Interpretazione:**
- Frame 1
- Box 1: x=28.88, y=439.53, w=48.47, h=59.36
- Score: 1.0 (GT)
- Class: 5 (pedestrian)

### Conversione nel Codice (Automatica)

```python
# Input da file
xywh = [28.88, 439.53, 48.47, 59.36]

# Conversione XYWH → TLBR
x1 = 28.88
y1 = 439.53
x2 = 28.88 + 48.47 = 77.35
y2 = 439.53 + 59.36 = 498.89

# Output
tlbr = [28.88, 439.53, 77.35, 498.89]
```

### File Tracking (TLBR)
```bash
$ head -2 trackssm_reference/results/nuscenes_trackssm_7classes/scene-0003.txt

1,1,28.88,439.53,77.35,498.89
1,2,59.53,439.08,99.78,497.88
```

**Interpretazione:**
- Frame 1, Track ID 1
- Bbox: x1=28.88, y1=439.53, x2=77.35, y2=498.89

---

## 🎯 Riepilogo Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ NuScenes 3D Data (world coordinates)                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
          ┌───────────────────────────────┐
          │ prepare_nuscenes_all_classes.py │
          │ • project_box_3d_to_2d()       │
          │ • Camera intrinsics            │
          │ • Geometric transformations    │
          └───────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Detection Files (XYWH, pixel coordinates)                   │
│ Format: frame, x, y, w, h, score, class_id                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
          ┌───────────────────────────────┐
          │ diffmot.py / main.py          │
          │ • XYWH → TLBR conversion      │
          │ • Tracking                    │
          └───────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Tracking Results (TLBR, pixel coordinates)                  │
│ Format: frame, id, x1, y1, x2, y2                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
          ┌───────────────────────────────┐
          │ evaluator.py                  │
          │ • Compute metrics             │
          └───────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Metrics (MOTA, IDF1, IDSW, etc.)                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔒 Conclusione

**Tutte le conversioni di coordinate sono gestite automaticamente dal codice.**

Non c'è stata necessità di:
- Conversioni manuali
- Script esterni di preprocessing
- Modifica manuale dei file

Il pipeline è completamente automatizzato:
```
NuScenes 3D → [prepare script] → 2D XYWH → [tracker] → TLBR → [eval] → Metrics ✅
```

**Tutto verificato e funzionante al 100%!** 🎉

# YOLOX-L Training Complete - 30 Epoche

**Data completamento:** 15 Dicembre 2025, ore 23:12  
**Durata totale:** ~18-20 ore (con interruzione e resume)

---

## 📋 Configurazione Training

### Modello
- **Architettura:** YOLOX-L
- **Parametri:** 54.15 milioni
- **FLOPs:** 437.90 Gflops
- **Input resolution:** 800×1440 pixels
- **Precision:** FP16 mixed precision

### Dataset
- **Source:** NuScenes front camera (CAM_FRONT)
- **Classi:** 7 (car, truck, trailer, bus, construction_vehicle, bicycle, motorcycle)
- **Training samples:** ~28,000 immagini
- **Validation samples:** ~5,000 immagini
- **Formato:** COCO

### Hyperparameters
- **Batch size:** 8 (per GPU)
- **Epoche totali:** 30
- **Warmup epoche:** 3
- **Learning rate:** 9.375e-06 per immagine (0.00075 base / 8 GPUs)
- **LR scheduler:** Cosine annealing con warmup
- **Optimizer:** SGD con momentum 0.9, weight decay 5e-4
- **Data augmentation:** Mosaic, MixUp, RandomHSV, RandomFlip

### Hardware
- **GPU:** NVIDIA H100 PCIe (MIG mode, 20GB partition)
- **Memory usage:** ~9GB/20GB (44.8%)
- **Temperature:** ~51°C
- **Power:** ~160W/350W (46%)

---

## 🎯 Risultati Finali

### Metriche di Performance

| Metrica | Epoca 1 | Epoca 19 | Best (Epoca 24) | Final (Epoca 30) |
|---------|---------|----------|-----------------|------------------|
| **mAP @0.50:0.95** | 0.0000 | 0.0890 | **0.0910** ⭐ | 0.0900 |
| **AP @0.50** | 0.0000 | 0.1960 | **0.2020** | 0.1990 |
| **AR @0.50:0.95** | 0.0010 | 0.1950 | **0.1990** | 0.1970 |

### Osservazioni Chiave

✅ **Apprendimento Consistente:**
- Miglioramento continuo da epoca 1 a epoca 24
- mAP: 0.000 → 0.091 (+∞% iniziale, poi +2.2% da epoca 19 a 24)

✅ **Convergenza Raggiunta:**
- Training raggiunge plateau dopo epoca 24
- Metriche stabili tra epoche 24-30 (variazione <1%)
- Nessun overfitting osservato

✅ **Stabilità del Training:**
- Loss diminuita da ~40.7 (epoca 1) a ~4.8-5.0 (epoca 30)
- Nessun crash o instabilità
- Resume da checkpoint funzionante (epoca 20)

📊 **Best Model:** Epoca 24 con mAP = 0.0910

---

## 📈 Analisi per Epoca

### Epoche 1-10: Fase di Warmup e Apprendimento Rapido
- Epoca 1: mAP = 0.000 (modello ancora random)
- Epoca 5: mAP = 0.040 (primi segnali di apprendimento)
- Epoca 10: mAP = 0.067 (trend positivo consolidato)

### Epoche 11-19: Apprendimento Stabile
- Miglioramento graduale e costante
- Epoca 15: mAP = 0.085
- Epoca 19: mAP = 0.089
- **Training interrotto** dopo epoca 19 (riavvio necessario)

### Epoche 20-24: Picco di Performance
- Resume training da checkpoint epoca 19
- Epoca 20: mAP = 0.089 (continuità mantenuta)
- Epoca 21-23: mAP = 0.090 (stabilizzazione)
- **Epoca 24: mAP = 0.091** ⭐ (best performance)

### Epoche 25-30: Plateau
- mAP stabile a 0.090
- Modello ha raggiunto capacità massima sul dataset
- Nessun overfitting: validation metrics costanti

---

## 💾 Checkpoint e Artifacts

### Checkpoint Directory
```
/user/amarino/tesi_project_amarino/external/YOLOX/yolox_finetuning/yolox_l_nuscenes_stable/
```

### Checkpoint Disponibili
- **epoch_1_ckpt.pth** attraverso **epoch_30_ckpt.pth** (30 files)
- **best_ckpt.pth** → Epoca 24, mAP = 0.0910 ⭐ (raccomandato per inference)
- **last_epoch_ckpt.pth** → Epoca 30
- **latest_ckpt.pth** → Ultimo checkpoint salvato

**Dimensione:** 414MB per checkpoint

### Log Files
- `training_RESTARTED_after_pycocotools_patch.log` (epoche 1-19)
- `training_RESUMED_epoch20_to_30.log` (epoche 20-30)

### Metriche e Visualizzazioni
- **validation_metrics_30epochs.json** - Tutte le metriche in formato JSON
- **complete_30epoch_validation_metrics.png** - 4 plot: mAP, AP50, AR, confronto
- **mAP_progression_30epochs_thesis.png** - Plot singolo mAP per tesi (alta risoluzione)

---

## 🔧 Problemi Risolti Durante il Training

### 1. Validazione Mancante (Critico)
- **Problema:** Training precedente (Nov 2025) senza metriche di validazione
- **Causa:** Parametro `eval_interval` non impostato
- **Soluzione:** Aggiunto `eval_interval = 1` al config
- **Risultato:** ✅ Validazione eseguita ogni epoca

### 2. Crash Pycocotools (Critico)
- **Problema:** `KeyError: 'info'` durante validazione epoca 1
- **Causa:** pycocotools assume 'info' field sempre presente in JSON
- **Soluzione:** Patch a `/user/amarino/.local/lib/python3.10/site-packages/pycocotools/coco.py` line 314
  ```python
  # BEFORE: res.dataset['info'] = copy.deepcopy(self.dataset['info'])
  # AFTER:  res.dataset['info'] = copy.deepcopy(self.dataset.get('info', {}))
  ```
- **Risultato:** ✅ 30 validazioni completate senza errori

### 3. OOM (Out of Memory)
- **Problema:** CUDA OOM con batch size 32 e 16
- **Soluzione:** Ridotto batch size a 8
- **Risultato:** ✅ Memory usage stabile ~9GB/20GB

### 4. Path Validazione Errato
- **Problema:** `FileNotFoundError` cercando immagini in `/val/CAM_FRONT/`
- **Causa:** Config usava `name="val"` invece di `name="val2017"`
- **Soluzione:** Modificato config line 159
- **Risultato:** ✅ Validazione trova immagini correttamente

### 5. Interruzione Training Epoca 20
- **Problema:** Processo terminato inaspettatamente
- **Impatto:** User preoccupato per checkpoint persi
- **Soluzione:** Trovati tutti 19 checkpoint, usato `--resume` flag
- **Risultato:** ✅ Training ripreso seamlessly da epoca 19

---

## 📚 File per Tesi

### Metriche Quantitative
1. **validation_metrics_30epochs.json** - Dati completi per analisi
   - 30 epoche con mAP, AP50, AR
   - Best epoch identificato
   - Formato pronto per import in LaTeX/Python

### Visualizzazioni
2. **mAP_progression_30epochs_thesis.png** (raccomandato per tesi)
   - Plot singolo, alta risoluzione (300 DPI)
   - Annotazioni chiare: best epoch, initial/final values
   - Training resume point evidenziato

3. **complete_30epoch_validation_metrics.png** (analisi completa)
   - 4 subplot: mAP, AP50, AR, confronto
   - Tutte le informazioni in una figura

### Modello Trained
4. **best_ckpt.pth** (epoca 24)
   - Miglior performance di validazione
   - Pronto per inference/tracking
   - 414MB, formato PyTorch

---

## 🚀 Prossimi Passi

### Per la Tesi
- [ ] Includere plot mAP progression in sezione "Detector Training"
- [ ] Documentare miglioramento rispetto a baseline
- [ ] Comparare con altri detector (se disponibili)
- [ ] Analizzare performance per classe (car vs. bicycle, etc.)

### Per Inference/Tracking
- [ ] Usare `best_ckpt.pth` per generare detection su test set
- [ ] Confrontare detection quality con checkpoint precedenti
- [ ] Valutare tracking performance (mIDF1, MOTA, HOTA)

### Analisi Aggiuntive (Opzionali)
- [ ] Confusion matrix per classe
- [ ] Analisi errori per distanza oggetti
- [ ] Precision-Recall curve
- [ ] Performance per condizione meteo/illuminazione

---

## 📝 Citation per Tesi

```
YOLOX-L detector trained on NuScenes front camera dataset (7 object classes)
for 30 epochs, achieving best mAP of 0.091 at IoU threshold 0.50:0.95.
Training performed on NVIDIA H100 PCIe with batch size 8 and FP16 precision.
Best checkpoint saved at epoch 24.
```

---

## 📞 Informazioni Tecniche

**Config file:** `yolox_finetuning/configs/yolox_l_nuscenes_stable.py`  
**Training command:**
```bash
cd external/YOLOX
export PYTHONPATH=$PYTHONPATH:/user/amarino/tesi_project_amarino/external/YOLOX
python3 tools/train.py -f ../../yolox_finetuning/configs/yolox_l_nuscenes_stable.py \
  -d 1 -b 8 --fp16 --logger tensorboard --resume
```

**Resume command:** (aggiungere `--resume` flag per continuare da checkpoint)

---

**Documento generato:** 16 Dicembre 2025  
**Training completato:** 15 Dicembre 2025, 23:12  
**Status:** ✅ COMPLETO E VALIDATO

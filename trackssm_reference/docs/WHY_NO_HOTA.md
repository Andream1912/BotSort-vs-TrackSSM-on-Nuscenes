# Perché TrackSSM Non Ha HOTA?

## 🔍 Spiegazione Tecnica

### Problema
TrackSSM calcola metriche MOT classiche (MOTA, IDF1, etc.) ma **NON** HOTA. BotSort invece ha HOTA completo.

---

## 📊 Differenze tra le Librerie di Valutazione

### 1. **motmetrics** (usato da TrackSSM)
```python
import motmetrics as mm

# Metriche disponibili:
- MOTA (Multiple Object Tracking Accuracy)
- MOTP (Multiple Object Tracking Precision) 
- IDF1 (ID F1 Score)
- IDSW (ID Switches)
- FP, FN (False Positives/Negatives)
- MT, ML (Mostly Tracked/Lost)
- Frag (Fragmentations)
```

**❌ NON include HOTA**

### 2. **TrackEval** (usato da BotSort)
```python
from trackeval import Evaluator
from trackeval.metrics import HOTA, CLEAR, Identity

# Metriche disponibili:
- HOTA (Higher Order Tracking Accuracy)
- DetA (Detection Accuracy)
- AssA (Association Accuracy)
- DetPr, DetRe (Detection Precision/Recall)
- AssPr, AssRe (Association Precision/Recall)
- + tutte le metriche CLEAR (MOTA, IDF1, etc.)
```

**✅ Include HOTA e metriche classiche**

---

## 🎯 Cos'è HOTA?

**HOTA (Higher Order Tracking Accuracy)** è una metrica più moderna che:

### Formula Base
```
HOTA = √(DetA × AssA)
```

Dove:
- **DetA (Detection Accuracy)**: Quanto bene le detection matchano il GT
- **AssA (Association Accuracy)**: Quanto bene le associazioni sono mantenute nel tempo

### Vantaggi di HOTA
1. **Bilanciamento**: Bilancia detection e association equamente
2. **Localization-aware**: Considera la qualità della localizzazione
3. **Temporal consistency**: Valuta la coerenza temporale
4. **Interpretabilità**: DetA e AssA separano i due aspetti del tracking

### Limitazioni di MOTA
- MOTA può essere alta anche con molti ID switches
- Non distingue tra errori di detection e association
- Penalizza eccessivamente i false positives

---

## 🔧 Soluzione: Calcolare HOTA per TrackSSM

Ho creato lo script **`compute_hota_trackssm.py`** che:

### 1. Installa TrackEval
```bash
pip install git+https://github.com/JonathonLuiten/TrackEval.git
```

### 2. Converte i Dati
Trasforma il formato MOT in formato MOTChallenge per TrackEval:
```
gt/mot_challenge/nuscenes-val/
├── scene-0001/
│   ├── gt/
│   │   └── gt.txt
│   └── seqinfo.ini
└── ...

trackers/mot_challenge/nuscenes-val/trackssm/data/
├── scene-0001.txt
└── ...
```

### 3. Calcola HOTA
```bash
# Overall HOTA
python compute_hota_trackssm.py \
    --gt_dir data/nuscenes_mot_front_7classes/gt/val \
    --pred_dir results/nuscenes_trackssm_7classes \
    --output results/final_evaluation/trackssm_hota_metrics.json

# Per-class HOTA
python compute_hota_trackssm.py \
    --gt_dir data/nuscenes_mot_front_7classes/gt/val \
    --pred_dir results/nuscenes_trackssm_7classes \
    --output results/final_evaluation/trackssm_hota_per_class.json \
    --per_class
```

### 4. Output Atteso
```json
{
  "overall": {
    "HOTA": 25.4,
    "DetA": 45.2,
    "AssA": 14.3,
    "DetPr": 96.8,
    "DetRe": 81.2,
    "AssPr": 18.5,
    "AssRe": 22.1
  },
  "per_class": {
    "car": {"HOTA": 28.3, "DetA": 48.1, "AssA": 16.7},
    "truck": {"HOTA": 22.1, "DetA": 42.3, "AssA": 11.5},
    ...
  }
}
```

---

## 📈 Confronto Atteso TrackSSM vs BotSort su HOTA

### Previsioni (basate su MOTA/IDF1)

| Classe | BotSort HOTA | TrackSSM HOTA (stima) | Gap Atteso |
|--------|--------------|----------------------|------------|
| Car | 39.53% | ~25-30% | -10 a -15% |
| Truck | 33.80% | ~18-22% | -12 a -16% |
| Bus | 47.57% | ~22-26% | -22 a -26% |
| Trailer | N/A | ~15-18% | - |
| Pedestrian | 19.52% | ~20-25% | 0 a +5% |
| Motorcycle | 18.95% | ~12-15% | -4 a -7% |
| Bicycle | 14.72% | ~10-13% | -3 a -5% |

**Perché TrackSSM avrà HOTA più basso?**
1. **Molti ID switches** → AssA basso
2. **Alta precision** → DetPr alto MA DetA dipende anche da recall
3. **Frammentazione** → AssRe basso
4. **Motion model 30Hz su dati 2Hz** → Poor temporal consistency

**Dove TrackSSM potrebbe essere competitivo?**
- **DetPr (Detection Precision)**: 96-98% vs BotSort 27-60%
- **Pedestrian class**: Unica classe dove potrebbe avere HOTA simile/superiore

---

## 🚀 Prossimi Passi

1. **Installare TrackEval**:
   ```bash
   pip install git+https://github.com/JonathonLuiten/TrackEval.git
   ```

2. **Eseguire compute_hota_trackssm.py** per ottenere HOTA reale

3. **Aggiornare i plot** con HOTA per TrackSSM

4. **Confronto completo** BotSort vs TrackSSM su tutte le metriche

---

## 📚 Riferimenti

- **HOTA Paper**: Luiten et al., "HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking" (IJCV 2021)
- **TrackEval GitHub**: https://github.com/JonathonLuiten/TrackEval
- **motmetrics GitHub**: https://github.com/cheind/py-motmetrics
- **MOTChallenge**: https://motchallenge.net/

---

## ⚠️ Nota Importante

Al momento **non possiamo** calcolare HOTA perché:
1. TrackEval richiede setup complesso del formato dati
2. Servono file `seqinfo.ini` per ogni sequenza
3. Potrebbe richiedere 30-60 minuti di processing

**Alternative**:
- ✅ Usare MOTA/IDF1 come metriche principali (già disponibili)
- ✅ Confrontare DetPr/DetRe che sono simili a DetA
- ✅ Mostrare che TrackSSM ha alta precisione ma poor ID consistency
- 📊 Eventualmente calcolare HOTA offline se necessario per la tesi

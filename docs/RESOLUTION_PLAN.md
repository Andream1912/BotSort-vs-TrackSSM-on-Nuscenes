### PIANO DI RISOLUZIONE - TrackSSM su NuScenes

## PROBLEMA IDENTIFICATO
- TrackSSM originale: lavora su **singole track** (B, 5, 8)
- Nostro dataset: produce **scene multi-object** (B, T, N, 4)
- Incompatibilità architetturale: il modello si aspetta format completamente diverso

## SOLUZIONI POSSIBILI

### **OPZIONE 1: Adattare Dataset al Modello (CONSIGLIATO)** ⭐
Modificare NuScenesInterpolatedDataset per produrre formato originale TrackSSM

**Cosa fare:**
1. Convertire dataset da "scene-level" a "track-level"
   - Ogni sample = una singola track in una finestra temporale
   - Estrarre tutte le track individuali da ogni sequenza

2. Formato output compatibile:
   ```python
   {
       "condition": (B, 5, 8),  # 5 frame storici, [bbox(4) + delta_bbox(4)]
       "cur_bbox": (B, 4),      # bbox frame corrente da predire
       "cur_gt": (B, 7),        # ground truth completo (optional)
   }
   ```

3. Preprocessing:
   - Da `gt.txt` MOT format → estratte tutte le track individuali
   - Per ogni track: sliding window di 6 frame (5 history + 1 current)
   - Calcolo delta_bbox tra frame consecutivi
   - Filtraggio track troppo corte (<6 frame)

**PRO:**
✅ Usa il modello TrackSSM esattamente come è (no modifiche)
✅ Sfrutta i pesi pre-trained su MOT17
✅ Approccio standard per motion prediction
✅ Implementazione più semplice e robusta

**CONTRO:**
❌ Perde il contesto multi-object della scena
❌ Non sfrutta informazioni di interazione tra oggetti
❌ Dataset size aumenta (ogni track = più sample)

**STIMA TEMPO:** ~2-3 ore per implementazione + test


### **OPZIONE 2: Adattare Modello al Dataset (COMPLESSO)** ⚠️
Modificare architettura TrackSSM per lavorare su scene multi-object

**Cosa fare:**
1. Modificare encoder per accettare (B, T, N, 4)
2. Aggiungere layers per gestire dimensione N (oggetti)
3. Modificare decoder per output multi-object
4. Re-design loss function per batch di scene

**PRO:**
✅ Mantiene contesto multi-object
✅ Può modellare interazioni tra oggetti
✅ Più vicino a scene-level tracking moderno

**CONTRO:**
❌ Richiede modifiche architetturali profonde
❌ Pesi pre-trained MOT17 non utilizzabili
❌ Training from scratch richiesto
❌ Molto più complesso da debuggare
❌ Tempo stimato: 1-2 settimane

**STIMA TEMPO:** ~1-2 settimane per implementazione + debug + training


### **OPZIONE 3: Hybrid Approach (COMPROMESSO)** 🔄
Adattare solo il data loading, mantenere architettura

**Cosa fare:**
1. Modificare solo collate_fn per "unroll" scene in track
2. Nel DataLoader: spacchetta (B, T, N, 4) → (B*N, 5, 8)
3. Post-processing: ricomponi output in formato scena

**PRO:**
✅ Modifiche minimali al codice
✅ Usa modello originale
✅ Veloce da implementare

**CONTRO:**
❌ Batch size limitato da memoria (B*N può essere grande)
❌ Training meno efficiente
❌ Collate_fn complessa

**STIMA TEMPO:** ~3-4 ore


## RACCOMANDAZIONE

🎯 **Scegliamo OPZIONE 1** perché:
1. TrackSSM è progettato per single-track prediction
2. Abbiamo pesi pre-trained da MOT17 da sfruttare
3. È l'approccio standard per motion forecasting
4. Più robusto e testato
5. Implementazione in 2-3 ore vs 1-2 settimane

## PROSSIMI STEP (Opzione 1)

1. **Creare nuovo dataset loader: `NuScenesTrackDataset`**
   - Legge gt.txt MOT format
   - Estrae tutte le track individuali
   - Sliding window di 6 frame per track
   - Output: (condition: 5x8, cur_bbox: 4, cur_gt: 7)

2. **Test del nuovo dataset:**
   - Verificare shape corrette
   - Check bbox normalization
   - Validare delta_bbox computation

3. **Aggiornare training script:**
   - Usare NuScenesTrackDataset invece di NuScenesInterpolatedDataset
   - Verificare collate_fn compatibile

4. **Lanciare training:**
   - Test su pochi batch
   - Full training se tutto ok

## TIMELINE STANOTTE

- Ora: 19:30
- Implementazione: 19:30 - 21:30 (2h)
- Testing: 21:30 - 22:00 (30min)
- Training launch: 22:00
- Domani mattina: training in corso

✅ **FATTIBILE PER STANOTTE!**

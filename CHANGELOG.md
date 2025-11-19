# Changelog - Project Reorganization

## [2025-11-18] Major Project Restructuring

### 🎯 Obiettivo
Riorganizzazione completa del progetto per una struttura più logica, pulita e manutenibile prima del lancio degli esperimenti di fine-tuning.

### ✅ Modifiche Implementate

#### 1. **Riorganizzazione Scripts** (`scripts/`)
Tutti gli script sono stati organizzati in sottodirectory tematiche:

- **`scripts/data_preparation/`** - Preparazione e preprocessing dataset
  - `prepare_nuscenes_interpolated.py` (moved from root)
  - `prepare_nuscenes_all_classes.py` (moved from root)
  - `prepare_finetuning_splits.py` (moved from root)
  - `sanity_check_projection.py` (moved from scripts/)
  - `generate_splits.sh` (moved from scripts/)

- **`scripts/training/`** - Training e fine-tuning
  - `train_phase1_decoder.py` (moved from root)
  - `train_phase2_full.py` (moved from root)
  - `run_phase1_training.sh` (moved from scripts/)
  - `run_phase2_training.sh` (moved from scripts/)

- **`scripts/evaluation/`** - Calcolo metriche
  - `compute_hota_trackssm.py` (moved from root)
  - `compute_hota_manual.py` (moved from root)
  - `compute_hota_trackssm_simple.py` (moved from root)
  - `compute_per_class_metrics.py` (moved from root)
  - `recompute_metrics_correct.py` (moved from root)

- **`scripts/plotting/`** - Visualizzazioni
  - `plot_comparison_7classes.py` (moved from root)
  - `generate_comparison_plots.py` (moved from scripts/)
  - `generate_final_plots.py` (moved from scripts/)
  - `plot_comprehensive_comparison.py` (moved from scripts/)

- **`scripts/utils/`** - Utility vari
  - `show_results.sh` (moved from root)
  - `show_plots_structure.sh` (moved from root)
  - `quickstart_test.sh` (moved from scripts/)

#### 2. **Consolidamento Config Files**
- ✅ Rimossi duplicati: `nuscenes_finetuning_phase1.yaml`, `nuscenes_finetuning_phase2.yaml`
- ✅ Mantenuti solo: `nuscenes_phase1.yaml`, `nuscenes_phase2.yaml`
- ✅ Organizzazione chiara in `configs/`

#### 3. **Pulizia Root Directory**
File mantenuti nella root (essenziali):
- `README.md` - Overview generale progetto
- `README_TRAINING.md` - Guida completa fine-tuning
- `main.py` - Entry point tracking
- `diffmot.py` - Core implementation
- `LICENSE` - License
- `requirement.txt` - Dependencies
- `.gitignore` - Git ignore patterns

File rimossi:
- ❌ `README_FINETUNING.md` (merged in README_TRAINING.md)
- ❌ `FINE_TUNING_PIPELINE.md` (merged in README_TRAINING.md)

#### 4. **Aggiornamento .gitignore**
Aggiunti pattern per ignorare:
```gitignore
# Python cache ricorsivi
**/__pycache__/
*.pyc

# Dataset generati
data/nuscenes_mot_6cams_interpolated/
data/nuscenes_test_interpolation/

# Output training
weights/
!weights/.gitkeep

# Sanity check output
sanity_check_output/

# Results
results/
```

#### 5. **Consolidamento Documentazione**
- ✅ Creato `README_TRAINING.md` unificato (merge di README_FINETUNING.md + FINE_TUNING_PIPELINE.md)
- ✅ Aggiornato `README.md` con:
  - Sezione NuScenes fine-tuning extension
  - Quick start guide
  - **Documentazione completa struttura progetto** con descrizione ogni directory
- ✅ Documentazione tecnica dettagliata:
  - Interpolazione temporale 2Hz → 12fps
  - Proiezione 3D → 2D pipeline
  - Dataset loader format
  - Two-phase training rationale
  - Troubleshooting guide

#### 6. **Aggiornamento Path negli Script**
- ✅ `run_phase1_training.sh`: path aggiornato a `scripts/training/train_phase1_decoder.py`
- ✅ `run_phase2_training.sh`: path aggiornato a `scripts/training/train_phase2_full.py`
- ✅ `generate_splits.sh`: path aggiornato a `scripts/data_preparation/prepare_nuscenes_interpolated.py`

#### 7. **Struttura Weights Directory**
- ✅ Creata struttura `weights/phase1/` e `weights/phase2/`
- ✅ Aggiunti `.gitkeep` files per preservare struttura in Git

### 📊 Struttura Finale

```
trackssm_reference/
├── README.md                    # Overview + NuScenes extension
├── README_TRAINING.md          # Guida completa fine-tuning
├── CHANGELOG.md                # Questo file
├── LICENSE
├── requirement.txt
├── .gitignore
├── main.py
├── diffmot.py
│
├── scripts/
│   ├── data_preparation/       # 5 files (prep + sanity + generate)
│   ├── training/              # 4 files (train + run scripts)
│   ├── evaluation/            # 5 files (compute metrics)
│   ├── plotting/              # 4 files (visualizations)
│   └── utils/                 # 3 files (show + quickstart)
│
├── configs/                   # YAML configs (no duplicates)
├── dataset/                   # PyTorch datasets
├── models/                    # Neural architectures
├── tracker/                   # Tracking algorithms
├── tools/                     # Export/inference tools
├── external/                  # External dependencies
│
├── data/                      # Generated datasets (gitignored)
├── weights/                   # Model checkpoints (gitignored)
│   ├── phase1/.gitkeep
│   └── phase2/.gitkeep
├── results/                   # Tracking results (gitignored)
└── logs/                      # Training logs (gitignored)
```

### 🎯 Vantaggi

1. **Organizzazione Logica**: Scripts organizzati per funzionalità (data/training/evaluation/plotting)
2. **Scalabilità**: Facile aggiungere nuovi script nelle categorie appropriate
3. **Manutenibilità**: Struttura chiara riduce confusione
4. **Documentazione**: README completo con struttura progetto documentata
5. **Pulizia**: Rimossi duplicati e file obsoleti
6. **Git-friendly**: .gitignore aggiornato per ignorare output generati

### 🚀 Prossimi Passi

Con la struttura pulita, ora si può procedere con:

1. **Generare dataset completo**: `bash scripts/data_preparation/generate_splits.sh`
2. **Lanciare Phase 1 training**: `bash scripts/training/run_phase1_training.sh`
3. **Lanciare Phase 2 training**: `bash scripts/training/run_phase2_training.sh`
4. **Evaluare risultati**: `python scripts/evaluation/compute_hota_trackssm.py`

### ⚠️ Note per Developer

- Tutti i path negli script sono stati aggiornati per riflettere la nuova struttura
- Se aggiungi nuovi script, posizionali nella categoria appropriata in `scripts/`
- La documentazione è ora in `README_TRAINING.md` (non più README_FINETUNING.md)
- I checkpoint verranno salvati in `weights/phase1/` e `weights/phase2/`

---

**Data**: 2025-11-18  
**Autore**: Project Reorganization  
**Status**: ✅ Completato

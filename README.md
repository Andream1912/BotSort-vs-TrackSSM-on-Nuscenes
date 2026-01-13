# BotSort vs TrackSSM on NuScenes (MOT)

Framework di tracking multi-oggetto per confrontare TrackSSM e BoT-SORT su NuScenes, con detector YOLOX e pipeline di evaluation.

## Asset (dataset / weights)

Per motivi di **licenza** e **dimensione**, il dataset nuScenes e diversi checkpoint non sono pensati per essere redistribuiti “embedded” nel repo pubblico.

- Dataset esperimenti (front camera, formato MOT): `data/nuscenes_mot_front/` (da generare o trasferire)
- Pesi principali (da scaricare/trasferire):
  - YOLOX fine-tuned: `yolox_finetuning/yolox_l_nuscenes_stable/epoch_30.pth`
  - TrackSSM fine-tuned (Phase2): `weights/trackssm/phase2/phase2_full_best.pth`

Per la riproduzione degli esperimenti già fissati: `results/GRID_SEARCH/best_config.json`.

## Installazione

```bash
pip install -r requirements.txt
```

Dipendenze esterne (non vendorizzate): vedi `external/README.md`.

### YOLOX (obbligatorio)

Il progetto si aspetta YOLOX in `external/YOLOX`:

```bash
git clone https://github.com/Megvii-BaseDetection/YOLOX external/YOLOX
```

## Quick start: tracking + valutazione

Esegui TrackSSM sul validation split incluso e salva i risultati in formato MOT.

### 1) TrackSSM (detector fine-tuned)

```bash
python track.py \
  --tracker trackssm \
  --data data/nuscenes_mot_front/val \
  --gt-data data/nuscenes_mot_front/val \
  --detector-weights yolox_finetuning/yolox_l_nuscenes_stable/epoch_30.pth \
  --conf-thresh 0.5 \
  --nms-thresh 0.6 \
  --track-thresh 0.6 \
  --match-thresh 0.8 \
  --cmc-method sparseOptFlow \
  --trackssm-batch \
  --output results/run_trackssm 
```

### 2) Valutazione (motmetrics)

```bash
python scripts/evaluation/evaluate_motmetrics.py \
  --gt-folder data/nuscenes_mot_front/val \
  --pred-folder results/run_trackssm/data \
  --output results/run_trackssm/metrics.json
```

## Configurazioni (preset)

I preset “ufficiali” sono in `configs/experiment_presets.json`:

- `baseline_defaults`: pesi COCO (`weights/detectors/yolox_x.pth`) + default del tracker
- `grid_search_optimal`: preset per esperimenti post-tuning

Per la miglior configurazione trovata dagli esperimenti già eseguiti, vedi `results/GRID_SEARCH/best_config.json`.

## Grid search (riproduzione)

Lo script principale è `scripts/grid_search/grid_search_parallel.py` (search space: `conf_thresh`, `match_thresh`, `track_thresh`, `nms_thresh`).

```bash
./scripts/grid_search/start_grid_search.sh 30 4
./scripts/grid_search/monitor_grid_search.sh
cat results/GRID_SEARCH/best_config.json | python -m json.tool
```

## Detector / Tracker: con e senza fine-tuning

- Senza fine-tuning detector: usa `weights/detectors/yolox_x.pth` (80 classi COCO + mapping verso classi NuScenes).
- Con fine-tuning detector: usa `yolox_finetuning/yolox_l_nuscenes_stable/epoch_30.pth` (7 classi NuScenes, risoluzione 800×1440).

Per TrackSSM:

- Default (Phase2 NuScenes): `weights/trackssm/phase2/phase2_full_best.pth`
- Baseline MOT17 (Phase1): `--use-mot17-checkpoint` (usa `weights/trackssm/pretrained/MOT17_epoch160.pt`)

## Performance (tesi): cosa abbiamo ottimizzato

- **Batched TrackSSM motion prediction** (default ON): evita una forward GPU per ogni track; usa un’unica forward batched per frame.
  - Flag: `--trackssm-batch` / `--trackssm-no-batch`.
- **Benchmark integrato** in `track.py`: `--benchmark` salva `benchmark.json` con breakdown (`detect_ms`, `track_ms`, `total_ms`) e FPS.
- **CMC trade-off**:
  - `--cmc-method none` massimizza FPS.
  - `--cmc-method sparseOptFlow` migliora significativamente le metriche (HOTA/IDF1/MOTA) ma costa tempo nel tracker.

## Jetson

Istruzioni step-by-step per Jetson: vedi [README_JETSON.md](README_JETSON.md).

## Dataset: generazione (opzionale)

Se vuoi rigenerare i dataset da una installazione locale di NuScenes:

- Esperimenti (MOT front): `tools/export_nuscenes_mot_front.py`
- Training TrackSSM (6 camere interpolato): `scripts/data_preparation/prepare_nuscenes_interpolated.py` (+ `scripts/data_preparation/generate_splits.sh`)
- Training YOLOX (COCO): `yolox_finetuning/scripts/generate_dataset.py`

## Pesi

File principali usati dal progetto:

- YOLOX COCO: `weights/detectors/yolox_x.pth` (opzionale: `weights/detectors/yolox_l.pth`)
- YOLOX fine-tuned: `yolox_finetuning/yolox_l_nuscenes_stable/epoch_30.pth`
- TrackSSM: `weights/trackssm/phase2/phase2_full_best.pth` (+ Phase1 / pretrained)
- ReID (BoT-SORT): `weights/reid/mot17_sbs_S50.pth`
| **HOTA** | Overall tracking quality | Higher |
| **MOTA** | Detection + ID accuracy | Higher |
| **IDF1** | Identity preservation | Higher |
| **IDSW** | Identity switches | Lower |
| **FP/FN** | False positives/negatives | Lower |
| **Precision/Recall** | Detection quality | Higher |

---

## 🎓 **Use Cases**

1. **Tracker Comparison**: Evaluate TrackSSM vs BotSort on same dataset
2. **Detector Impact**: Compare GT oracle vs YOLOX performance
3. **Ablation Studies**: Test Phase1 (MOT17) vs Phase2 (fine-tuned)
4. **Parameter Tuning**: Experiment with track_thresh, match_thresh
5. **Visualization**: Generate videos for qualitative analysis

---

## 📚 **Documentation**

- **TRACKING_GUIDE.md**: Complete usage guide with examples
- **Command Help**: `python track.py --help`
- **Evaluation Help**: `python evaluate.py --help`

---

## 🔬 **Technical Details**

**TrackSSM Architecture**:
- Mamba-based state space model
- Temporal sequence modeling (history_len=5)
- Phase1: MOT17 pretrained decoder
- Phase2: Full model fine-tuned on NuScenes

**BotSort Pipeline**:
- ByteTrack association
- Camera motion compensation (GMC)
- ReID appearance features (optional)

**YOLOX Detector**:
- YOLOX-X model (COCO pretrained)
- 1600x900 input resolution
- conf_thresh=0.5, nms_thresh=0.7

---

## 🤝 **Contributing**

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Submit pull request

---

## 📝 **Citation**

If you use this code, please cite:

```bibtex
@misc{marino2025mottracking,
  author = {Andrea Marino},
  title = {Multi-Object Tracking System: TrackSSM \& BotSort on NuScenes},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Andream1912/BotSort-vs-TrackSSM-on-Nuscenes}
}
```

---

## 🔗 **References**

- **TrackSSM**: State Space Models for Multi-Object Tracking
- **BotSort**: [BoT-SORT Paper](https://arxiv.org/abs/2206.14651)
- **NuScenes**: [Official Dataset](https://www.nuscenes.org/)
- **YOLOX**: [YOLOX Paper](https://arxiv.org/abs/2107.08430)
- **TrackEval**: [Official Evaluation Toolkit](https://github.com/JonathonLuiten/TrackEval)

---

## 👤 **Author**

**Andrea Marino**  
📧 GitHub: [@Andream1912](https://github.com/Andream1912)  
📚 Project: Master's Thesis - Multi-Object Tracking on Autonomous Driving Datasets

---

## ⭐ **Acknowledgments**

- NuScenes team for the autonomous driving dataset
- TrackSSM authors for the state space model architecture
- BotSort authors for the motion-based tracking approach
- TrackEval maintainers for the evaluation metrics



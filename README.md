# Multi-Object Tracking System: TrackSSM & BotSort on NuScenes

> **Unified MOT framework comparing state-of-the-art trackers (TrackSSM, BotSort) on the NuScenes autonomous driving dataset with fine-tuned YOLOX detector, hyperparameter optimization, and comprehensive evaluation.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 **Project Overview**

Complete **multi-object tracking (MOT) research framework** with:

1. **State-of-the-Art Trackers**:
   - **TrackSSM**: Mamba SSM-based motion predictor (replaces Kalman Filter)
   - **BotSort**: Motion + ReID tracker with camera motion compensation

2. **Fine-Tuned Detection**:
   - **YOLOX-L** fine-tuned on NuScenes (30 epochs, 7 classes)
   - Training loss: 10.70 → 4.55 (-37.4% reduction)
   - Stable training with warmup + no-augmentation phases

3. **Hyperparameter Optimization**:
   - Parallel grid search (4 workers)
   - Real-time best config tracking
   - 320 configurations explored (conf_thresh, match_thresh, track_thresh, nms_thresh)

4. **Comprehensive Evaluation**:
   - HOTA, MOTA, IDF1, IDSW metrics
   - Per-class and aggregate analysis
   - Automated evaluation pipeline

---

## 📊 **Latest Results**

**TrackSSM + YOLOX Fine-tuned (Epoch 30)**:
- **MOTA**: 36.02% | **IDF1**: 51.49% | **HOTA**: 52.78%
- **IDSW**: 3042 | **Recall**: 54.42% | **Precision**: 85.08%
- 151 validation scenes, 7 classes

**YOLOX-L Detector Performance**:
- Fine-tuned on NuScenes for 30 epochs
- Best checkpoint: Epoch 25 (loss 4.52)
- Stable convergence (CV=0.6% in no-aug phase)

---

## 🚀 **Quick Start**

### Run Tracking with Fine-Tuned Detector

```bash
# TrackSSM with YOLOX fine-tuned detector
python track.py \
    --tracker trackssm \
    --detector-weights yolox_finetuning/yolox_l_nuscenes_stable/epoch_30.pth \
    --conf-thresh 0.3 \
    --match-thresh 0.85 \
    --output results/my_experiment \
    --gt-data data/nuscenes_mot_front/val

# Evaluate results
python scripts/evaluation/evaluate_motmetrics.py \
    --pred-folder results/my_experiment/data \
    --output results/my_experiment/metrics.json
```

### Run Parallel Grid Search

```bash
# Start 4 parallel workers for hyperparameter optimization
./scripts/grid_search/start_grid_search.sh 30 4

# Monitor progress in real-time
./scripts/grid_search/monitor_grid_search.sh

# View best configuration anytime
cat results/GRID_SEARCH_PARALLEL/best_config.json | python -m json.tool
```

---

## 🏗️ **Project Structure**

```
tesi_project_amarino/
├── track.py                    # Main tracking script
├── README.md                   # This file
├── requirements.txt            # Python dependencies
│
├── scripts/                    # Organized scripts
│   ├── testing/               # Test scripts for different configs
│   ├── evaluation/            # Evaluation and analysis tools
│   ├── grid_search/           # Parallel hyperparameter search
│   ├── training/              # TrackSSM training scripts
│   └── plotting/              # Visualization scripts
│
├── docs/                       # Documentation
│   └── TRACKSSM_HISTORY_MANAGEMENT.md
│
├── logs_archive/               # Historical logs
│
├── yolox_finetuning/          # Detector fine-tuning
│   ├── training_stable.log    # Training log
│   ├── training_curve.png     # Training visualization
│   └── yolox_l_nuscenes_stable/
│       ├── epoch_1.pth        # Checkpoints
│       ├── epoch_10.pth
│       ├── epoch_25.pth       # Best loss (4.52)
│       └── epoch_30.pth       # Most refined
│
├── src/                        # Source code
│   ├── trackers/              # Tracker implementations
│   ├── detectors/             # Detector wrappers
│   └── utils/                 # Utilities
│
├── weights/                    # Model weights
│   ├── detectors/             # YOLOX pre-trained
│   ├── trackssm/              # TrackSSM checkpoints
│   └── reid/                  # ReID models
│
├── data/                       # Datasets
│   └── nuscenes_mot_front/    # NuScenes MOT format
│
└── results/                    # Experiment outputs
    ├── TRACKSSM_STABLE_EPOCH30/
    ├── GRID_SEARCH_PARALLEL/
    └── ...
```
tesi_project_amarino/
├── track.py                # Main tracking script
├── evaluate.py             # Standalone evaluation
├── requirements.txt        # Dependencies
├── TRACKING_GUIDE.md      # Complete documentation
│
├── src/                    # Core modules
│   ├── detectors/         # GT & YOLOX detectors
│   └── trackers/          # TrackSSM & BotSort implementations
│
├── weights/                # Model checkpoints
│   ├── yolox/             # YOLOX-X (757MB)
│   └── trackssm/
│       ├── phase1/        # MOT17 pretrained (33MB)
│       └── phase2/        # NuScenes fine-tuned (39MB)
│
├── data/                   # NuScenes MOT dataset
│   └── nuscenes_mot_front/
│       ├── train/
│       └── val/           # 150 scenes
│
└── results/                # Tracking outputs
```

---

## 📖 **Core Features**

### 1. **Dual Detection Modes**

| Mode | Use Case | Command |
|------|----------|---------|
| **GT Oracle** | Upper-bound analysis | `--use-gt-det` |
| **YOLOX** | Real-world performance | (default) |

### 2. **Automatic Evaluation**

```bash
# One command: track + evaluate
python track.py --tracker trackssm --data DATA --output OUT --evaluate

# Output: metrics_summary.json with HOTA, MOTA, IDF1, IDSW, FP, FN, Precision, Recall
```

### 3. **Video Visualization**

```bash
python track.py --tracker trackssm --use-gt-det \
    --data DATA --output OUT --scenes SCENE_LIST \
    --save-videos --video-fps 12
```

Generates MP4 videos with:
- Green bounding boxes
- Track ID labels
- Configurable FPS

### 4. **Multi-Checkpoint Support (TrackSSM)**

| Checkpoint | Description | Use Case |
|------------|-------------|----------|
| **Phase2** (default) | NuScenes fine-tuned | Production use |
| **Phase1** | MOT17 pretrained | Baseline/ablation |

```bash
# Use MOT17 pretrained checkpoint
python track.py --tracker trackssm --use-mot17-checkpoint \
    --data DATA --output OUT --evaluate
```

---

## 📊 **Metrics Output**

### metrics_summary.json (Primary)
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
  "Total_GT_IDs": 6,
  "Total_Predicted_IDs": 5,
  "experiment_config": {
    "tracker": "trackssm",
    "detector": "GT (Oracle)",
    "checkpoint_type": "NuScenes fine-tuned (Phase2)",
    ...
  }
}
```

### experiment_config.json
Logs all parameters:
- Tracker type & checkpoint
- Detector type
- Thresholds (track_thresh, match_thresh)
- Processing info (scenes, video export, etc.)

---

## 🔧 **Key Parameters**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--tracker` | trackssm \| botsort | trackssm |
| `--data` | Dataset root path | required |
| `--output` | Output directory | required |
| `--use-gt-det` | GT oracle mode | False |
| `--evaluate` | Auto-evaluate | False |
| `--save-videos` | Export videos | False |
| `--scenes` | Scene list (comma-separated) | all (150) |
| `--track-thresh` | Track confidence | 0.6 |
| `--match-thresh` | Matching threshold | 0.8 |
| `--use-mot17-checkpoint` | Use Phase1 checkpoint | False |
| `--video-fps` | Video framerate | 12 |

**See** `TRACKING_GUIDE.md` **for complete documentation.**

---

## 📈 **Understanding Metrics**

| Metric | What It Measures | Good Value |
|--------|------------------|------------|
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



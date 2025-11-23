# Multi-Object Tracking System: TrackSSM & BotSort on NuScenes

> **Unified MOT framework comparing state-of-the-art trackers (TrackSSM, BotSort) on the NuScenes autonomous driving dataset with comprehensive evaluation and flexible detection modes.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 **What This Project Does**

This is a **complete multi-object tracking (MOT) evaluation framework** designed to:

1. **Compare State-of-the-Art Trackers**:
   - **TrackSSM**: State Space Model-based tracker with Mamba architecture
   - **BotSort**: Motion + appearance-based tracker with camera motion compensation

2. **Flexible Detection Modes**:
   - **GT Oracle**: Uses ground truth detections for upper-bound performance analysis
   - **YOLOX Detector**: Real-world COCO-pretrained detector for practical scenarios

3. **Comprehensive Evaluation**:
   - HOTA, CLEAR (MOTA, MOTP, IDSW), Identity (IDF1) metrics
   - Per-class evaluation for multi-class datasets
   - Automatic metric computation with TrackEval integration

4. **Production-Ready Pipeline**:
   - Single command for tracking + evaluation
   - Dual output formats (MOT standard + multi-class)
   - Video export with bounding box visualization
   - Experiment tracking with full config logging

---

## 📊 **Key Results**

**TrackSSM on NuScenes validation set (GT detector, 150 scenes)**:
- HOTA: 57.26% | MOTA: 59.74% | IDF1: 59.20% | IDSW: 2

**System supports**:
- 150 NuScenes validation scenes
- 7 object classes (pedestrian, car, truck, bus, motorcycle, bicycle, trailer)
- 12 FPS video export
- Batch processing of all scenes

---

## 🚀 **Quick Start**

### Installation

```bash
git clone https://github.com/Andream1912/BotSort-vs-TrackSSM-on-Nuscenes.git
cd BotSort-vs-TrackSSM-on-Nuscenes

# Create environment
conda create -n mot python=3.10
conda activate mot
pip install -r requirements.txt
```

### Run Tracking + Evaluation

```bash
# TrackSSM with GT detector on all scenes
python track.py \
    --tracker trackssm \
    --use-gt-det \
    --data data/nuscenes_mot_front/val \
    --output results/trackssm_oracle \
    --evaluate

# With video export on specific scenes
python track.py \
    --tracker trackssm \
    --use-gt-det \
    --data data/nuscenes_mot_front/val \
    --output results/demo \
    --scenes scene-0003_CAM_FRONT,scene-0012_CAM_FRONT \
    --save-videos \
    --evaluate
```

**Output**:
```
results/trackssm_oracle/
├── metrics_summary.json    # ⭐ Main metrics file
├── experiment_config.json  # Full experiment log
├── data/                   # MOT format results
├── with_classes/           # Multi-class results
└── videos/                 # Visualization (if --save-videos)
```

---

## 🏗️ **Project Structure**

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





# TrackSSM
####  TrackSSM is a general motion predictor with the state space model.

> [**TrackSSM: A General Motion Predictor by State-Space Model**](https://arxiv.org/abs/2409.00487)
> 
> Bin Hu, Run Luo, Zelin Liu, Cheng Wang, Wenyu Liu
> 
> *[arXiv 2409.00487](https://arxiv.org/abs/2409.00487)*

---

## 🚀 NuScenes Fine-tuning Extension

This repository has been extended with a complete pipeline for fine-tuning TrackSSM on the **NuScenes dataset** for autonomous driving applications.

### Key Features
- ✅ **Multi-camera support**: 6 cameras (CAM_FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_LEFT, BACK_RIGHT)
- ✅ **Multi-class tracking**: 7 vehicle classes (car, truck, bus, trailer, pedestrian, motorcycle, bicycle)
- ✅ **Temporal interpolation**: 2Hz → 12fps for stable training
- ✅ **Two-phase fine-tuning**: Decoder-only → Full model with differential LR

**📚 Complete Guide**: See [README_TRAINING.md](README_TRAINING.md) for the full fine-tuning pipeline.

**Quick Start**:
```bash
# 1. Generate interpolated dataset
export NUSC_ROOT=/mnt/datasets/Nuscense
bash scripts/data_preparation/generate_splits.sh

# 2. Train Phase 1 (decoder-only)
bash scripts/training/run_phase1_training.sh

# 3. Train Phase 2 (full fine-tuning)
bash scripts/training/run_phase2_training.sh
```

---


## News
- Submitting the paper on Arxiv at Sep 4 2024.
 
## Tracking performance
### Results on MOT17, DanceTrack, SportsMOT test set
| Dataset    | HOTA | MOTA | IDF1 | AssA | DetA | 
|------------|-------|-------|------|------|-------|
|MOT17       | 61.4 | 78.5 | 74.1 | 59.6 | 63.6 |
|DanceTrack  | 57.7 | 92.2 | 57.5 | 41.0 | 81.5 |
|SportsMOT   | 74.4 | 96.8 | 74.5 | 62.4 | 88.8 |

 
## Installation
> Creating a new environment.
> 
> Running: pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
> 
> Compiling [mamba 1.2.0 post1](https://github.com/state-spaces/mamba/tree/v1.2.0.post1), ensuring triton == 2.1.0
> 
> Running: pip install -r requirement.txt
>
> cd external/YOLOX,run: pip install -r requirements.txt && python setup.py develop


## Data preparation
Download [MOT17](https://motchallenge.net/), [MOT20](https://motchallenge.net/), [DanceTrack](https://github.com/DanceTrack/DanceTrack), [SportsMOT](https://github.com/MCG-NJU/SportsMOT) and put them under ROOT/ in the following structure. The structure of the MIX dataset follows the method used in [DiffMOT](https://github.com/Kroery/DiffMOT):
```
ROOT
   |
   |——————TrackSSM(repo)
   |                         
   |——————mot(MIX)
   |        └——————train(MOT17 train set and MOT20 train set)
   |        └——————test(MOT17 test set and MOT20 test set)
   |——————DanceTrack
   |           └——————train
   |           └——————train_seqmap.txt
   |           └——————test
   |           └——————test_seqmap.txt
   |           └——————val
   |           └——————val_seqmap.txt
   └——————SportsMOT
              └——————train
              └——————test
              └——————val
              └——————splits_txt
                         └——————train.txt
                         └——————val.txt
                         └——————test.txt
```
and then, run
```
python dancetrack_data_process.py
python sports_data_process.py
python mot_data_process.py
```

## Model zoo
### Detection Model
Refer to [Detection Model](https://github.com/Kroery/DiffMOT/releases/tag/v1.0).

### Motion Model
Refer to :
[MOT17-61.4 HOTA](https://drive.google.com/file/d/1KuTmi4t9qwcm2dXCW6xPY2dVhWSs6jK8/view?usp=drive_link),
[DanceTrack-57.7 HOTA](https://drive.google.com/file/d/1VvOjZNG3QPI4TPWl13ibUzuVvFyxCTa9/view?usp=drive_link),
[SportsMOT-74.4 HOTA](https://drive.google.com/file/d/1Uu6S-kYZoTZAq1RbwlZtyBH5Y42W7vB2/view?usp=drive_link).






## Training
### Training Detection Model
Refer to [ByteTrack](https://github.com/ifzhang/ByteTrack).

### Training Motion Model
- Changing the data_dir in config
- Training on the MIX, DanceTrack and SportsMOT:
```
python main.py --config ./configs/dancetrack.yaml
python main.py --config ./configs/sportsmot.yaml
python main.py --config ./configs/mot.yaml
```
**Notes**:
  - For MIX, we should unenable line 60 in [motion_decoder.py](https://github.com/Xavier-Lin/TrackSSM/blob/main/models/motion_decoder.py).
  - For MIX and DanceTrack, we should unenable GIoU loss in [motion_decoder.py](https://github.com/Xavier-Lin/TrackSSM/blob/main/models/motion_decoder.py).
  - For SportsMOT,  we should use both GIoU loss and smooth L1 loss in [motion_decoder.py](https://github.com/Xavier-Lin/TrackSSM/blob/main/models/motion_decoder.py).


## Tracking
#### Tracking on MOT17 test set
```
python main.py --config ./configs/mot17_test.yaml
```

#### Tracking on DanceTrack test set
```
python main.py --config ./configs/dancetrack_test.yaml
```

#### Tracking on SportsMOT test set
```
python main.py --config ./configs/sportsmot_test.yaml
```
**Notes**:
  - For tracking on MOT17, we should unenable line 60 in [motion_decoder.py](https://github.com/Xavier-Lin/TrackSSM/blob/main/models/motion_decoder.py).
  - Before perform tracking process, change det_dir, info_dir and save_dir in config files.
  - The ***use_detection_model*** is an optional item. When making the ***use_detection_model*** project effective, the detector will participate in the process of tracking inference, not just the motion model.
  - The ***interval*** the length of the historical trajectory involved in training and inference.

 
## Citation
```bibtex
@misc{trackssm,
      title={TrackSSM: A General Motion Predictor by State-Space Model}, 
      author={Bin Hu and Run Luo and Zelin Liu and Cheng Wang and Wenyu Liu},
      year={2024},
      eprint={2409.00487},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.00487}, 
}
```

## Acknowledgements
A large part of the code is borrowed from [Mamba](https://github.com/state-spaces/mamba),[DiffMOT](https://github.com/Kroery/DiffMOT), [FairMOT](https://github.com/ifzhang/FairMOT), [ByteTrack](https://github.com/ifzhang/ByteTrack). 
 Many thanks for their wonderful works.

---

## 📁 Project Structure

```
trackssm_reference/
├── README.md                          # This file - project overview
├── README_TRAINING.md                 # Complete NuScenes fine-tuning guide
├── LICENSE                            # License information
├── requirement.txt                    # Python dependencies
├── .gitignore                         # Git ignore patterns
│
├── main.py                            # Main entry point for tracking
├── diffmot.py                         # Core DiffMOT implementation
│
├── configs/                           # Configuration files
│   ├── nuscenes_phase1.yaml          # Phase 1 training hyperparameters
│   ├── nuscenes_phase2.yaml          # Phase 2 training hyperparameters
│   ├── nuscenes_trackssm_7classes.yaml  # Inference config for NuScenes
│   ├── mot.yaml                      # MOT17/MOT20 training
│   ├── mot17_test.yaml               # MOT17 test
│   ├── dancetrack.yaml               # DanceTrack training
│   └── sportsmot.yaml                # SportsMOT training
│
├── scripts/                           # All executable scripts
│   ├── data_preparation/             # Dataset generation and preprocessing
│   │   ├── prepare_nuscenes_interpolated.py   # Generate interpolated MOT format
│   │   ├── sanity_check_projection.py         # Verify 3D→2D projection
│   │   └── generate_splits.sh                 # Generate train/val/test splits
│   │
│   ├── training/                     # Training scripts
│   │   ├── train_phase1_decoder.py   # Phase 1: Decoder-only fine-tuning
│   │   ├── train_phase2_full.py      # Phase 2: Full fine-tuning
│   │   ├── run_phase1_training.sh    # Launch Phase 1
│   │   └── run_phase2_training.sh    # Launch Phase 2
│   │
│   ├── evaluation/                   # Metrics computation
│   │   ├── compute_hota_trackssm.py  # HOTA metrics for TrackSSM
│   │   ├── compute_per_class_metrics.py  # Per-class breakdown
│   │   └── recompute_metrics_correct.py  # Recompute with fixes
│   │
│   ├── plotting/                     # Visualization scripts
│   │   ├── plot_comparison_7classes.py       # 7-class comparison plots
│   │   ├── generate_comparison_plots.py      # Generate comparison figures
│   │   └── generate_final_plots.py           # Final publication plots
│   │
│   └── utils/                        # Utility scripts
│       ├── show_results.sh           # Display results summary
│       └── quickstart_test.sh        # Quick test script
│
├── dataset/                           # PyTorch Dataset implementations
│   └── nuscenes_interpolated_dataset.py  # NuScenes dataset loader
│
├── models/                            # Neural network architectures
│   ├── mamba_encoder.py              # Mamba encoder (SSM backbone)
│   ├── motion_decoder.py             # Decoder for motion prediction
│   ├── condition_embedding.py        # Condition embedding module
│   ├── diffusion.py                  # Diffusion process
│   └── ...
│
├── tracker/                           # Tracking algorithms
│   ├── DiffMOTtracker.py             # DiffMOT tracker implementation
│   ├── BYTETracker.py                # ByteTrack baseline
│   └── ...
│
├── tracking_utils/                    # Tracking utilities
│   ├── kalman_filter.py              # Kalman filter
│   ├── matching.py                   # Data association
│   └── ...
│
├── tools/                             # Export and inference tools
│   ├── export_nuscenes_mot_front.py  # Export NuScenes to MOT format
│   └── infer_yolox_perframe.py       # YOLOX inference
│
├── external/                          # External dependencies
│   ├── YOLOX/                        # YOLOX detector
│   └── TrackEval/                    # Evaluation framework
│
├── data/                              # Generated datasets (gitignored)
│   ├── nuscenes_mot_6cams_interpolated/  # Interpolated dataset
│   ├── nuscenes_mot_front_7classes/      # Front camera only
│   └── ...
│
├── weights/                           # Model checkpoints (gitignored)
│   ├── phase1/                       # Phase 1 checkpoints
│   ├── phase2/                       # Phase 2 checkpoints
│   └── ...
│
├── results/                           # Tracking results (gitignored)
│   ├── nuscenes_trackssm_finetuned/  # Fine-tuned model results
│   └── ...
│
└── logs/                              # Training logs (gitignored)
```

### Directory Descriptions

- **`scripts/`**: All executable scripts organized by purpose
  - `data_preparation/`: Dataset generation, interpolation, sanity checks
  - `training/`: Training scripts for Phase 1 and Phase 2 fine-tuning
  - `evaluation/`: Metrics computation (HOTA, IDF1, MOTA, etc.)
  - `plotting/`: Visualization and plotting scripts
  - `utils/`: Miscellaneous utility scripts

- **`configs/`**: YAML configuration files for training and inference

- **`dataset/`**: PyTorch Dataset classes for loading data

- **`models/`**: Neural network architectures (encoder, decoder, diffusion)

- **`tracker/`**: Tracking algorithm implementations

- **`data/`**, **`weights/`**, **`results/`**: Generated outputs (gitignored)

---



# Data Directory

This directory should contain the nuScenes MOT dataset prepared for tracking experiments.

## Required Datasets

The following datasets must be prepared locally using the preparation scripts in `scripts/data_preparation/`:

### 1. NuScenes MOT Front Camera (Validation)
- **Path**: `data/nuscenes_mot_front/val/`
- **Format**: MOT Challenge format
- **Content**: Front camera validation sequences
- **Preparation**: Run `scripts/data_preparation/prepare_nuscenes_all_classes.py`

### 2. NuScenes MOT 7 Classes
- **Path**: `data/nuscenes_mot_front_7classes/`
- **Format**: MOT with 7 object classes
- **Classes**: car, truck, bus, trailer, pedestrian, motorcycle, bicycle
- **Preparation**: Run `scripts/data_preparation/prepare_nuscenes_all_classes.py --classes 7`

### 3. NuScenes MOT 6 Cameras (Interpolated)
- **Path**: `data/nuscenes_mot_6cams_interpolated/`
- **Format**: MOT with temporal interpolation (12 FPS)
- **Content**: All 6 camera views with interpolated annotations
- **Usage**: For fine-tuning detection models
- **Preparation**: Run `scripts/data_preparation/prepare_nuscenes_interpolated.py`

## Preparation Instructions

1. Download nuScenes dataset from [nuScenes.org](https://www.nuscenes.org/nuscenes#download)
2. Extract to `/mnt/datasets/Nuscense/` (or update paths in scripts)
3. Run preparation scripts:

```bash
# Prepare front camera MOT format
python scripts/data_preparation/prepare_nuscenes_all_classes.py \
    --nusc_root /mnt/datasets/Nuscense \
    --version v1.0-trainval \
    --output_dir data/nuscenes_mot_front \
    --cameras CAM_FRONT

# Prepare 7-class annotations
python scripts/data_preparation/prepare_nuscenes_all_classes.py \
    --nusc_root /mnt/datasets/Nuscense \
    --version v1.0-trainval \
    --output_dir data/nuscenes_mot_front_7classes \
    --cameras CAM_FRONT \
    --split val

# Prepare 6-camera interpolated dataset
python scripts/data_preparation/prepare_nuscenes_interpolated.py \
    --dataroot /mnt/datasets/Nuscense \
    --version v1.0-trainval \
    --output_dir data/nuscenes_mot_6cams_interpolated \
    --split train
```

## Expected Structure

```
data/
├── nuscenes_mot_front/
│   └── val/
│       └── scene-XXXX_CAM_FRONT/
│           ├── det/
│           ├── gt/
│           ├── img1/
│           └── seqinfo.ini
├── nuscenes_mot_front_7classes/
│   ├── train/
│   └── val/
│       └── scene-XXXX/
│           ├── gt/
│           │   └── gt.txt
│           └── seqinfo.ini
└── nuscenes_mot_6cams_interpolated/
    ├── train/
    └── val/
        └── scene-XXXX-CAM_YYY/
            ├── gt/
            │   └── gt.txt
            └── seqinfo.ini
```

## Notes

- All datasets are in **MOT Challenge format**
- GT files use format: `frame_id,track_id,x,y,w,h,conf,x,y,class_id`
- Class IDs: 1=car, 2=truck, 3=bus, 4=trailer, 5=pedestrian, 6=motorcycle, 7=bicycle
- Data files are **not included** in git repository due to size

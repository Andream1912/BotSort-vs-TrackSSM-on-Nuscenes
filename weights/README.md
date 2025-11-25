# Model Weights Directory

This directory contains pre-trained model weights for detection and ReID.

## Required Models

### Detection Models

#### YOLOX-X (Primary Detector)
- **File**: `detectors/yolox_x.pth`
- **Download**: [YOLOX GitHub](https://github.com/Megvii-BaseDetection/YOLOX/releases)
- **Size**: ~378 MB
- **Direct link**: `https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth`

#### YOLOX-L (Lighter Alternative)
- **File**: `detectors/yolox_l.pth`
- **Download**: [YOLOX GitHub](https://github.com/Megvii-BaseDetection/YOLOX/releases)
- **Size**: ~207 MB
- **Direct link**: `https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth`

### ReID Models

#### FastReID ResNet50-IBN
- **File**: `reid/fast_reid_resnet50_ibn.pth`
- **Source**: FastReID model zoo
- **Usage**: Appearance feature extraction for tracking

## Download Script

```bash
#!/bin/bash
# Download YOLOX weights

mkdir -p weights/detectors
cd weights/detectors

# YOLOX-X
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth

# YOLOX-L
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth

cd ../..
echo "✓ Weights downloaded"
```

## Directory Structure

```
weights/
├── detectors/
│   ├── yolox_x.pth
│   ├── yolox_l.pth
│   └── yolov8x.pt (optional)
└── reid/
    └── fast_reid_resnet50_ibn.pth
```

## Notes

- Weight files are **not included** in git repository (too large)
- Models are COCO-pretrained
- For better performance on nuScenes, consider fine-tuning on autonomous driving data

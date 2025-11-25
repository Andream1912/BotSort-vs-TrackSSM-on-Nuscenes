# External Dependencies

This directory contains external repositories required for the project.

## Required Repositories

### 1. YOLOX (Detection)
```bash
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip install -v -e .
```

### 2. FastReID (ReID Features)
```bash
git clone https://github.com/JDAI-CV/fast-reid.git
cd fast-reid
pip install -r requirements.txt
```

### 3. TrackEval (Optional - Evaluation)
```bash
git clone https://github.com/JonathonLuiten/TrackEval.git
```

## Notes

- External dependencies are **not included** in git repository
- Clone and install them separately following instructions above
- Make sure to install in the correct Python environment

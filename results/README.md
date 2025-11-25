# Results Directory

This directory contains tracking results and evaluation metrics.

## Generated During Experiments

When you run tracking experiments using `track.py`, results are saved here automatically.

## Structure

```
results/
├── trackssm/
│   └── experiment_name/
│       ├── data/
│       │   └── *.txt (MOT format predictions)
│       ├── videos/
│       │   └── *.mp4 (visualization videos)
│       ├── metrics_corrected.json
│       └── experiment_config.json
└── botsort/
    └── experiment_name/
        └── ...
```

## Metrics File Format

`metrics_corrected.json` contains:
- Overall MOTA, IDF1, Precision, Recall
- Per-class metrics for 7 object classes
- ID switches, false positives, false negatives
- Frame processing times

## Notes

- Results are **not included** in git repository
- Videos can be large (150+ scenes × ~4MB each)
- Use `--save-videos` flag to generate visualization videos

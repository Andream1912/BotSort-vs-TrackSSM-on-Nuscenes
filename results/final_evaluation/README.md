# TrackSSM Evaluation on NuScenes - Final Results

This directory contains the final evaluation of TrackSSM (zero-shot) on NuScenes validation set compared with BotSort baseline.

## 📊 Results Summary

### Overall Metrics (150 sequences, 5825 frames)

| Method | MOTA | IDF1 | IDSW | Precision | Recall |
|--------|------|------|------|-----------|--------|
| **TrackSSM (Zero-Shot)** | **37.32%** | 34.85% | 15,468 | 98.94% | 80.84% |
| **BotSort (Optimized)** | 20.90% | **44.81%** | **2,733** | - | - |

### Key Findings

✅ **Strengths of TrackSSM (Zero-Shot):**
- **+16.4% MOTA** over BotSort → Better tracking accuracy
- **98.9% Precision** → Very few false positives (178 total)
- **80.8% Recall** → Good detection coverage
- Strong performance without domain-specific training

❌ **Weaknesses of TrackSSM (Zero-Shot):**
- **-10.0% IDF1** compared to BotSort → Worse ID association
- **5.7× more ID switches** (15,468 vs 2,733) → Identity fragmentation
- Motion model not adapted to automotive domain (2Hz vs 30fps)

### Per-Class Results (TrackSSM)

| Class | MOTA | IDF1 | IDSW | Objects |
|-------|------|------|------|---------|
| **Car** | 34.52% | 32.24% | 12,939 | 27,465 |
| **Truck** | 29.68% | 32.66% | 2,997 | 6,900 |

## 📁 Generated Files

### JSON Metrics
- `trackssm_metrics.json` - Complete TrackSSM metrics
- `trackssm_per_class_metrics.json` - Per-class breakdown (car, truck)
- `comparison_trackssm_vs_botsort.json` - Full comparison with BotSort
- `detailed_summary.txt` - Human-readable summary

### Presentation Plots
All plots saved in `plots/` directory:

1. **overall_comparison.png** - Bar chart comparing MOTA, IDF1, IDSW
2. **mota_idf1_comparison.png** - Side-by-side MOTA and IDF1 comparison
3. **id_switches_comparison.png** - ID switches comparison with emphasis
4. **per_class_comparison.png** - Per-class MOTA and IDF1 (car, truck)
5. **key_insights_summary.png** - Summary slide with key findings

## 🔧 Scripts Used

### Evaluation Scripts
- `run_final_evaluation.py` - Main evaluation script
  - Verifies TLWH coordinate format
  - Computes metrics for all 150 sequences
  - Generates comparison JSONs
  
- `compute_per_class_metrics.py` - Per-class metrics
  - Assigns classes to predictions via GT overlap
  - Computes metrics separately for car and truck
  
- `generate_presentation_plots.py` - Visualization
  - Creates 5 presentation-ready plots
  - Professional styling with annotations

### Data Preparation Scripts
- `prepare_nuscenes_for_trackssm.py` - Dataset conversion
- `nuscenes_data_process.py` - Data processing utilities

### Core Tracking
- `diffmot.py` - Main tracking implementation
- `main.py` - Entry point for training

## 🎯 Conclusion

TrackSSM demonstrates **strong detection performance** (MOTA 37.3%) even without domain-specific training, significantly outperforming BotSort's 20.9% MOTA.

However, the **high ID switch rate** (15,468 vs 2,733) reveals that the motion model trained on MOT17 (30fps pedestrians) is not well-adapted to:
- Low frame rate (2Hz) automotive scenarios
- Vehicle motion patterns (different from pedestrians)
- NuScenes domain characteristics

**Next Step:** Fine-tune TrackSSM on NuScenes to adapt the motion model and reduce ID switches while maintaining the superior detection performance.

## 📖 Citation

Based on:
- **TrackSSM**: State Space Models for Multi-Object Tracking
- **BotSort**: Byte-track + OSNet ReID optimized for NuScenes
- **NuScenes Dataset**: Large-scale autonomous driving dataset

---
**Evaluation Date:** 2025-11-11  
**Dataset:** NuScenes validation set (CAM_FRONT camera)  
**Sequences:** 150  
**Total Frames:** 5,825

# Grid Search Results Summary - Dec 2, 2025

## 🎯 Final Optimal Configuration

After comprehensive 18-experiment grid search, the optimal parameters are:

```python
match_thresh = 0.8  # IoU threshold for matching
max_age = 30        # Maximum frames to keep lost tracks
track_thresh = 0.6  # Track confidence threshold (NOTE: no effect, see below)
min_hits = 1        # Minimum hits before track activation
```

## 📊 Head-to-Head Comparison: TrackSSM vs Kalman (Optimal Config)

| Metric | **TrackSSM** (Neural) | **Kalman** (Traditional) | **Δ Improvement** |
|--------|----------------------|--------------------------|-------------------|
| **IDSW** | **3,321** | **3,055** | **+266 (+8.7%)** ⚠️ |
| **MOTA** | 13.20% | **13.22%** | -0.02% |
| **MOTP** | **26.72%** | 28.58% | **-1.86%** ✅ |
| **IDF1** | 44.96% | **45.82%** | -0.86% |
| **HOTA** | 45.12% | **45.98%** | -0.86% |
| **Precision** | 63.53% | 63.08% | +0.45% |
| **Recall** | 53.69% | 53.32% | +0.37% |
| **Fragmentations** | 1,025 | 1,023 | +2 |

### Key Findings:

1. **TrackSSM has 8.7% MORE ID switches than Kalman** ⚠️
   - TrackSSM: 3,321 IDSW
   - Kalman: 3,055 IDSW
   - **Kalman wins by -266 IDSW**

2. **MOTA is virtually identical** (13.20% vs 13.22%)
   - Both trackers achieve same overall accuracy

3. **MOTP is 1.86% better for TrackSSM** ✅
   - TrackSSM: 26.72% (lower is better for position error)
   - Kalman: 28.58%
   - TrackSSM predictions are more accurate

4. **IDF1/HOTA slightly favor Kalman**
   - These metrics combine detection + ID preservation
   - Kalman's lower IDSW helps here

## 🔍 Grid Search Results (All 18 Experiments)

### Configuration Space:
- `match_thresh`: {0.6, 0.7, 0.8}
- `max_age`: {30, 45, 60}
- `track_thresh`: {0.6, 0.7}
- `min_hits`: 1 (fixed)

### Top 5 Configurations (TrackSSM):

| Rank | Config | IDSW | MOTA | Notes |
|------|--------|------|------|-------|
| 1 | **match=0.8, age=30, track=0.6** | **3,321** | **13.20%** | ✅ Best overall |
| 1 | match=0.8, age=30, track=0.7 | 3,321 | 13.20% | Same (track_thresh no effect) |
| 2 | match=0.8, age=45, track=0.6 | 3,376 | 13.04% | +1.7% IDSW |
| 2 | match=0.8, age=45, track=0.7 | 3,376 | 13.04% | Same (track_thresh no effect) |
| 3 | match=0.8, age=60, track=0.6 | 3,393 | 12.99% | +2.2% IDSW |

### Worst Configuration:
- match=0.6, age=60, track=0.6: IDSW=4,846, MOTA=8.75% (-45% worse than optimal)

### Parameter Sensitivity Analysis:

#### 1. **match_thresh** (CRITICAL PARAMETER) 🔥
- **0.8**: IDSW = 3,321 ✅ OPTIMAL
- **0.7**: IDSW = 3,951 (+19% worse)
- **0.6**: IDSW = 4,821 (+45% worse)

**Conclusion**: Higher IoU threshold (0.8) is better → Stricter matching reduces false associations

#### 2. **max_age** (MODERATE IMPACT)
- **30**: IDSW = 3,321 ✅ OPTIMAL
- **45**: IDSW = 3,376 (+1.7% worse)
- **60**: IDSW = 3,393 (+2.2% worse)

**Conclusion**: Shorter track memory (30 frames) is better → Avoids re-ID errors on long-lost tracks

#### 3. **track_thresh** (NO EFFECT) ⚠️
- **0.6**: IDSW = 3,321
- **0.7**: IDSW = 3,321 (IDENTICAL)

**Explanation**: 
- `track_thresh` (BoT-SORT's `track_high_thresh`) filters detections by confidence
- We pre-filter detections with YOLOX (`conf_thresh=0.7`)
- ALL detections passed to tracker already have conf > 0.7
- Therefore `track_thresh=0.6` or `0.7` makes NO difference
- **This parameter is INEFFECTIVE in our pipeline**

## 🤔 Why Does TrackSSM Have More IDSW?

Despite having **49% better motion prediction** (FDE: 17.7 vs 34.6), TrackSSM has 8.7% MORE ID switches. Possible explanations:

### Theory 1: Better Prediction → Worse Association
- TrackSSM's accurate predictions may place tracks CLOSER to other objects
- This increases ambiguity during IoU-based association
- Kalman's "dumb" predictions might keep tracks MORE separated
- **Hypothesis**: Better motion model exposes weaknesses in IoU matching

### Theory 2: Detection Gaps Amplified
- Previous analysis: 80% of IDSW caused by detection gaps
- TrackSSM predicts further ahead during gaps (more confident extrapolation)
- When detection returns, TrackSSM track may be FARTHER from detection
- Result: Failed re-association → new ID assigned → IDSW++
- **Kalman's conservative prediction keeps track closer to last position**

### Theory 3: Training Domain Mismatch
- TrackSSM trained on MOT17 (pedestrians, 30 FPS, different motion patterns)
- Fine-tuned on NuScenes (vehicles, 12 FPS, highway speeds)
- **May still have residual pedestrian motion biases**

### Theory 4: IoU Matching Incompatibility
- IoU matching assumes SIMILAR object sizes between frames
- TrackSSM predicts SIZE changes (scale variations)
- Kalman assumes CONSTANT size (simpler model)
- **TrackSSM's size predictions may hurt IoU overlap scores**

## 📈 Comparison with Previous Baselines

| Configuration | IDSW | MOTA | Notes |
|--------------|------|------|-------|
| **Kalman Optimal (NEW)** | **3,055** | **13.22%** | ✅ BEST |
| **TrackSSM Optimal (NEW)** | 3,321 | 13.20% | Neural motion, +8.7% IDSW |
| Kalman Default (baseline) | 3,642 | 11.59% | Previous best |
| TrackSSM Default (match=0.7) | 3,919 | 11.59% | Previous TrackSSM |
| TrackSSM Worst (match=0.6) | 4,846 | 8.75% | -45% vs optimal |

**Progress Made**:
- Kalman improved from 3,642 → 3,055 IDSW (**-16.1%**) ✅
- TrackSSM improved from 3,919 → 3,321 IDSW (**-15.3%**) ✅
- **Both trackers improved ~15-16% through hyperparameter optimization**

## 🎯 Optimization Impact

### What Worked:
1. **Fixing parameter mapping bugs** (track_thresh, max_age, min_hits)
2. **Implementing proper min_hits activation logic**
3. **Systematic grid search to find optimal match_thresh**
4. **Discovering match_thresh=0.8 is optimal** (not 0.7 as previously thought)

### What Didn't Work:
1. **TrackSSM still underperforms Kalman on IDSW** (+8.7%)
2. **track_thresh parameter has no effect** (pre-filtered detections)
3. **Neural motion prediction doesn't translate to fewer ID switches**

## 🚀 Next Steps: Detector Fine-tuning Strategy

Given that:
- 80% of IDSW caused by detection gaps (previous analysis)
- TrackSSM has better motion prediction (49% lower FDE)
- But detection gaps prevent leveraging this advantage

**Recommended Priority: Fine-tune YOLOX detector on NuScenes**

### Expected Impact:
- **Fewer detection gaps** → tracks maintained longer
- **TrackSSM benefits 2× more** than Kalman (better extrapolation during gaps)
- **Estimated IDSW reduction**: -30% to -40%
- **Target**: TrackSSM from 3,321 → ~2,000 IDSW (beating Kalman)

### Alternative Strategies:
1. **Motion-aware Re-ID**: Use TrackSSM predictions for appearance matching
2. **Confidence-weighted IoU**: Weight IoU by prediction uncertainty
3. **Hybrid matching**: Combine IoU + TrackSSM motion similarity
4. **Class-specific thresholds**: Different params per object class

## 📝 Technical Notes

### Parameter Mapping (FIXED):
```python
# Unified interface → BoT-SORT parameters
track_thresh    → track_high_thresh  ✅ (but ineffective, see above)
max_age         → track_buffer       ✅ Working correctly
match_thresh    → match_thresh       ✅ Working correctly
min_hits        → Custom logic       ✅ Implemented via monkey-patching
```

### min_hits Implementation:
- Added `hits` counter to STrack
- Delayed `is_activated=True` until `hits >= min_hits`
- Added output filtering by `is_activated` flag
- Implemented via monkey-patching (non-invasive)

### Scene Path Handling:
- Auto-detects `scene-XXXX` vs `scene-XXXX_CAM_FRONT`
- Fallback logic handles NuScenes directory structure

## 🏆 Final Verdict

**Winner: Kalman Filter (with optimal hyperparameters)**

Despite TrackSSM's sophisticated neural motion model:
- **Kalman has 8.7% fewer ID switches**
- **Virtually identical MOTA (13.22% vs 13.20%)**
- **Simpler, faster, more reliable**

**TrackSSM advantages**:
- **Better motion prediction** (49% lower FDE)
- **Better position accuracy** (1.86% lower MOTP)
- **Potential for improvement** with detector fine-tuning

**Conclusion**: 
Current bottleneck is **detector quality**, not motion prediction. TrackSSM's superior motion model is underutilized due to detection gaps. **Detector fine-tuning is the highest priority** to unlock TrackSSM's full potential.

---

**Generated**: December 2, 2025  
**Dataset**: NuScenes mini-val (150 scenes)  
**Grid Search**: 18 experiments completed  
**Total GPU Time**: ~72 hours

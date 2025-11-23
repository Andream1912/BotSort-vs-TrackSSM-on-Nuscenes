# Bug Investigation Report

## Date: 2025-11-23

## Executive Summary
Investigation into duplicate track IDs and low MOTA scores in TrackSSM tracking on NuScenes dataset.

---

## BUG #1: DUPLICATE TRACK IDs ✅ FIXED

### Description
The same track ID appears multiple times in the same frame, causing TrackEval to crash.

### Root Cause
**File**: `src/trackers/trackssm_tracker.py`  
**Line**: 306 (before fix)

When a lost track is re-identified and recovered, it is **always** appended to `self.tracked_tracks` without checking if it's already present:

```python
for track_idx, det_idx in matched_lost:
    track = self.lost_tracks[track_idx]
    det = low_conf_dets[det_idx]
    track.update(det['bbox'], self.frame_id, det['confidence'])
    track.is_lost = False
    self.tracked_tracks.append(track)  # ❌ NO DUPLICATE CHECK
```

### Reproduction
**Test**: `test_duplicate_bug.py`
- Creates a track, marks it as lost, removes from tracked_tracks
- Re-identifies and appends back to tracked_tracks
- Appends again → **DUPLICATE DETECTED** ✅

### Impact
- **Experiment 1** (final_evaluation): 169 duplicates across 67 files
- **Experiment 2** (phase2_conf0.01_track0.1): 1,510 duplicates across 136 files
- TrackEval crashes on duplicate (frame_id, track_id) pairs
- Requires post-processing with `fix_duplicates.py`

### Fix Applied
**File**: `src/trackers/trackssm_tracker.py`, line ~308

```python
for track_idx, det_idx in matched_lost:
    track = self.lost_tracks[track_idx]
    det = low_conf_dets[det_idx]
    track.update(det['bbox'], self.frame_id, det['confidence'])
    track.is_lost = False
    # Fix: Prevent duplicate tracks in tracked_tracks
    if track not in self.tracked_tracks:  # ✅ ADD THIS CHECK
        self.tracked_tracks.append(track)
```

### Verification
- Test script confirms fix prevents duplicates
- Pattern matches existing check at line 276-277 for `lost_tracks`

---

## BUG #2: MOTP METRIC SEMANTIC MISMATCH ⚠️ DOCUMENTATION ISSUE

### Description
MOTP shows ~0.75 (75%) but SOTA papers report ~0.471 meters.

### Root Cause
**NOT A CODE BUG** - Different metric definitions:
- **TrackEval MOTP**: Average IoU between matched detections (0-1 scale, dimensionless)
- **NuScenes SOTA MOTP**: Average 3D euclidean distance in meters (0-∞ scale)

### Evidence
**File**: `evaluate.py`, line 197
```python
if 'MOTP' in aggregated['CLEAR']:
    summary['MOTP'] = aggregated['CLEAR']['MOTP']  # This is IoU-based
```

TrackEval uses 2D IoU for MOTP calculation, not 3D distance.

### Impact
- **Cannot directly compare** with SOTA: 0.7509 IoU ≠ 0.471 meters
- Our MOTP=0.75 means 75% average IoU overlap (GOOD)
- SOTA MOTP=0.471m means 47cm average position error (GOOD)
- Both are valid but measure different things

### Recommendation
1. **Document clearly** in results: "MOTP is IoU-based (TrackEval 2D), not distance-based (NuScenes 3D)"
2. Consider using NuScenes official evaluation for 3D metrics
3. Focus on HOTA, MOTA, IDF1, IDSW for fair comparison

---

## BUG #3: CATASTROPHICALLY LOW MOTA ⚠️ DETECTOR ISSUE

### Description
MOTA = 3.13% (experiment 1) vs 59.8% SOTA - extremely low performance.

### Root Cause
**Detector threshold too conservative**, causing massive false negatives.

### Evidence
**Experiment 1** (conf=0.1, track=0.2):
- Recall: 21% (only 21% of GT objects detected)
- Precision: 57%
- FP: 4,352 | FN: 21,692 (82% of objects missed!)
- Total Predicted: 10,125 vs GT: 27,465 (only 37% detected)

**Experiment 2** (conf=0.01, track=0.1):
- Recall: 68% (much better!)
- Precision: 27% (worse)
- MOTA: -23.82% (negative due to FP penalty)

### Analysis
MOTA formula: `MOTA = 1 - (FP + FN + IDSW) / GT`

With conf=0.1:
- MOTA = 1 - (4352 + 21692 + 560) / 27465 = 1 - 0.9687 = **3.13%**
- Main issue: 21,692 FN (79% of GT missed)

### Solution
Grid search for optimal detection threshold:
- Test conf_thresh: [0.03, 0.05, 0.07, 0.08]
- Target: Recall >60%, Precision >35%, MOTA >30%
- Sweet spot likely around conf=0.05-0.07

---

## BUG #4: IDSW TRACKING (Actually Good!) ✅

### Status
IDSW = 560 (experiment 1) vs ~407 SOTA target

### Analysis
- **Relatively good** for first iteration
- Only 560 identity switches across 150 scenes (3.7 per scene)
- TrackSSM motion model working well
- Room for improvement but not critical

---

## ADDITIONAL CHECKS ✅ ALL CLEAR

### Code Review Findings

1. **Lost Tracks Management** (line 276-277):
   ```python
   if track not in self.lost_tracks:
       self.lost_tracks.append(track)
   ```
   ✅ Already has duplicate check (consistent with our fix)

2. **Track Creation** (line 292):
   ```python
   new_track = Track(track_id=self._get_next_id(), ...)
   self.tracked_tracks.append(new_track)
   ```
   ✅ No duplicate possible (always new Track object)

3. **Matching Algorithm** (line 324-365):
   - Uses Hungarian algorithm (linear_sum_assignment)
   - IoU-based cost matrix
   - Threshold filtering
   ✅ Logic correct

4. **Confidence Handling** (line 51-64):
   - Updates on each detection match
   - Initialized to 1.0
   ✅ No issues found

5. **Output Generation** (line 317-320):
   ```python
   for track in self.tracked_tracks:
       if track.is_activated:
           output_tracks.append(track.get_state())
   ```
   ✅ Duplicates in output caused by duplicates in tracked_tracks (fixed by Bug #1)

---

## SUMMARY

| Bug | Status | Severity | Fix Status |
|-----|--------|----------|------------|
| #1: Duplicate Track IDs | ✅ FIXED | CRITICAL | Code patched |
| #2: MOTP Semantic | ⚠️ DOCUMENTED | LOW | Need doc update |
| #3: Low MOTA | ⚠️ IDENTIFIED | HIGH | Grid search needed |
| #4: IDSW | ✅ ACCEPTABLE | MEDIUM | Monitor in next run |

---

## NEXT STEPS

1. **Immediate**:
   - ✅ Test fixed tracker on single scene (verify no duplicates)
   - Run 5-10 scenes with fixed code

2. **Short Term**:
   - Grid search detector thresholds: [0.03, 0.05, 0.07, 0.08]
   - Target: conf_thresh ≈ 0.05-0.07 for optimal recall/precision

3. **Full Experiment**:
   - Run 150 scenes with optimal config
   - Expect: MOTA >30%, IDSW <600, No duplicates

4. **Documentation**:
   - Add note about MOTP metric difference in evaluate.py
   - Update README with metric definitions

---

## TEST SCRIPTS CREATED

1. **test_duplicate_bug.py**: Unit test reproducing and verifying fix
2. **fix_duplicates.py**: Post-processing to clean existing data
3. **debug_duplicates.py**: Runtime duplicate detection for live testing

---

## VERIFICATION COMMANDS

```bash
# Test the fix
python test_duplicate_bug.py

# Check for duplicates in results
awk '{print $1, $2}' results/trackssm/EXPERIMENT/data/*.txt | sort | uniq -c | awk '$1 > 1'

# Verify no duplicates after fix
python track.py --scene scene-0003 --conf-thresh 0.05 --track-thresh 0.15
# Then check output for duplicates
```

---

## METRICS REFERENCE

### TrackEval Metrics (What We Use)
- **HOTA**: Higher-Order Tracking Accuracy (0-100%, higher better)
- **MOTA**: Multiple Object Tracking Accuracy (-∞ to 100%, higher better)
- **MOTP**: Multiple Object Tracking Precision (**IoU-based**, 0-1, higher better)
- **IDF1**: ID F1 Score (0-100%, higher better)
- **IDSW**: ID Switches (count, lower better)

### NuScenes Official Metrics (SOTA Papers)
- **AMOTA**: Average MOTA across confidence thresholds
- **AMOTP**: Average 3D distance error in **meters** (lower better)
- **IDS**: Identity Switches
- **Recall**: Detection recall

**⚠️ WARNING**: MOTP is NOT comparable between TrackEval (IoU) and NuScenes (meters)!

---

End of Report

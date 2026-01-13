#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd

# Reuse the evaluator already in the repo
sys.path.append(str(Path(__file__).resolve().parents[2]))
from scripts.evaluation.evaluate_motmetrics import evaluate_dataset  # noqa: E402


def _to_metric_scale(summary: pd.DataFrame) -> pd.DataFrame:
    """Convert motmetrics summary dataframe into friendlier scales.

    motmetrics gives most rates in [0,1]; we convert to percentage.
    """
    df = summary.copy()

    percent_cols = [
        'mota', 'motp', 'idf1', 'precision', 'recall', 'idp', 'idr'
    ]
    for c in percent_cols:
        if c in df.columns:
            df[c] = df[c] * 100.0

    # Keep OVERALL row but mark it clearly
    df = df.reset_index(names='scene')
    return df


def evaluate_per_scene(gt_folder: Path, pred_folder: Path, iou: float) -> pd.DataFrame:
    summary = evaluate_dataset(str(gt_folder), str(pred_folder), scenes=None, iou_threshold=iou)
    if summary is None or summary.empty:
        raise RuntimeError(f"No evaluation results for pred_folder={pred_folder}")
    return _to_metric_scale(summary)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Density proxy: objects per frame
    if 'num_objects' in out.columns and 'num_frames' in out.columns:
        out['obj_per_frame'] = out['num_objects'] / out['num_frames'].replace(0, np.nan)

    # Basic derived counters are already in the summary
    return out


def merge_and_diff(trackssm_df: pd.DataFrame, botsort_df: pd.DataFrame) -> pd.DataFrame:
    # Drop OVERALL for per-scene comparison, keep separately if needed
    t = trackssm_df[trackssm_df['scene'] != 'OVERALL'].copy()
    b = botsort_df[botsort_df['scene'] != 'OVERALL'].copy()

    merged = t.merge(b, on='scene', how='inner', suffixes=('_trackssm', '_botsort'))

    # Define deltas: positive means TrackSSM better (except switches/frags where lower is better)
    delta_specs = {
        'mota': +1,
        'idf1': +1,
        'idp': +1,
        'idr': +1,
        'precision': +1,
        'recall': +1,
        'num_switches': -1,
        'num_fragmentations': -1,
        'num_false_positives': -1,
        'num_misses': -1,
    }

    for base, sign in delta_specs.items():
        a = f"{base}_trackssm"
        c = f"{base}_botsort"
        if a in merged.columns and c in merged.columns:
            merged[f"delta_{base}"] = sign * (merged[a] - merged[c])

    # A simple composite per-scene score delta (aligned to your grid-search score weights)
    # Score uses MOTA/IDF1/HOTA/IDSW; here HOTA is approximated as sqrt(idp*idr)
    for suffix in ['trackssm', 'botsort']:
        idp = merged.get(f'idp_{suffix}', np.nan) / 100.0
        idr = merged.get(f'idr_{suffix}', np.nan) / 100.0
        hota = np.sqrt(idp * idr) * 100.0
        merged[f'hota_{suffix}'] = hota

    merged['score_trackssm'] = (
        0.35 * merged.get('mota_trackssm', 0)
        + 0.30 * merged.get('idf1_trackssm', 0)
        + 0.25 * merged.get('hota_trackssm', 0)
        + 0.10 * np.maximum(0, 100 - (merged.get('num_switches_trackssm', 10000) / 30))
    )
    merged['score_botsort'] = (
        0.35 * merged.get('mota_botsort', 0)
        + 0.30 * merged.get('idf1_botsort', 0)
        + 0.25 * merged.get('hota_botsort', 0)
        + 0.10 * np.maximum(0, 100 - (merged.get('num_switches_botsort', 10000) / 30))
    )
    merged['delta_score'] = merged['score_trackssm'] - merged['score_botsort']

    return merged


def compute_summary_stats(delta_df: pd.DataFrame) -> dict:
    # Focus on the most thesis-relevant metrics
    metrics = ['delta_idf1', 'delta_mota', 'delta_num_switches', 'delta_score']
    stats = {}

    stats['n_scenes'] = int(len(delta_df))
    stats['wins_trackssm_score'] = int((delta_df['delta_score'] > 0).sum())
    stats['wins_botsort_score'] = int((delta_df['delta_score'] < 0).sum())
    stats['ties_score'] = int((delta_df['delta_score'] == 0).sum())

    for m in metrics:
        if m in delta_df.columns:
            stats[m] = {
                'mean': float(delta_df[m].mean()),
                'median': float(delta_df[m].median()),
                'p25': float(delta_df[m].quantile(0.25)),
                'p75': float(delta_df[m].quantile(0.75)),
                'min': float(delta_df[m].min()),
                'max': float(delta_df[m].max()),
            }

    # Simple correlations to motivate “quando succede” (crowdedness proxy)
    for col in ['obj_per_frame_trackssm', 'num_objects_trackssm', 'num_frames_trackssm']:
        if col in delta_df.columns and 'delta_score' in delta_df.columns:
            # Pearson correlation (robust enough for a quick write-up)
            x = delta_df[col].replace([np.inf, -np.inf], np.nan)
            y = delta_df['delta_score'].replace([np.inf, -np.inf], np.nan)
            valid = x.notna() & y.notna()
            if valid.sum() >= 10:
                stats[f'corr(delta_score, {col})'] = float(np.corrcoef(x[valid], y[valid])[0, 1])

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt-folder', default='data/nuscenes_mot_front/val')
    ap.add_argument('--trackssm-pred', default='results/FINAL_trackssm/data')
    ap.add_argument('--botsort-pred', default='results/FINAL_botsort/data')
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--out-dir', default='results/FINAL_COMPARISON')
    args = ap.parse_args()

    gt_folder = Path(args.gt_folder)
    trackssm_pred = Path(args.trackssm_pred)
    botsort_pred = Path(args.botsort_pred)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"GT: {gt_folder}")
    print(f"TrackSSM preds: {trackssm_pred}")
    print(f"BoT-SORT preds: {botsort_pred}")

    trackssm_df = add_derived_columns(evaluate_per_scene(gt_folder, trackssm_pred, args.iou))
    botsort_df = add_derived_columns(evaluate_per_scene(gt_folder, botsort_pred, args.iou))

    trackssm_df.to_csv(out_dir / 'per_scene_trackssm.csv', index=False)
    botsort_df.to_csv(out_dir / 'per_scene_botsort.csv', index=False)

    delta_df = merge_and_diff(trackssm_df, botsort_df)
    delta_df.to_csv(out_dir / 'per_scene_delta.csv', index=False)

    # Rankings
    top_trackssm = delta_df.sort_values('delta_score', ascending=False).head(15)
    top_botsort = delta_df.sort_values('delta_score', ascending=True).head(15)
    top_trackssm.to_csv(out_dir / 'top_trackssm_wins.csv', index=False)
    top_botsort.to_csv(out_dir / 'top_botsort_wins.csv', index=False)

    stats = compute_summary_stats(delta_df)
    with open(out_dir / 'per_scene_summary.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print("\nSaved:")
    for p in [
        out_dir / 'per_scene_trackssm.csv',
        out_dir / 'per_scene_botsort.csv',
        out_dir / 'per_scene_delta.csv',
        out_dir / 'top_trackssm_wins.csv',
        out_dir / 'top_botsort_wins.csv',
        out_dir / 'per_scene_summary.json',
    ]:
        print(f"  - {p}")

    # Console preview (useful for thesis writing)
    cols = ['scene', 'delta_score', 'delta_idf1', 'delta_mota', 'delta_num_switches', 'obj_per_frame_trackssm']
    cols = [c for c in cols if c in delta_df.columns]

    print("\nTop TrackSSM wins (by delta_score):")
    print(top_trackssm[cols].to_string(index=False))

    print("\nTop BoT-SORT wins (by delta_score):")
    print(top_botsort[cols].to_string(index=False))


if __name__ == '__main__':
    main()

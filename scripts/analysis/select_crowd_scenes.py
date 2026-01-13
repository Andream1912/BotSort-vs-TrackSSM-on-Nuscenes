#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


def max_gt_in_frame(gt_txt: Path) -> int:
    if not gt_txt.exists():
        return 0

    # GT is MOT: frame,id,x,y,w,h,conf,class,vis,unused
    df = pd.read_csv(
        gt_txt,
        header=None,
        usecols=[0],
        names=['frame'],
    )
    if df.empty:
        return 0
    return int(df.groupby('frame').size().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--delta-csv', default='results/FINAL_COMPARISON/per_scene_delta.csv')
    ap.add_argument('--gt-folder', default='data/nuscenes_mot_front/val')
    ap.add_argument('--min-delta-score', type=float, default=1.5)
    ap.add_argument('--top', type=int, default=15)
    args = ap.parse_args()

    delta = pd.read_csv(args.delta_csv)

    # Only TrackSSM wins, and not tiny wins
    cand = delta[(delta['delta_score'] > args.min_delta_score)].copy()
    cand = cand.sort_values('delta_score', ascending=False)

    gt_root = Path(args.gt_folder)

    max_counts = []
    for scene in cand['scene'].tolist():
        gt_txt = gt_root / scene / 'gt' / 'gt.txt'
        max_counts.append(max_gt_in_frame(gt_txt))

    cand['max_gt_per_frame'] = max_counts

    # Rank by crowding first, then by improvement
    ranked = cand.sort_values(['max_gt_per_frame', 'delta_score'], ascending=[False, False]).head(args.top)

    cols = [
        'scene',
        'max_gt_per_frame',
        'delta_score',
        'delta_idf1',
        'delta_mota',
        'delta_num_switches',
    ]
    cols = [c for c in cols if c in ranked.columns]

    out_path = Path('results/FINAL_COMPARISON/top_trackssm_crowd_candidates.csv')
    ranked[cols].to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")
    print(ranked[cols].to_string(index=False))


if __name__ == '__main__':
    main()

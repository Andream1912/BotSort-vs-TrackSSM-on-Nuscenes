#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image


@dataclass
class SceneStats:
    scene: str
    num_frames: int
    num_rows: int
    avg_gt_per_frame: float
    max_gt_per_frame: int
    brightness_mean: float
    is_night: bool
    complexity: str


def load_gt_counts(gt_txt: Path) -> tuple[int, int, float, int]:
    """Return (num_frames, num_rows, avg_objs_per_frame, max_objs_per_frame)."""
    if not gt_txt.exists():
        return 0, 0, 0.0, 0

    df = pd.read_csv(gt_txt, header=None, usecols=[0], names=['frame'])
    if df.empty:
        return 0, 0, 0.0, 0

    counts = df.groupby('frame').size()
    num_frames = int(counts.index.nunique())
    num_rows = int(len(df))
    avg = float(counts.mean())
    mx = int(counts.max())
    return num_frames, num_rows, avg, mx


def pick_representative_frame(scene_dir: Path) -> int:
    """Pick a frame id to represent the scene (middle frame if possible)."""
    img_dir = scene_dir / 'img1'
    if not img_dir.exists():
        return 1

    jpgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() == '.jpg'])
    if not jpgs:
        return 1

    mid = jpgs[len(jpgs) // 2]
    try:
        return int(mid.stem)
    except Exception:
        return 1


def compute_brightness(img_path: Path) -> float:
    if not img_path.exists():
        return float('nan')

    img = Image.open(img_path).convert('RGB')
    arr = np.asarray(img, dtype=np.float32)
    # perceived luminance
    lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    return float(np.mean(lum))


def classify_complexity(avg_gt_per_frame: float, max_gt_per_frame: int) -> str:
    """Binary split: simple vs complex.

    Heuristic chosen to match intuition on nuScenes front-camera MOT:
    - complex if either sustained density is high (avg >= 6) or peak density is high (max >= 18)
    - else simple
    """
    if avg_gt_per_frame >= 6.0 or max_gt_per_frame >= 18:
        return 'complex'
    return 'simple'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt-folder', default='data/nuscenes_mot_front/val')
    ap.add_argument('--out-dir', default='results/FINAL_COMPARISON')
    ap.add_argument('--night-threshold', type=float, default=60.0,
                    help='Mean luminance threshold under which a frame is considered night')
    args = ap.parse_args()

    gt_root = Path(args.gt_folder)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = sorted([d for d in gt_root.iterdir() if d.is_dir() and (d / 'gt' / 'gt.txt').exists()])

    rows: list[SceneStats] = []
    for scene_dir in scenes:
        scene = scene_dir.name
        gt_txt = scene_dir / 'gt' / 'gt.txt'
        num_frames, num_rows, avg_pf, max_pf = load_gt_counts(gt_txt)

        frame = pick_representative_frame(scene_dir)
        img_path = scene_dir / 'img1' / f"{frame:06d}.jpg"
        brightness = compute_brightness(img_path)
        is_night = bool(np.isfinite(brightness) and brightness < args.night_threshold)

        complexity = classify_complexity(avg_pf, max_pf)

        rows.append(SceneStats(
            scene=scene,
            num_frames=num_frames,
            num_rows=num_rows,
            avg_gt_per_frame=avg_pf,
            max_gt_per_frame=max_pf,
            brightness_mean=brightness,
            is_night=is_night,
            complexity=complexity,
        ))

    df = pd.DataFrame([r.__dict__ for r in rows])

    out_csv = out_dir / 'scene_complexity_report.csv'
    df.to_csv(out_csv, index=False)

    summary = {
        'n_scenes': int(len(df)),
        'night_threshold': float(args.night_threshold),
        'simple': int((df['complexity'] == 'simple').sum()),
        'complex': int((df['complexity'] == 'complex').sum()),
        'night': int(df['is_night'].sum()),
        'night_simple': int(((df['is_night']) & (df['complexity'] == 'simple')).sum()),
        'night_complex': int(((df['is_night']) & (df['complexity'] == 'complex')).sum()),
        'avg_max_gt_per_frame': float(df['max_gt_per_frame'].mean()) if len(df) else 0.0,
        'avg_avg_gt_per_frame': float(df['avg_gt_per_frame'].mean()) if len(df) else 0.0,
    }

    out_json = out_dir / 'scene_complexity_summary.json'
    out_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    # Print small human-readable summary
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

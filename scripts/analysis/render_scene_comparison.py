#!/usr/bin/env python3

import argparse
from pathlib import Path
import hashlib

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def load_mot_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis', 'unused'])
    return pd.read_csv(
        path,
        header=None,
        names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis', 'unused'],
    )


def stable_color(track_id: int) -> tuple[int, int, int]:
    h = hashlib.md5(str(int(track_id)).encode('utf-8')).digest()
    # Bright-ish colors
    r = 64 + (h[0] % 192)
    g = 64 + (h[1] % 192)
    b = 64 + (h[2] % 192)
    return int(r), int(g), int(b)


def draw_boxes(img: Image.Image, df: pd.DataFrame, title: str) -> Image.Image:
    out = img.copy().convert('RGB')
    draw = ImageDraw.Draw(out)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Header
    draw.rectangle([(0, 0), (out.width, 32)], fill=(0, 0, 0))
    draw.text((10, 7), title, fill=(255, 255, 255), font=font)

    for _, r in df.iterrows():
        x, y, w, h = float(r.x), float(r.y), float(r.w), float(r.h)
        tid = int(r.id)
        color = stable_color(tid)

        x1, y1, x2, y2 = x, y, x + w, y + h
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        draw.text((x1 + 2, max(34, y1 + 2)), str(tid), fill=color, font=font_small)

    return out


def choose_frame(gt_df: pd.DataFrame) -> int:
    # Pick the frame with max number of GT objects (dense moment)
    counts = gt_df.groupby('frame').size().sort_values(ascending=False)
    if counts.empty:
        return 1
    return int(counts.index[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True, help='e.g. scene-0557_CAM_FRONT')
    ap.add_argument('--gt-folder', default='data/nuscenes_mot_front/val')
    ap.add_argument('--trackssm-pred', default='results/FINAL_trackssm/data')
    ap.add_argument('--botsort-pred', default='results/FINAL_botsort/data')
    ap.add_argument('--frame', type=int, default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    gt_root = Path(args.gt_folder)
    scene_dir = gt_root / args.scene
    gt_file = scene_dir / 'gt' / 'gt.txt'
    img_dir = scene_dir / 'img1'

    botsort_pred_file = Path(args.botsort_pred) / f"{args.scene}.txt"
    trackssm_pred_file = Path(args.trackssm_pred) / f"{args.scene}.txt"

    gt_df = load_mot_df(gt_file)
    b_df = load_mot_df(botsort_pred_file)
    t_df = load_mot_df(trackssm_pred_file)

    frame = args.frame if args.frame is not None else choose_frame(gt_df)

    img_path = img_dir / f"{frame:06d}.jpg"
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = Image.open(img_path)

    b_frame = b_df[b_df['frame'] == frame]
    t_frame = t_df[t_df['frame'] == frame]

    left = draw_boxes(img, b_frame, f"BoT-SORT | {args.scene} | frame {frame}")
    right = draw_boxes(img, t_frame, f"TrackSSM | {args.scene} | frame {frame}")

    # Side-by-side
    canvas = Image.new('RGB', (left.width + right.width, max(left.height, right.height)))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)

    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()

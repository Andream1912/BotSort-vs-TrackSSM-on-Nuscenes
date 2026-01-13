#!/usr/bin/env python3

import argparse
from pathlib import Path
import random

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def pick_frame(scene_dir: Path) -> int:
    img_dir = scene_dir / 'img1'
    jpgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() == '.jpg'])
    if not jpgs:
        return 1
    mid = jpgs[len(jpgs) // 2]
    try:
        return int(mid.stem)
    except Exception:
        return 1


def load_img(scene_dir: Path, frame: int) -> Image.Image:
    img_path = scene_dir / 'img1' / f"{frame:06d}.jpg"
    return Image.open(img_path).convert('RGB')


def draw_label(img: Image.Image, lines: list[str]) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    pad = 10
    # Background box height
    text_h = 0
    text_w = 0
    for ln in lines:
        bbox = draw.textbbox((0, 0), ln, font=font)
        text_w = max(text_w, bbox[2] - bbox[0])
        text_h += (bbox[3] - bbox[1]) + 2

    box_h = text_h + 2 * pad
    draw.rectangle([(0, 0), (out.width, box_h)], fill=(0, 0, 0))

    y = pad
    for ln in lines:
        draw.text((pad, y), ln, fill=(255, 255, 255), font=font)
        bbox = draw.textbbox((0, 0), ln, font=font)
        y += (bbox[3] - bbox[1]) + 2

    return out


def choose_scenes(report_csv: Path, n_each: int, seed: int) -> dict:
    df = pd.read_csv(report_csv)

    # Categories we want to show
    # - simple day
    # - complex day
    # - night (prefer complex night, else any night)
    rng = random.Random(seed)

    simple_day = df[(df['complexity'] == 'simple') & (~df['is_night'])].copy()
    complex_day = df[(df['complexity'] == 'complex') & (~df['is_night'])].copy()
    night_any = df[df['is_night']].copy()
    night_complex = df[(df['is_night']) & (df['complexity'] == 'complex')].copy()
    night_simple = df[(df['is_night']) & (df['complexity'] == 'simple')].copy()

    # Rank by crowding for complex examples
    complex_day = complex_day.sort_values(['max_gt_per_frame', 'avg_gt_per_frame'], ascending=False)

    # Rank by "simplicity" for simple examples
    simple_day = simple_day.sort_values(['max_gt_per_frame', 'avg_gt_per_frame'], ascending=True)

    picks = []

    def sample_rows(sub: pd.DataFrame, k: int, fallback: pd.DataFrame | None = None):
        nonlocal picks
        if len(sub) >= k:
            # take k with some diversity: random choice from top 30% (or all if small)
            pool = sub.head(max(k, int(len(sub) * 0.3)))
            chosen = pool.sample(n=k, random_state=seed)
            picks.extend(chosen.to_dict('records'))
            return
        if fallback is not None and len(fallback) >= k:
            chosen = fallback.sample(n=k, random_state=seed)
            picks.extend(chosen.to_dict('records'))
            return
        # last resort: whatever exists
        if len(sub) > 0:
            picks.extend(sub.to_dict('records')[:k])

    # We want total 6 images: 2 simple day, 2 complex day, 2 night (1 simple night + 1 complex night if possible)
    sample_rows(simple_day, 2)
    sample_rows(complex_day, 2)

    if len(night_complex) >= 1:
        picks.append(night_complex.sample(n=1, random_state=seed).iloc[0].to_dict())
    elif len(night_any) >= 1:
        picks.append(night_any.sample(n=1, random_state=seed).iloc[0].to_dict())

    if len(night_simple) >= 1:
        picks.append(night_simple.sample(n=1, random_state=seed + 1).iloc[0].to_dict())
    elif len(night_any) >= 2:
        picks.append(night_any.sample(n=1, random_state=seed + 1).iloc[0].to_dict())

    # Ensure we have exactly 6 and unique scenes
    unique = {}
    for r in picks:
        unique[r['scene']] = r

    picks = list(unique.values())
    if len(picks) > 6:
        rng.shuffle(picks)
        picks = picks[:6]

    # If still less than 6, fill with random scenes
    if len(picks) < 6:
        remaining = df[~df['scene'].isin([p['scene'] for p in picks])]
        if len(remaining) > 0:
            fill = remaining.sample(n=min(6 - len(picks), len(remaining)), random_state=seed + 2)
            picks.extend(fill.to_dict('records'))

    return {'picks': picks}


def montage(images: list[Image.Image], labels: list[list[str]], cols: int, tile_w: int, tile_h: int, pad: int) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    canvas_w = cols * tile_w + (cols + 1) * pad
    canvas_h = rows * tile_h + (rows + 1) * pad

    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))

    for i, (img, lab) in enumerate(zip(images, labels)):
        r = i // cols
        c = i % cols
        x0 = pad + c * (tile_w + pad)
        y0 = pad + r * (tile_h + pad)

        # Fit center crop
        im = img.copy()
        iw, ih = im.size
        target_ar = tile_w / tile_h
        cur_ar = iw / ih

        if cur_ar > target_ar:
            # too wide, crop width
            new_w = int(ih * target_ar)
            left = (iw - new_w) // 2
            im = im.crop((left, 0, left + new_w, ih))
        else:
            # too tall, crop height
            new_h = int(iw / target_ar)
            top = (ih - new_h) // 2
            im = im.crop((0, top, iw, top + new_h))

        im = im.resize((tile_w, tile_h), Image.BICUBIC)
        im = draw_label(im, lab)

        canvas.paste(im, (x0, y0))

    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt-folder', default='data/nuscenes_mot_front/val')
    ap.add_argument('--report-csv', default='results/FINAL_COMPARISON/scene_complexity_report.csv')
    ap.add_argument('--out', default='/user/amarino/Elaborato/images/dataset_diversity_montage.png')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--tile-w', type=int, default=820)
    ap.add_argument('--tile-h', type=int, default=460)
    ap.add_argument('--cols', type=int, default=3)
    ap.add_argument('--pad', type=int, default=18)
    args = ap.parse_args()

    gt_root = Path(args.gt_folder)
    report_csv = Path(args.report_csv)

    sel = choose_scenes(report_csv, n_each=2, seed=args.seed)
    picks = sel['picks']

    images = []
    labels = []

    for r in picks:
        scene = r['scene']
        scene_dir = gt_root / scene
        frame = pick_frame(scene_dir)
        img = load_img(scene_dir, frame)

        cplx = str(r['complexity']).upper()
        night = 'NIGHT' if bool(r['is_night']) else 'DAY'
        max_pf = int(r.get('max_gt_per_frame', 0))
        avg_pf = float(r.get('avg_gt_per_frame', 0.0))

        images.append(img)
        labels.append([
            scene,
            f"{cplx} | {night}",
            f"GT peak={max_pf}  avg={avg_pf:.1f}/frame",
        ])

    out_img = montage(images, labels, cols=args.cols, tile_w=args.tile_w, tile_h=args.tile_h, pad=args.pad)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)

    # Also save the chosen scenes list for reproducibility
    out_list = Path('results/FINAL_COMPARISON/dataset_montage_selection.json')
    out_list.write_text(pd.DataFrame(picks).to_json(orient='records', indent=2), encoding='utf-8')

    print(f"Saved montage: {out_path}")
    print(f"Saved selection: {out_list}")
    print("Chosen scenes:")
    for r in picks:
        print(f"  - {r['scene']} | {r['complexity']} | night={bool(r['is_night'])} | peak={int(r.get('max_gt_per_frame', 0))}")


if __name__ == '__main__':
    main()

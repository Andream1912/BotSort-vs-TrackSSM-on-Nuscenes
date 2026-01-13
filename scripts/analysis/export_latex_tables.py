#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


def fmt(x, nd=2, sign=False):
    if pd.isna(x):
        return "--"
    if isinstance(x, (int,)):
        return str(x)
    if sign:
        return f"{x:+.{nd}f}"
    return f"{x:.{nd}f}"


def make_table(df: pd.DataFrame, title: str, label: str, n: int) -> str:
    cols = ['scene', 'delta_score', 'delta_idf1', 'delta_mota', 'delta_num_switches']
    cols = [c for c in cols if c in df.columns]
    df = df[cols].head(n).copy()

    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("  \\centering")
    lines.append(f"  \\caption{{{title}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("  \\small")
    lines.append("  \\begin{tabular}{lrrrr}")
    lines.append("    \\toprule")
    lines.append("    Scene & $\\Delta$Score & $\\Delta$IDF1 & $\\Delta$MOTA & $\\Delta$IDSW \\\\")
    lines.append("    \\midrule")

    for _, r in df.iterrows():
        scene = str(r['scene']).replace('_', '\\_')
        dscore = fmt(r.get('delta_score'), 2, sign=True)
        didf1 = fmt(r.get('delta_idf1'), 2, sign=True)
        dmota = fmt(r.get('delta_mota'), 2, sign=True)
        didsw = fmt(r.get('delta_num_switches'), 0, sign=True)
        lines.append(f"    {scene} & {dscore} & {didf1} & {dmota} & {didsw} \\\\")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', default='results/FINAL_COMPARISON')
    ap.add_argument('--out-tex', default='/user/amarino/Elaborato/chapters/per_scene_tables.tex')
    ap.add_argument('-n', type=int, default=5)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    top_t = pd.read_csv(in_dir / 'top_trackssm_wins.csv')
    top_b = pd.read_csv(in_dir / 'top_botsort_wins.csv')

    # For BoT-SORT wins, invert sign columns so that positive means “BoT-SORT better” in table.
    # delta_score in the CSV is TrackSSM - BoT-SORT; so we flip it.
    flip_cols = ['delta_score', 'delta_idf1', 'delta_mota', 'delta_num_switches']
    for c in flip_cols:
        if c in top_b.columns:
            top_b[c] = -top_b[c]

    parts = []
    parts.append(make_table(
        top_t,
        title=f"Top-{args.n} scene in cui TrackSSM migliora di pi\u00f9 (val split, configurazione finale).",
        label="tab:top_trackssm_scenes",
        n=args.n,
    ))
    parts.append(make_table(
        top_b,
        title=f"Top-{args.n} scene in cui BoT-SORT migliora di pi\u00f9 (val split, configurazione finale).",
        label="tab:top_botsort_scenes",
        n=args.n,
    ))

    out_tex = Path(args.out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(parts), encoding='utf-8')
    print(f"Wrote: {out_tex}")


if __name__ == '__main__':
    main()

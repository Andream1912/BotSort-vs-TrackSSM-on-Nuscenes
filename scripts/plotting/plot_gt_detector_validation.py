#!/usr/bin/env python3
"""Plot thesis-ready summary for GT-detector (oracle) validation.

Reads a single `metrics.json` produced by `scripts/evaluation/evaluate_motmetrics.py`
and generates a compact PNG dashboard (percent metrics + count/ratio summary).

Usage example:
  python scripts/plotting/plot_gt_detector_validation.py \
    --metrics results/thesis/gt_detector_validation/botsort_default_gt_full/metrics.json \
    --output  results/thesis/gt_detector_validation/botsort_default_gt_full/gt_detector_validation_summary.png \
    --thesis-output ../Elaborato/images/gt_detector_validation_botsort_default.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _safe_get(metrics: dict, key: str, default):
    value = metrics.get(key, default)
    return default if value is None else value


def plot_dashboard(metrics_path: Path, output_path: Path, title: str | None) -> None:
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    percent_metric_specs = [
        ("MOTA", "MOTA"),
        ("IDF1", "IDF1"),
        ("HOTA", "HOTA*"),
        ("IDP", "IDP"),
        ("IDR", "IDR"),
        ("precision", "Prec."),
        ("recall", "Rec."),
    ]

    labels = [label for _, label in percent_metric_specs]
    values = [float(_safe_get(metrics, key, 0.0)) for key, _ in percent_metric_specs]

    # Styling tuned for thesis figures
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.linewidth"] = 1.0

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(12.8, 4.9), gridspec_kw={"width_ratios": [1.15, 1.05]}
    )

    # Left: horizontal bars (more readable in print)
    y = np.arange(len(values))
    bars = ax_left.barh(y, values, color="#4C78A8", alpha=0.9)
    ax_left.set_yticks(y)
    ax_left.set_yticklabels(labels)
    ax_left.invert_yaxis()
    lo = min(values) if values else 0.0
    if lo < 0:
        ax_left.set_xlim(lo - 5.0, 100)
    else:
        ax_left.set_xlim(0, 100)
    ax_left.set_xlabel("Score (%)")
    ax_left.set_title("Summary metrics")
    ax_left.grid(True, axis="x", linestyle="--", alpha=0.35)

    for bar, v in zip(bars, values):
        y_text = bar.get_y() + bar.get_height() / 2.0
        if v >= 0:
            x_text = min(v + 1.0, 99.0)
            ha = "left"
        else:
            x_text = v - 1.0
            ha = "right"

        ax_left.text(
            x_text,
            y_text,
            f"{v:.1f}%",
            va="center",
            ha=ha,
            fontsize=10,
            fontweight="bold",
            color="#1a1a1a",
        )

    # Right: compact table summary
    ax_right.axis("off")

    idsw = int(_safe_get(metrics, "IDSW", 0))
    fp = int(_safe_get(metrics, "FP", 0))
    fn = int(_safe_get(metrics, "FN", 0))
    frags = int(_safe_get(metrics, "num_fragmentations", 0))
    tp = int(_safe_get(metrics, "TP", 0))

    mt = int(_safe_get(metrics, "mostly_tracked", 0))
    pt = int(_safe_get(metrics, "partially_tracked", 0))
    ml = int(_safe_get(metrics, "mostly_lost", 0))

    mt_ratio = float(_safe_get(metrics, "MT_ratio", 0.0))
    pt_ratio = float(_safe_get(metrics, "PT_ratio", 0.0))
    ml_ratio = float(_safe_get(metrics, "ML_ratio", 0.0))

    frames = int(_safe_get(metrics, "num_frames", 0))
    gt_ids = int(_safe_get(metrics, "num_unique_objects", 0))
    gt_objs = int(_safe_get(metrics, "num_objects", 0))
    preds = int(_safe_get(metrics, "num_predictions", 0))

    table_rows = [
        ["TP", f"{tp:,}"],
        ["FP", f"{fp:,}"],
        ["FN", f"{fn:,}"],
        ["IDSW", f"{idsw:,}"],
        ["Frag", f"{frags:,}"],
        ["MT", f"{mt:,}  ({mt_ratio:.1f}%)"],
        ["PT", f"{pt:,}  ({pt_ratio:.1f}%)"],
        ["ML", f"{ml:,}  ({ml_ratio:.1f}%)"],
        ["Frames", f"{frames:,}"],
        ["GT IDs", f"{gt_ids:,}"],
        ["GT objects", f"{gt_objs:,}"],
        ["Predictions", f"{preds:,}"],
    ]

    tbl = ax_right.table(
        cellText=table_rows,
        colLabels=["Indicator", "Value"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1.0, 1.25)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_linewidth(0.5)
        cell.set_edgecolor("#d0d0d0")
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#f3f3f3")

    if title:
        fig.suptitle(title, fontweight="bold", y=1.02)

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_barchart(metrics_path: Path, output_path: Path, title: str | None) -> None:
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    percent_metric_specs = [
        ("MOTA", "MOTA"),
        ("IDF1", "IDF1"),
        ("HOTA", "HOTA*"),
        ("IDP", "IDP"),
        ("IDR", "IDR"),
        ("precision", "Prec."),
        ("recall", "Rec."),
    ]

    labels = [label for _, label in percent_metric_specs]
    values = [float(_safe_get(metrics, key, 0.0)) for key, _ in percent_metric_specs]

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.linewidth"] = 1.0

    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    x = np.arange(len(values))

    bars = ax.bar(x, values, color="#4C78A8", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_title("Summary metrics" if not title else title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    lo = min(values) if values else 0.0
    if lo < 0:
        ax.set_ylim(lo - 5.0, 100)
    else:
        ax.set_ylim(0, 100)

    for bar, v in zip(bars, values):
        x_text = bar.get_x() + bar.get_width() / 2.0
        if v >= 0:
            y_text = min(v + 1.0, 99.0)
            va = "bottom"
        else:
            y_text = v - 1.0
            va = "top"

        ax.text(
            x_text,
            y_text,
            f"{v:.1f}%",
            ha="center",
            va=va,
            fontsize=10,
            fontweight="bold",
            color="#1a1a1a",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, type=Path, help="Path to metrics.json")
    parser.add_argument("--output", required=True, type=Path, help="Output PNG path")
    parser.add_argument(
        "--thesis-output",
        type=Path,
        default=None,
        help="Optional: also write/copy the PNG to this path (e.g., ../Elaborato/images/...)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="BoT-SORT validation with GT detections (nuScenes val)",
        help="Figure title",
    )
    parser.add_argument(
        "--output-barchart",
        type=Path,
        default=None,
        help="Optional: output PNG path for a bar-chart-only version (*_barchart.png)",
    )
    parser.add_argument(
        "--thesis-output-barchart",
        type=Path,
        default=None,
        help="Optional: also write/copy the bar-chart-only PNG to this path",
    )

    args = parser.parse_args()

    if not args.metrics.exists():
        raise FileNotFoundError(f"metrics file not found: {args.metrics}")

    plot_dashboard(args.metrics, args.output, args.title)

    if args.thesis_output is not None:
        args.thesis_output.parent.mkdir(parents=True, exist_ok=True)
        args.thesis_output.write_bytes(args.output.read_bytes())

    if args.output_barchart is not None:
        plot_barchart(args.metrics, args.output_barchart, title=None)
        if args.thesis_output_barchart is not None:
            args.thesis_output_barchart.parent.mkdir(parents=True, exist_ok=True)
            args.thesis_output_barchart.write_bytes(args.output_barchart.read_bytes())

    print(f"✓ Wrote plot: {args.output}")
    if args.thesis_output is not None:
        print(f"✓ Copied plot to: {args.thesis_output}")

    if args.output_barchart is not None:
        print(f"✓ Wrote barchart: {args.output_barchart}")
        if args.thesis_output_barchart is not None:
            print(f"✓ Copied barchart to: {args.thesis_output_barchart}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

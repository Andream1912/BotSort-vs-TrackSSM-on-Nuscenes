#!/usr/bin/env python3
"""Create thesis-style plots summarizing grid-search hyperparameter optimization.

This reads experiments stored as:
  results/GRID_SEARCH/exp_0001/{config.json, metrics.json}

Outputs:
- A compact dashboard (hyperparams + key metrics + summary table)
- A top-k score bar chart

Designed to match the visual style used by other thesis figures in
`scripts/plotting/` (seaborn whitegrid + 200dpi + bold labels).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


WEIGHTS = {
    "MOTA": 0.35,
    "IDF1": 0.30,
    "HOTA": 0.25,
    "IDSW": -0.10,
}


def compute_score(metrics: dict) -> float:
    mota = float(metrics.get("MOTA", 0) or 0)
    idf1 = float(metrics.get("IDF1", 0) or 0)
    hota = float(metrics.get("HOTA", 0) or 0)
    idsw = float(metrics.get("IDSW", 10000) or 10000)
    idsw_norm = max(0.0, 100.0 - (idsw / 30.0))
    return (
        WEIGHTS["MOTA"] * mota
        + WEIGHTS["IDF1"] * idf1
        + WEIGHTS["HOTA"] * hota
        + WEIGHTS["IDSW"] * idsw_norm
    )


@dataclass(frozen=True)
class Experiment:
    exp_id: int
    config: dict
    metrics: dict
    score: float


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_experiments(grid_dir: Path) -> list[Experiment]:
    experiments: list[Experiment] = []
    for exp_dir in sorted(grid_dir.glob("exp_*")):
        cfg_path = exp_dir / "config.json"
        met_path = exp_dir / "metrics.json"
        if not cfg_path.exists() or not met_path.exists():
            continue

        try:
            cfg = _read_json(cfg_path)
            metrics = _read_json(met_path)
            exp_id = int(cfg.get("experiment_id") or cfg.get("exp_id") or exp_dir.name.split("_")[-1])
            config = cfg.get("config", {})
            score = compute_score(metrics)
            experiments.append(Experiment(exp_id=exp_id, config=config, metrics=metrics, score=score))
        except Exception:
            # Skip malformed experiments
            continue

    return experiments


def _apply_thesis_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.linewidth"] = 1.0


def _safe_get(metrics: dict, key: str, default):
    value = metrics.get(key, default)
    return default if value is None else value


def plot_dashboard(best: Experiment, output_path: Path, title: str | None) -> None:
    _apply_thesis_style()

    # --- Figure layout (2 rows left, 1 big table right)
    fig = plt.figure(figsize=(12.8, 6.4))
    gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[1.25, 1.05], height_ratios=[1.0, 1.0])

    ax_params = fig.add_subplot(gs[0, 0])
    ax_metrics = fig.add_subplot(gs[1, 0])
    ax_table = fig.add_subplot(gs[:, 1])

    # --- Top-left: optimized hyperparameters
    param_specs = [
        ("conf_thresh", "conf"),
        ("match_thresh", "match"),
        ("track_thresh", "track"),
        ("nms_thresh", "nms"),
    ]
    p_labels = [lbl for _, lbl in param_specs]
    p_values = [float(best.config.get(k, 0.0) or 0.0) for k, _ in param_specs]

    y = np.arange(len(p_values))
    bars = ax_params.barh(y, p_values, color="#4C78A8", alpha=0.9)
    ax_params.set_yticks(y)
    ax_params.set_yticklabels(p_labels)
    ax_params.invert_yaxis()
    ax_params.set_xlim(0.0, 1.0)
    ax_params.set_xlabel("Value")
    ax_params.set_title("Optimized hyperparameters")
    ax_params.grid(True, axis="x", linestyle="--", alpha=0.35)

    for bar, v in zip(bars, p_values):
        ax_params.text(
            min(v + 0.02, 0.98),
            bar.get_y() + bar.get_height() / 2.0,
            f"{v:.2f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#1a1a1a",
        )

    # --- Bottom-left: key tracking metrics (percent)
    metric_specs = [
        ("MOTA", "MOTA"),
        ("IDF1", "IDF1"),
        ("HOTA", "HOTA"),
        ("precision", "Prec."),
        ("recall", "Rec."),
    ]
    m_labels = [lbl for _, lbl in metric_specs]
    m_values = [float(_safe_get(best.metrics, k, 0.0)) for k, _ in metric_specs]

    y2 = np.arange(len(m_values))
    bars2 = ax_metrics.barh(y2, m_values, color="#72B7B2", alpha=0.9)
    ax_metrics.set_yticks(y2)
    ax_metrics.set_yticklabels(m_labels)
    ax_metrics.invert_yaxis()

    lo = min(m_values) if m_values else 0.0
    if lo < 0:
        ax_metrics.set_xlim(lo - 5.0, 100)
    else:
        ax_metrics.set_xlim(0, 100)

    ax_metrics.set_xlabel("Score (%)")
    ax_metrics.set_title("Best-run metrics")
    ax_metrics.grid(True, axis="x", linestyle="--", alpha=0.35)

    for bar, v in zip(bars2, m_values):
        y_text = bar.get_y() + bar.get_height() / 2.0
        if v >= 0:
            x_text = min(v + 1.0, 99.0)
            ha = "left"
        else:
            x_text = v - 1.0
            ha = "right"

        ax_metrics.text(
            x_text,
            y_text,
            f"{v:.1f}%",
            va="center",
            ha=ha,
            fontsize=10,
            fontweight="bold",
            color="#1a1a1a",
        )

    # --- Right: compact table summary
    ax_table.axis("off")

    idsw = int(_safe_get(best.metrics, "IDSW", 0))
    fp = int(_safe_get(best.metrics, "FP", 0))
    fn = int(_safe_get(best.metrics, "FN", 0))
    frags = int(_safe_get(best.metrics, "num_fragmentations", 0))
    tp = int(_safe_get(best.metrics, "TP", 0))

    mt = int(_safe_get(best.metrics, "mostly_tracked", 0))
    pt = int(_safe_get(best.metrics, "partially_tracked", 0))
    ml = int(_safe_get(best.metrics, "mostly_lost", 0))

    mt_ratio = float(_safe_get(best.metrics, "MT_ratio", 0.0))
    pt_ratio = float(_safe_get(best.metrics, "PT_ratio", 0.0))
    ml_ratio = float(_safe_get(best.metrics, "ML_ratio", 0.0))

    frames = int(_safe_get(best.metrics, "num_frames", 0))
    gt_ids = int(_safe_get(best.metrics, "num_unique_objects", 0))
    gt_objs = int(_safe_get(best.metrics, "num_objects", 0))
    preds = int(_safe_get(best.metrics, "num_predictions", 0))

    table_rows = [
        ["Best exp", f"{best.exp_id:04d}"],
        ["Score", f"{best.score:.2f}"],
        ["conf/match", f"{best.config.get('conf_thresh', 0):.2f} / {best.config.get('match_thresh', 0):.2f}"],
        ["track/nms", f"{best.config.get('track_thresh', 0):.2f} / {best.config.get('nms_thresh', 0):.2f}"],
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

    tbl = ax_table.table(
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
        fig.suptitle(title, fontweight="bold", y=1.01)

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_topk(experiments: list[Experiment], topk: int, output_path: Path, title: str | None) -> None:
    _apply_thesis_style()

    if not experiments:
        raise ValueError("no experiments to plot")

    best_sorted = sorted(experiments, key=lambda e: e.score, reverse=True)[: max(1, topk)]

    labels = [f"#{e.exp_id:04d}" for e in best_sorted]
    values = [e.score for e in best_sorted]

    fig, ax = plt.subplots(figsize=(12.8, 4.6))

    x = np.arange(len(values))
    bars = ax.bar(x, values, color="#4C78A8", alpha=0.9, edgecolor="black", linewidth=0.8)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.2,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Composite score")
    ax.set_xlabel("Experiment")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    if title:
        ax.set_title(title, fontweight="bold", pad=12)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid-dir",
        type=Path,
        default=Path("results/GRID_SEARCH"),
        help="Grid search directory (contains exp_* subfolders)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/GRID_SEARCH/thesis_plots"),
        help="Where to write PNGs",
    )
    parser.add_argument(
        "--thesis-dir",
        type=Path,
        default=None,
        help="Optional: thesis image folder (e.g. ../Elaborato/images)",
    )
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--title",
        type=str,
        default="Grid search: BoT-SORT hyperparameter optimization (nuScenes val)",
    )

    args = parser.parse_args()

    if not args.grid_dir.exists():
        raise FileNotFoundError(f"grid search dir not found: {args.grid_dir}")

    experiments = load_experiments(args.grid_dir)
    if not experiments:
        raise RuntimeError(f"no completed experiments found under {args.grid_dir}")

    best = max(experiments, key=lambda e: e.score)

    out_dashboard = args.output_dir / "grid_search_dashboard.png"
    out_topk = args.output_dir / "grid_search_topk.png"

    plot_dashboard(best, out_dashboard, args.title)
    plot_topk(experiments, args.topk, out_topk, f"Top-{max(1, args.topk)} configurations (composite score)")

    if args.thesis_dir is not None:
        args.thesis_dir.mkdir(parents=True, exist_ok=True)
        (args.thesis_dir / out_dashboard.name).write_bytes(out_dashboard.read_bytes())
        (args.thesis_dir / out_topk.name).write_bytes(out_topk.read_bytes())

    print(f"✓ Wrote: {out_dashboard}")
    print(f"✓ Wrote: {out_topk}")
    if args.thesis_dir is not None:
        print(f"✓ Copied to thesis dir: {args.thesis_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

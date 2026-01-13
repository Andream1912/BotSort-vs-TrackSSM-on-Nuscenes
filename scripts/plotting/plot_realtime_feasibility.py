#!/usr/bin/env python3
"""Generate slide-ready figures for *Real Time Feasibility*.

This script reads one or more `benchmark.json` files produced by `track.py --benchmark`
(and saved under `{output}/benchmark.json`) and renders clean, presentation-ready PNGs.

Defaults are wired to the repo's thesis benchmark folder:
    results/thesis/realtime_benchmark/

Outputs (PNG, slide-ready):
    - realtime_feasibility_trackssm_fps.png
    - realtime_feasibility_trackssm_latency.png
    - realtime_feasibility_io_vs_compute.png

Usage:
  python scripts/plotting/plot_realtime_feasibility.py \
    --root results/thesis/realtime_benchmark \
    --thesis-output ../Elaborato/images
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Bench:
    label: str
    path: Path
    tracker: str
    fps_overall: float
    total_mean_ms: float
    total_median_ms: float
    total_p95_ms: float
    detect_median_ms: float
    track_median_ms: float
    read_median_ms: float | None
    include_io: bool

    @property
    def fps_median(self) -> float:
        return float(1000.0 / self.total_median_ms) if self.total_median_ms > 0 else 0.0

    @property
    def fps_p95(self) -> float:
        return float(1000.0 / self.total_p95_ms) if self.total_p95_ms > 0 else 0.0


def _read_benchmark(path: Path, label: str) -> Bench:
    with open(path, "r") as f:
        data = json.load(f)

    def _get_nested(d: dict, k: str, sub: str, default=None):
        v = d.get(k) or {}
        out = v.get(sub, default)
        if out is None:
            return None
        return float(out)

    read_median = _get_nested(data, "read_ms", "median_ms", default=None)
    include_io = bool(data.get("include_io", False))

    return Bench(
        label=label,
        path=path,
        tracker=str(data.get("tracker", "")),
        fps_overall=float(data.get("fps_overall", 0.0) or 0.0),
        total_mean_ms=float(_get_nested(data, "total_ms", "mean_ms", default=0.0) or 0.0),
        total_median_ms=float(_get_nested(data, "total_ms", "median_ms", default=0.0) or 0.0),
        total_p95_ms=float(_get_nested(data, "total_ms", "p95_ms", default=0.0) or 0.0),
        detect_median_ms=float(_get_nested(data, "detect_ms", "median_ms", default=0.0) or 0.0),
        track_median_ms=float(_get_nested(data, "track_ms", "median_ms", default=0.0) or 0.0),
        read_median_ms=(None if read_median is None else float(read_median)),
        include_io=include_io,
    )


def _setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 220
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.linewidth"] = 1.0


def _annotate_bar(ax, bar, text: str) -> None:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.35,
        text,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#111",
    )


def plot_trackssm_fps(configs: list[Bench], out: Path, target_fps: float = 25.0) -> None:
    """Plot mean-FPS vs median-FPS for selected TrackSSM configs."""
    _setup_style()
    labels = [b.label for b in configs]
    x = np.arange(len(labels))

    mean_fps = [b.fps_overall for b in configs]
    median_fps = [b.fps_median for b in configs]

    width = 0.38
    fig, ax = plt.subplots(figsize=(12.6, 5.1))

    bars1 = ax.bar(
        x - width / 2,
        mean_fps,
        width,
        label="Mean FPS (sensitive to spikes)",
        color="#ff7f0e",
        alpha=0.9,
        edgecolor="#222",
        linewidth=0.9,
    )
    bars2 = ax.bar(
        x + width / 2,
        median_fps,
        width,
        label="Median FPS (typical)",
        color="#1f77b4",
        alpha=0.9,
        edgecolor="#222",
        linewidth=0.9,
    )

    ax.axhline(y=target_fps, color="#2ca02c", linestyle="--", linewidth=2.0, alpha=0.9)
    ax.text(
        0.99,
        target_fps + 0.6,
        f"Target: {target_fps:.0f} FPS",
        ha="right",
        va="bottom",
        transform=ax.get_yaxis_transform(),
        fontweight="bold",
        color="#2ca02c",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("FPS")
    ax.set_title("TrackSSM real-time feasibility (GT detections)\nBatching + speed-oriented settings")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_ylim(0, max(max(mean_fps), max(median_fps), target_fps) * 1.35)

    for bar, b in zip(bars1, configs):
        _annotate_bar(ax, bar, f"{b.fps_overall:.1f}")
    for bar, b in zip(bars2, configs):
        _annotate_bar(ax, bar, f"{b.fps_median:.1f}")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_trackssm_latency(configs: list[Bench], out: Path) -> None:
    """Plot typical (median) and tail (p95) latency in ms."""
    _setup_style()

    labels = [b.label for b in configs]
    x = np.arange(len(labels))

    p50 = [b.total_median_ms for b in configs]
    p95 = [b.total_p95_ms for b in configs]

    width = 0.38
    fig, ax = plt.subplots(figsize=(12.6, 5.1))
    bars1 = ax.bar(
        x - width / 2,
        p50,
        width,
        label="p50 latency (median)",
        color="#4C78A8",
        alpha=0.9,
        edgecolor="#222",
        linewidth=0.9,
    )
    bars2 = ax.bar(
        x + width / 2,
        p95,
        width,
        label="p95 latency (tail)",
        color="#F58518",
        alpha=0.9,
        edgecolor="#222",
        linewidth=0.9,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency distribution (per-frame)\nMedian vs tail latency")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_ylim(0, max(p95) * 1.25)

    for bar, v in zip(bars1, p50):
        _annotate_bar(ax, bar, f"{v:.1f}")
    for bar, v in zip(bars2, p95):
        _annotate_bar(ax, bar, f"{v:.0f}")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_io_vs_compute(default_compute: Bench, default_io: Bench, rt_compute: Bench, rt_io: Bench, out: Path) -> None:
    """Compare compute-only vs include-IO FPS (median), for two configs."""
    _setup_style()

    labels = ["Default", "Real-time"]
    compute = [default_compute.fps_median, rt_compute.fps_median]
    with_io = [default_io.fps_median, rt_io.fps_median]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11.6, 4.8))
    bars1 = ax.bar(
        x - width / 2,
        compute,
        width,
        label="Compute-only (det + track)",
        color="#1f77b4",
        alpha=0.9,
        edgecolor="#222",
        linewidth=0.9,
    )
    bars2 = ax.bar(
        x + width / 2,
        with_io,
        width,
        label="End-to-end (incl. read/decode)",
        color="#ff7f0e",
        alpha=0.9,
        edgecolor="#222",
        linewidth=0.9,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Median FPS")
    ax.set_title("Compute vs end-to-end runtime (why FPS numbers differ)")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_ylim(0, max(max(compute), max(with_io)) * 1.25)

    for bar, v in zip(bars1, compute):
        _annotate_bar(ax, bar, f"{v:.1f}")
    for bar, v in zip(bars2, with_io):
        _annotate_bar(ax, bar, f"{v:.1f}")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _copy_if_requested(src: Path, thesis_dir: Path | None) -> None:
    if thesis_dir is None:
        return
    thesis_dir.mkdir(parents=True, exist_ok=True)
    (thesis_dir / src.name).write_bytes(src.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/thesis/realtime_benchmark"),
        help="Root folder containing benchmark run subfolders",
    )
    parser.add_argument(
        "--thesis-output",
        type=Path,
        default=Path("../Elaborato/images"),
        help="Folder to also copy the final PNGs into (slide-ready)",
    )

    args = parser.parse_args()

    # Preferred benchmark set for presentation (dense scenes)
    p_default_compute = args.root / "trackssm_gt_default_batch_dense" / "benchmark.json"
    p_rt_compute = args.root / "trackssm_gt_realtime_batch_dense" / "benchmark.json"
    p_rt_nobatch_compute = args.root / "trackssm_gt_realtime_no_batch_dense" / "benchmark.json"

    # Optional end-to-end (includes read/decode)
    p_default_io = args.root / "trackssm_gt_default_batch_dense_io" / "benchmark.json"
    p_rt_io = args.root / "trackssm_gt_realtime_batch_dense_io" / "benchmark.json"

    required = [p_default_compute, p_rt_compute, p_rt_nobatch_compute]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required benchmark.json files (run the benchmarks first):\n" + "\n".join(str(p) for p in missing)
        )

    default_compute = _read_benchmark(p_default_compute, "Default\n(CMC+ReID)")
    rt_compute = _read_benchmark(p_rt_compute, "Real-time\n(CMC off, ReID off)")
    rt_nobatch_compute = _read_benchmark(p_rt_nobatch_compute, "Ablation\n(no batching)")

    out_dir = args.root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    f_fps = out_dir / "realtime_feasibility_trackssm_fps.png"
    f_lat = out_dir / "realtime_feasibility_trackssm_latency.png"
    f_io = out_dir / "realtime_feasibility_io_vs_compute.png"

    plot_trackssm_fps([default_compute, rt_nobatch_compute, rt_compute], f_fps, target_fps=25.0)
    plot_trackssm_latency([default_compute, rt_nobatch_compute, rt_compute], f_lat)

    # Only generate IO-vs-compute if both IO files exist
    if p_default_io.exists() and p_rt_io.exists():
        default_io = _read_benchmark(p_default_io, "Default (IO)")
        rt_io = _read_benchmark(p_rt_io, "Real-time (IO)")
        plot_io_vs_compute(default_compute, default_io, rt_compute, rt_io, f_io)
        _copy_if_requested(f_io, args.thesis_output)
        print(f"✓ Wrote: {f_io}")
    else:
        f_io = None

    _copy_if_requested(f_fps, args.thesis_output)
    _copy_if_requested(f_lat, args.thesis_output)

    print(f"✓ Wrote: {f_fps}")
    print(f"✓ Wrote: {f_lat}")
    if args.thesis_output is not None:
        print(f"✓ Copied PNGs to: {args.thesis_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

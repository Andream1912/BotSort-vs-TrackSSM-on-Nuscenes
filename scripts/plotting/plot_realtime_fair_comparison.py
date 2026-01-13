#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_benchmark(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_ms_stats(obj: Any) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (mean_ms, median_ms, p95_ms) from either a dict or None."""
    if not isinstance(obj, dict):
        return None, None, None
    return (
        obj.get("mean_ms"),
        obj.get("median_ms"),
        obj.get("p95_ms"),
    )


def _fps_from_ms(ms: Optional[float]) -> Optional[float]:
    if ms is None or ms <= 0:
        return None
    return 1000.0 / ms


def plot_fair_comparison(trackssm_json: Path, botsort_json: Path, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)

    trackssm = _load_benchmark(trackssm_json)
    botsort = _load_benchmark(botsort_json)

    labels = ["TrackSSM", "Kalman (BoT-SORT)"]
    data = [trackssm, botsort]

    fps_overall = [d.get("fps_overall") for d in data]
    total_mean_ms = [_get_ms_stats(d.get("total_ms"))[0] for d in data]
    total_median_ms = [_get_ms_stats(d.get("total_ms"))[1] for d in data]
    fps_median = [_fps_from_ms(ms) for ms in total_median_ms]

    detect_median_ms = [_get_ms_stats(d.get("detect_ms"))[1] for d in data]
    track_median_ms = [_get_ms_stats(d.get("track_ms"))[1] for d in data]

    # Sanity: ensure apples-to-apples by checking detector settings match.
    # If not, we still plot but add a note in the title.
    warn = ""
    for key in ["include_io", "sync_cuda", "warmup_frames_per_scene", "max_frames_per_scene"]:
        if trackssm.get(key) != botsort.get(key):
            warn = " (settings differ!)"
            break

    fig = plt.figure(figsize=(12, 5), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25])

    # Panel A: FPS comparison (mean-based vs median-based)
    ax0 = fig.add_subplot(gs[0, 0])
    x = [0, 1]
    w = 0.35
    ax0.bar([xi - w / 2 for xi in x], fps_overall, width=w, label="FPS (overall, mean)")
    ax0.bar([xi + w / 2 for xi in x], fps_median, width=w, label="FPS (typical, median)")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=12, ha="right")
    ax0.set_ylabel("FPS")
    ax0.set_title("Real-time throughput" + warn)
    ax0.grid(True, axis="y", alpha=0.3)
    ax0.legend(fontsize=8)

    for xi, f_mean, f_med in zip(x, fps_overall, fps_median):
        if f_mean is not None:
            ax0.text(xi - w / 2, f_mean + 0.6, f"{f_mean:.1f}", ha="center", va="bottom", fontsize=8)
        if f_med is not None:
            ax0.text(xi + w / 2, f_med + 0.6, f"{f_med:.1f}", ha="center", va="bottom", fontsize=8)

    # Panel B: median breakdown (detect + track)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(x, detect_median_ms, label="detect (median ms)")
    ax1.bar(x, track_median_ms, bottom=detect_median_ms, label="track (median ms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=12, ha="right")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Median latency breakdown")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(fontsize=8)

    for xi, det_ms, trk_ms, tot_ms in zip(x, detect_median_ms, track_median_ms, total_median_ms):
        if det_ms is not None:
            ax1.text(xi, det_ms / 2, f"{det_ms:.1f}", ha="center", va="center", fontsize=8, color="white")
        if trk_ms is not None and det_ms is not None:
            ax1.text(xi, det_ms + trk_ms / 2, f"{trk_ms:.1f}", ha="center", va="center", fontsize=8, color="white")
        if tot_ms is not None:
            ax1.text(xi, (det_ms or 0) + (trk_ms or 0) + 1.0, f"total {tot_ms:.1f}ms", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Fair real-time comparison (same detector, same scenes)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = outdir / "realtime_fair_comparison.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackssm", required=True, help="Path to TrackSSM benchmark.json")
    ap.add_argument("--botsort", required=True, help="Path to BoT-SORT benchmark.json")
    ap.add_argument("--outdir", default="results/thesis/realtime_benchmark/plots", help="Output directory")
    ap.add_argument("--thesis-output", default=None, help="Optional folder to copy the PNG into (e.g. ../Elaborato/images)")
    args = ap.parse_args()

    out_path = plot_fair_comparison(Path(args.trackssm), Path(args.botsort), Path(args.outdir))
    print(f"✓ Wrote: {out_path}")

    if args.thesis_output:
        dst_dir = Path(args.thesis_output)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / out_path.name
        dst.write_bytes(out_path.read_bytes())
        print(f"✓ Copied: {dst}")


if __name__ == "__main__":
    main()

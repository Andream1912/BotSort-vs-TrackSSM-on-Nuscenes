#!/usr/bin/env python3
import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class EpochCurves:
    epochs: List[int]
    train_loss: List[float]
    val_loss: List[float]


def parse_epoch_curves_from_log(log_path: Path) -> EpochCurves:
    epoch_re = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)")
    summary_re = re.compile(
        r"Train Loss:\s*([0-9]*\.?[0-9]+)\s*\|\s*Val Loss:\s*([0-9]*\.?[0-9]+)"
    )

    current_epoch: Optional[int] = None
    per_epoch: Dict[int, Tuple[float, float]] = {}

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_epoch = epoch_re.search(line)
            if m_epoch:
                try:
                    current_epoch = int(m_epoch.group(1))
                except Exception:
                    current_epoch = None

            m_sum = summary_re.search(line)
            if m_sum and current_epoch is not None:
                try:
                    tr = float(m_sum.group(1))
                    va = float(m_sum.group(2))
                except Exception:
                    continue
                per_epoch[current_epoch] = (tr, va)

    if not per_epoch:
        raise RuntimeError(
            f"Nessuna riga 'Train Loss: ... | Val Loss: ...' trovata in {log_path}"
        )

    epochs = sorted(per_epoch.keys())
    train_loss = [per_epoch[e][0] for e in epochs]
    val_loss = [per_epoch[e][1] for e in epochs]
    return EpochCurves(epochs=epochs, train_loss=train_loss, val_loss=val_loss)


def write_curves_csv(curves: EpochCurves, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss"])
        for e, tr, va in zip(curves.epochs, curves.train_loss, curves.val_loss):
            w.writerow([e, tr, va])


def write_curves_json(curves: EpochCurves, out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epochs": curves.epochs,
        "train_loss": curves.train_loss,
        "val_loss": curves.val_loss,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_tensorboard_scalars(event_file: Path) -> Dict[str, List[Dict[str, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    ea.Reload()
    out: Dict[str, List[Dict[str, float]]] = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        out[tag] = [
            {"step": float(e.step), "wall_time": float(e.wall_time), "value": float(e.value)}
            for e in events
        ]
    return out


def find_single_event_file(run_dir: Path) -> Path:
    files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(f"Nessun file events.out.tfevents.* in {run_dir}")
    if len(files) > 1:
        # scegli il più grande (di solito contiene tutto)
        files = sorted(files, key=lambda p: p.stat().st_size, reverse=True)
    return files[0]


def plot_train_val_loss(curves: EpochCurves, title: str, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.5, 4.8), dpi=150)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(curves.epochs, curves.train_loss, label="Train loss", linewidth=2)
    ax.plot(curves.epochs, curves.val_loss, label="Val loss", linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_lr_phase1(tb: Dict[str, List[Dict[str, float]]], num_epochs: int, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lr_events = tb.get("Train/LR", [])
    if not lr_events:
        return

    steps = [e["step"] for e in lr_events]
    lrs = [e["value"] for e in lr_events]

    max_step = max(steps) if steps else 0.0
    steps_per_epoch = max_step / max(1, num_epochs)
    x_epoch = [s / steps_per_epoch if steps_per_epoch > 0 else 0.0 for s in steps]

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.5, 4.2), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_epoch, lrs, label="LR", linewidth=1.5)
    ax.set_title("Phase 1 (decoder) — Learning rate")
    ax.set_xlabel("Epoch (approx)")
    ax.set_ylabel("LR")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_lr_phase2(tb: Dict[str, List[Dict[str, float]]], num_epochs: int, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    enc = tb.get("Train/LR_Encoder", [])
    dec = tb.get("Train/LR_Decoder", [])
    if not enc and not dec:
        return

    # usa lo step dell'encoder come riferimento
    ref = enc if enc else dec
    steps = [e["step"] for e in ref]
    max_step = max(steps) if steps else 0.0
    steps_per_epoch = max_step / max(1, num_epochs)

    def to_epoch(events: List[Dict[str, float]]) -> Tuple[List[float], List[float]]:
        xs = [e["step"] / steps_per_epoch if steps_per_epoch > 0 else 0.0 for e in events]
        ys = [e["value"] for e in events]
        return xs, ys

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.5, 4.2), dpi=150)
    ax = fig.add_subplot(1, 1, 1)

    if enc:
        x, y = to_epoch(enc)
        ax.plot(x, y, label="LR encoder", linewidth=1.5)
    if dec:
        x, y = to_epoch(dec)
        ax.plot(x, y, label="LR decoder", linewidth=1.5)

    ax.set_title("Phase 2 (full) — Learning rates")
    ax.set_xlabel("Epoch (approx)")
    ax.set_ylabel("LR")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract and plot TrackSSM fine-tuning curves (from logs + TensorBoard).")
    ap.add_argument(
        "--phase1-log",
        type=Path,
        default=Path("/user/amarino/tesi_project_ARCHIVE/old_logs/phase1_training.log"),
    )
    ap.add_argument(
        "--phase2-log",
        type=Path,
        default=Path("/user/amarino/tesi_project_ARCHIVE/old_logs/phase2_full_training.log"),
    )
    ap.add_argument(
        "--phase1-run",
        type=Path,
        default=Path("/user/amarino/tesi_project_ARCHIVE/old_logs/trackssm_runs/phase1_decoder_20251119_153154"),
    )
    ap.add_argument(
        "--phase2-run",
        type=Path,
        default=Path("/user/amarino/tesi_project_ARCHIVE/old_logs/trackssm_runs/phase2_full_20251120_153846"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/user/amarino/tesi_project_amarino/thesis_outputs_trackssm"),
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: parse epoch curves
    phase1_curves = parse_epoch_curves_from_log(args.phase1_log)
    write_curves_csv(phase1_curves, args.out_dir / "phase1_decoder_train_val_loss.csv")
    write_curves_json(phase1_curves, args.out_dir / "phase1_decoder_train_val_loss.json")
    plot_train_val_loss(
        phase1_curves,
        title="TrackSSM fine-tuning — Phase 1 (decoder)",
        out_png=args.out_dir / "phase1_decoder_train_val_loss.png",
    )

    # Phase 2: parse epoch curves
    phase2_curves = parse_epoch_curves_from_log(args.phase2_log)
    write_curves_csv(phase2_curves, args.out_dir / "phase2_full_train_val_loss.csv")
    write_curves_json(phase2_curves, args.out_dir / "phase2_full_train_val_loss.json")
    plot_train_val_loss(
        phase2_curves,
        title="TrackSSM fine-tuning — Phase 2 (full)",
        out_png=args.out_dir / "phase2_full_train_val_loss.png",
    )

    # TensorBoard scalars (LR)
    phase1_event = find_single_event_file(args.phase1_run)
    phase2_event = find_single_event_file(args.phase2_run)

    tb1 = read_tensorboard_scalars(phase1_event)
    (args.out_dir / "phase1_decoder_tensorboard_scalars.json").write_text(
        json.dumps(tb1, indent=2), encoding="utf-8"
    )
    plot_lr_phase1(tb1, num_epochs=len(phase1_curves.epochs), out_png=args.out_dir / "phase1_decoder_lr.png")

    tb2 = read_tensorboard_scalars(phase2_event)
    (args.out_dir / "phase2_full_tensorboard_scalars.json").write_text(
        json.dumps(tb2, indent=2), encoding="utf-8"
    )
    plot_lr_phase2(tb2, num_epochs=len(phase2_curves.epochs), out_png=args.out_dir / "phase2_full_lr_encoder_decoder.png")

    print("OK")
    print("Outputs:")
    for p in sorted(args.out_dir.glob("*.png")):
        print(" -", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

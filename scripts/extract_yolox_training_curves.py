#!/usr/bin/env python3
"""Extract YOLOX fine-tuning training/validation curves from logs.

This script is intentionally log-driven (no TensorBoard dependency) so the
produced plots remain reproducible from archived log files.

Expected sources (as in this repo):
- Training logs containing lines like:
  "epoch: 20/30, iter: 10/2667, ... total_loss: 4.6, iou_loss: 1.9, ... lr: 1.156e-05"
- Validation-loss logs containing summary lines like:
  "✅ Epoch 10: Validation Loss = 53.82 (iou: 0.00, conf: 53.82, cls: 0.00)"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class YoloXTrainPoint:
    epoch: int
    total_loss: float
    iou_loss: float
    conf_loss: float
    cls_loss: float
    lr: float


@dataclass(frozen=True)
class YoloXValPoint:
    epoch: int
    val_total: float
    val_iou: float
    val_conf: float
    val_cls: float


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def parse_train_points(log_paths: List[Path]) -> List[YoloXTrainPoint]:
    # Example:
    # ... - epoch: 20/30, iter: 10/2667, ..., total_loss: 4.6, iou_loss: 1.9, ..., conf_loss: 2.0, cls_loss: 0.8, lr: 1.156e-05, ...
    re_line = re.compile(
        r"epoch:\s*(\d+)\/\d+.*?total_loss:\s*([0-9]*\.?[0-9]+)\s*,\s*"
        r"iou_loss:\s*([0-9]*\.?[0-9]+)\s*,\s*"
        r"l1_loss:\s*([0-9]*\.?[0-9]+)\s*,\s*"
        r"conf_loss:\s*([0-9]*\.?[0-9]+)\s*,\s*"
        r"cls_loss:\s*([0-9]*\.?[0-9]+).*?lr:\s*([0-9eE+\-\.]+)"
    )

    points: List[YoloXTrainPoint] = []
    for p in log_paths:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = re_line.search(line)
                if not m:
                    continue
                epoch = int(m.group(1))
                total = float(m.group(2))
                iou = float(m.group(3))
                # l1 is m.group(4) but we don't store it
                conf = float(m.group(5))
                cls = float(m.group(6))
                lr = float(m.group(7))
                points.append(
                    YoloXTrainPoint(
                        epoch=epoch,
                        total_loss=total,
                        iou_loss=iou,
                        conf_loss=conf,
                        cls_loss=cls,
                        lr=lr,
                    )
                )

    if not points:
        raise RuntimeError("Nessuna riga di training parsabile trovata nei log YOLOX forniti.")

    points.sort(key=lambda x: (x.epoch))
    return points


def aggregate_train_by_epoch(points: List[YoloXTrainPoint]) -> Dict[int, Dict[str, float]]:
    per_epoch: Dict[int, Dict[str, List[float]]] = {}
    for pt in points:
        bucket = per_epoch.setdefault(
            pt.epoch,
            {
                "total_loss": [],
                "iou_loss": [],
                "conf_loss": [],
                "cls_loss": [],
                "lr": [],
            },
        )
        bucket["total_loss"].append(pt.total_loss)
        bucket["iou_loss"].append(pt.iou_loss)
        bucket["conf_loss"].append(pt.conf_loss)
        bucket["cls_loss"].append(pt.cls_loss)
        bucket["lr"].append(pt.lr)

    out: Dict[int, Dict[str, float]] = {}
    for epoch, vals in per_epoch.items():
        out[epoch] = {k: _mean(v) for k, v in vals.items()}
    return out


def parse_val_points(val_log_path: Path) -> List[YoloXValPoint]:
    # Example:
    # ✅ Epoch 10: Validation Loss = 53.82 (iou: 0.00, conf: 53.82, cls: 0.00)
    re_line = re.compile(
        r"Epoch\s+(\d+):\s+Validation\s+Loss\s*=\s*([0-9]*\.?[0-9]+)\s*\(iou:\s*"
        r"([0-9]*\.?[0-9]+)\s*,\s*conf:\s*([0-9]*\.?[0-9]+)\s*,\s*cls:\s*([0-9]*\.?[0-9]+)\s*\)"
    )

    points: List[YoloXValPoint] = []
    with val_log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re_line.search(line)
            if not m:
                continue
            points.append(
                YoloXValPoint(
                    epoch=int(m.group(1)),
                    val_total=float(m.group(2)),
                    val_iou=float(m.group(3)),
                    val_conf=float(m.group(4)),
                    val_cls=float(m.group(5)),
                )
            )

    if not points:
        raise RuntimeError(f"Nessuna riga 'Validation Loss' trovata in {val_log_path}")

    points.sort(key=lambda x: x.epoch)
    return points


def write_csv(path: Path, header: List[str], rows: Iterable[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def plot_yolox_curves(
    train_epoch: Dict[int, Dict[str, float]],
    val_points: Optional[List[YoloXValPoint]],
    out_png: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = sorted(train_epoch.keys())
    train_total = [train_epoch[e]["total_loss"] for e in epochs]

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.5, 4.8), dpi=150)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(epochs, train_total, label="Train total loss (avg)", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.grid(True, alpha=0.3)

    if val_points:
        # Put validation on a secondary axis to avoid misleading scale mixing.
        ax2 = ax.twinx()
        ve = [p.epoch for p in val_points]
        vt = [p.val_total for p in val_points]
        ax2.plot(ve, vt, label="Val loss (normalized)", linewidth=2, linestyle="--")
        ax2.set_ylabel("Validation loss")

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper right")
    else:
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract and plot YOLOX fine-tuning curves from logs.")
    ap.add_argument(
        "--train-log",
        type=Path,
        action="append",
        required=True,
        help="Path to a YOLOX training log (repeatable).",
    )
    ap.add_argument(
        "--val-log",
        type=Path,
        default=None,
        help="Path to a YOLOX validation-loss log (e.g., compute_validation_losses_NORMALIZED.log).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/user/amarino/tesi_project_amarino/thesis_outputs_detector"),
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_points = parse_train_points(args.train_log)
    train_epoch = aggregate_train_by_epoch(train_points)

    val_points: Optional[List[YoloXValPoint]] = None
    if args.val_log is not None:
        val_points = parse_val_points(args.val_log)

    # Persist extracted data
    epochs = sorted(train_epoch.keys())
    write_csv(
        args.out_dir / "yolox_train_loss_by_epoch.csv",
        ["epoch", "total_loss", "iou_loss", "conf_loss", "cls_loss", "lr"],
        [[e, train_epoch[e]["total_loss"], train_epoch[e]["iou_loss"], train_epoch[e]["conf_loss"], train_epoch[e]["cls_loss"], train_epoch[e]["lr"]] for e in epochs],
    )

    payload: Dict[str, object] = {"train": {str(e): train_epoch[e] for e in epochs}}
    if val_points is not None:
        payload["val"] = [p.__dict__ for p in val_points]
    (args.out_dir / "yolox_curves.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    plot_yolox_curves(
        train_epoch=train_epoch,
        val_points=val_points,
        out_png=args.out_dir / "yolox_train_val_loss.png",
        title="YOLOX-L fine-tuning — training/validation loss",
    )

    print("OK")
    print("Outputs:")
    for p in sorted(args.out_dir.glob("*.png")):
        print(" -", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

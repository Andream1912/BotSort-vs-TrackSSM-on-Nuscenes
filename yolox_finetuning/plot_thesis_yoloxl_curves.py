#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


TRAIN_LINE_RE = re.compile(
    r"epoch:\s*(?P<epoch>\d+)/(?P<epoch_total>\d+),\s*"
    r"iter:\s*(?P<iter>\d+)/(?P<iter_total>\d+),.*?"
    r"total_loss:\s*(?P<total_loss>[0-9]*\.?[0-9]+)"
)

VAL_LINE_RE = re.compile(
    r"✅\s*Epoch\s*(?P<epoch>\d+):\s*Validation\s*Loss\s*=\s*(?P<val_loss>[0-9]*\.?[0-9]+)"
)


def parse_train_log(train_log_path: Path) -> dict[int, list[float]]:
    per_epoch: dict[int, list[float]] = {}
    with train_log_path.open("r", errors="ignore") as f:
        for line in f:
            m = TRAIN_LINE_RE.search(line)
            if not m:
                continue
            epoch = int(m.group("epoch"))
            total_loss = float(m.group("total_loss"))
            per_epoch.setdefault(epoch, []).append(total_loss)
    return per_epoch


def parse_val_log(val_log_path: Path) -> dict[int, float]:
    per_epoch: dict[int, float] = {}
    with val_log_path.open("r", errors="ignore") as f:
        for line in f:
            m = VAL_LINE_RE.search(line)
            if not m:
                continue
            epoch = int(m.group("epoch"))
            val_loss = float(m.group("val_loss"))
            per_epoch[epoch] = val_loss
    return per_epoch


def parse_val_json(val_json_path: Path) -> dict[int, float]:
    data = json.loads(val_json_path.read_text())
    epochs = data.get("epochs", [])
    losses = data.get("val_loss", [])
    per_epoch: dict[int, float] = {}
    for e, v in zip(epochs, losses):
        try:
            per_epoch[int(e)] = float(v)
        except Exception:
            continue
    return per_epoch


def normalize(values_by_epoch: dict[int, float], reference_epoch: int) -> dict[int, float]:
    if reference_epoch not in values_by_epoch:
        return values_by_epoch
    ref = values_by_epoch[reference_epoch]
    if ref == 0:
        return values_by_epoch
    return {e: v / ref for e, v in values_by_epoch.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a clean, thesis-ready YOLOX-L fine-tuning loss curve (per-epoch)."
    )
    parser.add_argument(
        "--train-log",
        type=Path,
        default=Path(
            "external/YOLOX/yolox_finetuning/yolox_l_nuscenes_stable/train_log.txt"
        ),
    )
    parser.add_argument(
        "--val-log",
        type=Path,
        default=Path("yolox_finetuning/logs/compute_validation_losses_NORMALIZED.log"),
        help="Fallback validation-loss log (used only if --val-json is missing).",
    )
    parser.add_argument(
        "--val-json",
        type=Path,
        default=Path("yolox_finetuning/validation_losses_all_epochs_REAL.json"),
        help="Preferred validation-loss source (per-epoch total loss from checkpoints).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("yolox_finetuning/yoloxl_finetuning_training_loss_curves.png"),
    )
    args = parser.parse_args()

    if not args.train_log.exists():
        raise FileNotFoundError(f"Train log not found: {args.train_log}")
    if not args.val_json.exists() and not args.val_log.exists():
        raise FileNotFoundError(
            f"Neither val JSON nor val log found: {args.val_json} / {args.val_log}"
        )

    per_epoch_train_samples = parse_train_log(args.train_log)
    train_mean = {e: sum(v) / len(v) for e, v in per_epoch_train_samples.items() if v}

    if args.val_json.exists():
        val_loss = parse_val_json(args.val_json)
    else:
        val_loss = parse_val_log(args.val_log)

    if not train_mean:
        raise RuntimeError(f"No training lines parsed from {args.train_log}")

    max_epoch = max(train_mean)
    epochs = list(range(1, max_epoch + 1))

    # Normalize each curve by its epoch-1 value (when available) to make trends comparable.
    train_mean = normalize(train_mean, reference_epoch=1)
    if 1 in val_loss:
        val_loss = normalize(val_loss, reference_epoch=1)

    train_y = [train_mean.get(e) for e in epochs]
    val_y = [val_loss.get(e) for e in epochs]

    plt.figure(figsize=(8.5, 4.0))
    plt.plot(epochs, train_y, label="Train loss (mean/epoch, normalized)", linewidth=2)
    plt.plot(epochs, val_y, label="Val loss (normalized)", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Normalized loss")
    plt.xlim(1, max_epoch)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()

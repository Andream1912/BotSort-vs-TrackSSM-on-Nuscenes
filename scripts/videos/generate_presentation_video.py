#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _find_font_file() -> str | None:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    return None


def _pick_scenes(data_root: Path, requested: list[str] | None, n_default: int) -> list[str]:
    val_dir = data_root / "val"
    if requested:
        return requested

    if not val_dir.exists():
        raise FileNotFoundError(f"Dataset val directory not found: {val_dir}")

    all_scenes = sorted([p.name for p in val_dir.iterdir() if p.is_dir() and p.name.startswith("scene-")])
    if not all_scenes:
        raise RuntimeError(f"No scene folders found under: {val_dir}")

    curated = [
        "scene-0014_CAM_FRONT",
        "scene-0036_CAM_FRONT",
        "scene-0093_CAM_FRONT",
        "scene-0520_CAM_FRONT",
    ]
    scenes = [s for s in curated if (val_dir / s).exists()]
    if len(scenes) >= n_default:
        return scenes[:n_default]

    for s in all_scenes:
        if s not in scenes:
            scenes.append(s)
        if len(scenes) >= n_default:
            break
    return scenes


def _load_best_config(best_config_path: Path) -> dict:
    data = json.loads(best_config_path.read_text())
    if "config" not in data:
        raise ValueError(f"best_config.json missing 'config': {best_config_path}")
    return data["config"]


def _ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def run_tracking(
    tracker: str,
    scenes: list[str],
    out_dir: Path,
    data_root: Path,
    detector_weights: Path,
    config: dict,
    fps: int,
    max_frames: int,
    cmc_method: str,
    reid: bool,
    trackssm_batch: bool,
    device: str,
) -> None:
    track_py = REPO_ROOT / "track.py"

    cmd = [
        "python3",
        str(track_py),
        "--tracker",
        tracker,
        "--data",
        str(data_root / "val"),
        "--gt-data",
        str(data_root / "val"),
        "--output",
        str(out_dir),
        "--device",
        device,
        "--detector-weights",
        str(detector_weights),
        "--conf-thresh",
        str(config.get("conf_thresh", 0.3)),
        "--nms-thresh",
        str(config.get("nms_thresh", 0.5)),
        "--track-thresh",
        str(config.get("track_thresh", 0.6)),
        "--match-thresh",
        str(config.get("match_thresh", 0.8)),
        "--cmc-method",
        cmc_method,
        "--save-videos",
        "--video-fps",
        str(fps),
        "--benchmark",
        "--benchmark-warmup",
        "0",
        "--benchmark-max-frames",
        str(max_frames),
        "--benchmark-sync-cuda",
        "--no-save-results",
        "--scenes",
        ",".join(scenes),
    ]

    if reid:
        cmd.append("--with-reid")
    else:
        cmd.append("--no-reid")

    if tracker == "trackssm":
        cmd.append("--trackssm-batch" if trackssm_batch else "--trackssm-no-batch")

    _run(cmd)


def _scene_video_path(run_dir: Path, scene: str) -> Path:
    return run_dir / "videos" / f"{scene}.mp4"


def make_side_by_side(
    left_path: Path,
    right_path: Path,
    out_path: Path,
    left_label: str,
    right_label: str,
    font_file: str | None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    draw_left = (
        f"drawtext=text='{left_label}':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.45:boxborderw=8"
    )
    draw_right = (
        f"drawtext=text='{right_label}':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.45:boxborderw=8"
    )
    if font_file:
        draw_left = f"drawtext=fontfile={font_file}:" + draw_left[len("drawtext:") :]
        draw_right = f"drawtext=fontfile={font_file}:" + draw_right[len("drawtext:") :]

    filter_complex = (
        f"[0:v]scale=-2:720,{draw_left}[v0];"
        f"[1:v]scale=-2:720,{draw_right}[v1];"
        f"[v0][v1]hstack=inputs=2[v]"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(left_path),
        "-i",
        str(right_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    _run(cmd)


def concat_videos(clips: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    list_file = out_path.parent / "concat_list.txt"
    lines = [f"file '{clip.resolve()}'" for clip in clips]
    list_file.write_text("\n".join(lines) + "\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    _run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a long presentation video: TrackSSM vs BoT-SORT (side-by-side)")
    parser.add_argument("--data-root", default="data/nuscenes_mot_front", help="Dataset root containing val/")
    parser.add_argument("--best-config", default="results/GRID_SEARCH/best_config.json", help="Path to best_config.json")
    parser.add_argument("--detector-weights", default="yolox_finetuning/yolox_l_nuscenes_stable/epoch_30.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cmc-method", default="sparseOptFlow", choices=["sparseOptFlow", "orb", "sift", "ecc", "none"])
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=300, help="Max frames per scene (controls clip length)")
    parser.add_argument("--n-scenes", type=int, default=3, help="How many scenes to include if --scenes not provided")
    parser.add_argument("--scenes", default="", help="Comma-separated scenes; if empty chooses defaults")
    parser.add_argument("--skip-tracking", action="store_true", help="Only compose side-by-side/concat assuming videos already exist")
    parser.add_argument("--with-reid", action="store_true", help="Enable ReID for BOTH trackers (default: off)")
    parser.add_argument("--trackssm-batch", action="store_true", help="Enable TrackSSM batching (default: on)")
    parser.add_argument("--trackssm-no-batch", action="store_true", help="Disable TrackSSM batching")
    parser.add_argument("--workdir", default="results/PRESENTATION_VIDEO", help="Working directory under repo")
    parser.add_argument("--out", default=str((Path("..") / "Elaborato" / "videos" / "trackssm_vs_botsort_presentation.mp4")), help="Final output mp4")

    args = parser.parse_args()

    data_root = (REPO_ROOT / args.data_root).resolve()
    best_config_path = (REPO_ROOT / args.best_config).resolve()
    detector_weights = (REPO_ROOT / args.detector_weights).resolve()
    workdir = (REPO_ROOT / args.workdir).resolve()
    out_final = (REPO_ROOT / args.out).resolve()

    requested_scenes = [s.strip() for s in args.scenes.split(",") if s.strip()] or None
    scenes = _pick_scenes(data_root, requested_scenes, args.n_scenes)

    _ensure_exists(best_config_path, "best-config")
    _ensure_exists(detector_weights, "detector-weights")

    cfg = _load_best_config(best_config_path)

    trackssm_batch = True
    if args.trackssm_no_batch:
        trackssm_batch = False
    elif args.trackssm_batch:
        trackssm_batch = True

    print("\n=== Presentation video generation ===")
    print("Scenes:", scenes)
    print("Best config:", cfg)
    print("Detector weights:", detector_weights)
    print("Workdir:", workdir)
    print("Final output:", out_final)

    trackssm_dir = workdir / "runs" / "trackssm"
    botsort_dir = workdir / "runs" / "botsort"

    if not args.skip_tracking:
        missing_trackssm = [s for s in scenes if not _scene_video_path(trackssm_dir, s).exists()]
        if missing_trackssm:
            print(f"\nTrackSSM: generating {len(missing_trackssm)}/{len(scenes)} missing scene videos...")
            run_tracking(
                tracker="trackssm",
                scenes=missing_trackssm,
                out_dir=trackssm_dir,
                data_root=data_root,
                detector_weights=detector_weights,
                config=cfg,
                fps=args.fps,
                max_frames=args.max_frames,
                cmc_method=args.cmc_method,
                reid=args.with_reid,
                trackssm_batch=trackssm_batch,
                device=args.device,
            )
        else:
            print(f"\nTrackSSM: all {len(scenes)} scene videos already exist; skipping tracking.")

        missing_botsort = [s for s in scenes if not _scene_video_path(botsort_dir, s).exists()]
        if missing_botsort:
            print(f"\nBoT-SORT: generating {len(missing_botsort)}/{len(scenes)} missing scene videos...")
            run_tracking(
                tracker="botsort",
                scenes=missing_botsort,
                out_dir=botsort_dir,
                data_root=data_root,
                detector_weights=detector_weights,
                config=cfg,
                fps=args.fps,
                max_frames=args.max_frames,
                cmc_method=args.cmc_method,
                reid=args.with_reid,
                trackssm_batch=trackssm_batch,
                device=args.device,
            )
        else:
            print(f"\nBoT-SORT: all {len(scenes)} scene videos already exist; skipping tracking.")

        

    font_file = _find_font_file()
    if font_file:
        print("Using font:", font_file)

    side_dir = workdir / "side_by_side"
    clips: list[Path] = []
    for scene in scenes:
        left = _scene_video_path(trackssm_dir, scene)
        right = _scene_video_path(botsort_dir, scene)
        _ensure_exists(left, f"TrackSSM video for {scene}")
        _ensure_exists(right, f"BoT-SORT video for {scene}")

        out_clip = side_dir / f"{scene}.mp4"
        make_side_by_side(
            left_path=left,
            right_path=right,
            out_path=out_clip,
            left_label="TrackSSM",
            right_label="BoT-SORT",
            font_file=font_file,
        )
        clips.append(out_clip)

    out_concat = workdir / "trackssm_vs_botsort_concat.mp4"
    concat_videos(clips, out_concat)

    out_final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_concat, out_final)

    print("\nDONE")
    print("Final video:", out_final)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Unified Tracking Script - Run any tracker with YOLOX detector

Usage:
    python track.py --tracker trackssm --data data/nuscenes_mot_front/val --output results/trackssm
    python track.py --tracker botsort --data data/nuscenes_mot_front/val --output results/botsort
"""

import os
import sys

# Set Triton environment variables for performance (before any imports)
os.environ['TRITON_CACHE_DIR'] = os.path.expanduser('~/.triton/cache')
os.environ['TRITON_INTERPRET'] = '1'  # Use interpreter - avoid slow JIT autotuning
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

import argparse
import cv2
import torch
import numpy as np
import json
import time
from pathlib import Path
from tqdm import tqdm

# Performance knobs (safe defaults for modern NVIDIA GPUs)
try:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
except Exception:
    pass

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'external', 'YOLOX'))

from src.detectors.yolox_detector import YOLOXDetector
from src.trackers.tracker_factory import TrackerFactory
from src.evaluation.mot_evaluator import NuScenesMultiClassEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Unified MOT tracking')
    
    # Tracker selection
    parser.add_argument('--tracker', type=str, required=True,
                       choices=['trackssm', 'botsort'],
                       help='Tracker to use')
    
    # Data paths
    parser.add_argument('--data', type=str, default= 'data/nuscenes_mot_front/val',
                       help='Path to dataset with images (e.g., data/nuscenes_mot_front/val)')
    parser.add_argument('--gt-data', type=str, default=None,
                       help='Path to GT dataset for evaluation (default: data/nuscenes_mot_front/val)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for tracking results')
    
    # Detector config
    parser.add_argument('--use-gt-det', action='store_true',
                       help='Use ground truth detections instead of YOLOX (oracle mode)')
    parser.add_argument('--detector-weights', type=str,
                       default='weights/detectors/yolox_x.pth',
                       help='Path to YOLOX weights (default: YOLOX-X COCO weights)')
    parser.add_argument('--conf-thresh', type=float, default=0.1,
                       help='Detection confidence threshold (default: 0.1)')
    parser.add_argument('--nms-thresh', type=float, default=0.65,
                       help='NMS threshold (default: 0.65)')
    parser.add_argument('--detector-test-size', type=int, nargs=2, default=None,
                       metavar=('H', 'W'),
                       help='Override YOLOX test size as two ints: H W (e.g., 640 1152). If not set, uses the default policy based on checkpoint type.')
    parser.add_argument('--detector-fp16', action='store_true',
                       help='Enable FP16 autocast for YOLOX inference on CUDA (faster, may slightly change outputs)')
    
    # Tracker config
    parser.add_argument('--track-thresh', type=float, default=0.6,
                       help='Track confidence threshold (default: 0.6)')
    parser.add_argument('--match-thresh', type=float, default=0.8,
                       help='Matching IoU threshold (default: 0.8)')
    parser.add_argument('--max-age', type=int, default=30,
                       help='Maximum frames to keep lost track (default: 30)')
    parser.add_argument('--min-hits', type=int, default=3,
                       help='Minimum hits before track is activated (default: 3)')
    parser.add_argument('--cmc-method', type=str, default='sparseOptFlow',
                       choices=['sparseOptFlow', 'orb', 'sift', 'ecc', 'file', 'files', 'none'],
                       help='Camera motion compensation method for BoT-SORT (default: sparseOptFlow). Use none for faster runtime.')
    parser.add_argument('--trackssm-checkpoint', type=str,
                       default='weights/trackssm/phase2/phase2_full_best.pth',
                       help='TrackSSM checkpoint path (default: Phase2 NuScenes fine-tuned)')
    parser.add_argument('--use-mot17-checkpoint', action='store_true',
                       help='Use MOT17 pretrained checkpoint (Phase1) instead of NuScenes fine-tuned (Phase2)')

    # TrackSSM performance options
    batch_group = parser.add_mutually_exclusive_group()
    batch_group.add_argument('--trackssm-batch', dest='trackssm_batch', action='store_true',
                             help='Enable batched TrackSSM motion prediction (faster; default)')
    batch_group.add_argument('--trackssm-no-batch', dest='trackssm_batch', action='store_false',
                             help='Disable batched TrackSSM prediction (slow; for ablation/comparison)')
    parser.set_defaults(trackssm_batch=True)

    # BoT-SORT internals
    reid_group = parser.add_mutually_exclusive_group()
    reid_group.add_argument('--with-reid', dest='with_reid', action='store_true',
                           help='Enable ReID inside BoT-SORT (default)')
    reid_group.add_argument('--no-reid', dest='with_reid', action='store_false',
                           help='Disable ReID inside BoT-SORT (useful for fast benchmarking)')
    parser.set_defaults(with_reid=True)
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    # Processing
    parser.add_argument('--scenes', type=str, default=None,
                       help='Comma-separated list of scene names to process (default: all)')
    
    # Visualization
    parser.add_argument('--save-videos', action='store_true',
                       help='Save visualization videos with tracking results')
    parser.add_argument('--video-fps', type=int, default=12,
                       help='Output video FPS (default: 12)')
    
    # Evaluation
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation after tracking (uses 7-class GT automatically)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for evaluation matching (default: 0.5)')
    parser.add_argument('--metrics-output', type=str, default=None,
                       help='Output file for metrics JSON (default: {output}/metrics.json)')

    # Benchmarking (Jetson / deployment-like testing)
    parser.add_argument('--benchmark', action='store_true',
                       help='Measure inference timing (FPS/latency) and write a benchmark.json report')
    parser.add_argument('--benchmark-warmup', type=int, default=5,
                       help='Warmup frames per scene excluded from stats (default: 5)')
    parser.add_argument('--benchmark-max-frames', type=int, default=0,
                       help='Max frames per scene for benchmarking (0 = all frames)')
    parser.add_argument('--benchmark-sync-cuda', action='store_true',
                       help='Synchronize CUDA when measuring timings (recommended for accurate GPU timings)')
    parser.add_argument('--benchmark-include-io', action='store_true',
                       help='Include image read/decode time in totals (default: off; totals will exclude IO)')
    parser.add_argument('--benchmark-output', type=str, default=None,
                       help='Benchmark JSON output file (default: {output}/benchmark.json)')
    parser.add_argument('--no-save-results', action='store_true',
                       help='Do not write MOT output files (useful for pure inference benchmarking)')
    
    return parser.parse_args()


def get_scenes(data_root, scene_list=None):
    """Get list of scenes to process"""
    data_path = Path(data_root)
    
    if scene_list:
        scenes = [s.strip() for s in scene_list.split(',')]
    else:
        # Get all scenes
        # NuScenes scenes in this project follow: scene-XXXX(_CAM_FRONT)
        # Filter out helper folders like "seqmaps" to avoid warnings and bogus eval entries.
        scenes = sorted([d.name for d in data_path.iterdir() if d.is_dir() and d.name.startswith('scene-')])
    
    return scenes


def _maybe_sync_cuda(enabled: bool, device: str):
    if not enabled:
        return
    if device is None:
        return
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()


def _ms(dt_seconds: float) -> float:
    return float(dt_seconds * 1000.0)


def process_scene(
    scene_name,
    data_root,
    detector,
    tracker,
    output_dir,
    use_gt_det=False,
    save_video=False,
    video_fps=12,
    benchmark_cfg=None,
    no_save_results=False,
    device='cuda'
):
    """Process a single scene"""
    scene_path = Path(data_root) / scene_name
    img_dir = scene_path / 'img1'
    
    # Handle NuScenes naming: if scene-XXXX doesn't exist, try scene-XXXX_CAM_FRONT
    if not scene_path.exists():
        scene_path_cam = Path(data_root) / f"{scene_name}_CAM_FRONT"
        if scene_path_cam.exists():
            scene_path = scene_path_cam
            img_dir = scene_path / 'img1'
    
    if not img_dir.exists():
        print(f"⚠️  Image directory not found: {img_dir}")
        return {'processed': False, 'timing_rows': None}
    
    # Get all images
    images = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
    
    if len(images) == 0:
        print(f"⚠️  No images found in {img_dir}")
        return {'processed': False, 'timing_rows': None}
    
    # Reset tracker for new scene
    tracker.reset()
    
    # Output file
    output_file = Path(output_dir) / f"{scene_name}.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup video writer if requested
    video_writer = None
    if save_video:
        # Read first image to get dimensions
        sample_img = cv2.imread(str(images[0]))
        if sample_img is not None:
            h, w = sample_img.shape[:2]
            video_dir = Path(output_dir) / 'videos'
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"{scene_name}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, video_fps, (w, h))
            print(f"  Saving video to: {video_path}")
    
    results = []
    timing_rows = []
    
    # Process each frame (no per-frame progress bar for cleaner output)
    max_frames = 0
    warmup = 0
    sync_cuda = False
    include_io = False
    if benchmark_cfg:
        max_frames = int(benchmark_cfg.get('max_frames', 0) or 0)
        warmup = int(benchmark_cfg.get('warmup', 0) or 0)
        sync_cuda = bool(benchmark_cfg.get('sync_cuda', False))
        include_io = bool(benchmark_cfg.get('include_io', False))

    for frame_id, img_path in enumerate(images, start=1):
        if max_frames > 0 and frame_id > max_frames:
            break

        t_frame_start = time.perf_counter()

        # Load image
        t0 = time.perf_counter()
        img = cv2.imread(str(img_path))
        t_read = time.perf_counter() - t0
        
        if img is None:
            print(f"⚠️  Failed to load image: {img_path}")
            continue
        
        # Detect objects
        t0 = time.perf_counter()
        if use_gt_det:
            detections = detector.detect(scene_name, frame_id)
        else:
            detections = detector.detect(img)
        _maybe_sync_cuda(sync_cuda, device)
        t_det = time.perf_counter() - t0
        
        # Filter invalid bboxes (width or height too small for ReID)
        # ReID requires minimum bbox size for cv2.resize (at least 5x5 pixels)
        valid_detections = []
        for det in detections:
            x, y, w, h = det['bbox']
            if w >= 5 and h >= 5:  # Min size for ReID processing
                valid_detections.append(det)
        
        # Update tracker
        t0 = time.perf_counter()
        tracks = tracker.update(valid_detections, frame=img)
        _maybe_sync_cuda(sync_cuda, device)
        t_track = time.perf_counter() - t0
        
        # Draw tracks on frame if saving video
        if video_writer is not None:
            vis_img = img.copy()
            for track in tracks:
                x, y, w, h = track['bbox']
                track_id = track['track_id']
                
                # Draw bounding box
                cv2.rectangle(vis_img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
                
                # Draw track ID
                label = f"ID:{track_id}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(vis_img, (int(x), int(y)-label_size[1]-10), 
                            (int(x)+label_size[0], int(y)), (0, 255, 0), -1)
                cv2.putText(vis_img, label, (int(x), int(y)-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            video_writer.write(vis_img)
        
        # Benchmark row (exclude warmup frames from summary later)
        if benchmark_cfg is not None:
            t_total = time.perf_counter() - t_frame_start
            timing_rows.append({
                'scene': scene_name,
                'frame': int(frame_id),
                'read_ms': _ms(t_read),
                'detect_ms': _ms(t_det),
                'track_ms': _ms(t_track),
                # total_ms is either full end-to-end (incl IO) or compute-only
                'total_ms': _ms(t_total if include_io else (t_det + t_track)),
                'n_dets': int(len(detections) if detections is not None else 0),
                'n_tracks': int(len(tracks) if tracks is not None else 0),
                'is_warmup': bool(frame_id <= warmup),
            })

        if no_save_results:
            continue

        # Save results in MOT format
        # Format: frame, track_id, x, y, w, h, conf, class_id, visibility
        for track in tracks:
            x, y, w, h = track['bbox']
            # Save with original class_id
            results.append({
                'frame': frame_id,
                'track_id': track['track_id'],
                'bbox': (x, y, w, h),
                'confidence': track['confidence'],
                'class_id': track['class_id']
            })
    
    output_file = None
    if not no_save_results:
        # Write results in MOT format with 7-class NuScenes IDs
        # Save in data/ subdirectory for evaluation
        output_file = Path(output_dir) / 'data' / f"{scene_name}.txt"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            for r in results:
                # MOT format: frame,id,x,y,w,h,conf,class,visibility,unused (10 columns)
                # class_id is native NuScenes (1=car, 2=truck, 3=bus, 4=trailer, 5=pedestrian, 6=motorcycle, 7=bicycle)
                f.write(f"{r['frame']},{r['track_id']},{r['bbox'][0]:.2f},{r['bbox'][1]:.2f},"
                       f"{r['bbox'][2]:.2f},{r['bbox'][3]:.2f},{r['confidence']:.4f},{r['class_id']},1,-1\n")
    
    # Release video writer if used
    if video_writer is not None:
        video_writer.release()
        print(f"  ✓ Video saved: {len(images)} frames")
    
    num_unique_tracks = len(set(r['track_id'] for r in results))
    if not no_save_results and output_file is not None:
        print(f"  ✓ Saved {num_unique_tracks} unique tracks ({len(results)} total detections)")
        print(f"  ✓ Results: {output_file}")

    return {
        'processed': True,
        'timing_rows': timing_rows if benchmark_cfg is not None else None,
    }



def main():
    args = parse_args()
    
    print("="*80)
    print(f"Unified MOT Tracking - {args.tracker.upper()}")
    if args.use_gt_det:
        print("Mode: GT Detections (Oracle)")
    else:
        print("Mode: YOLOX Detector")
    print("="*80)
    
    # Initialize detector
    if args.use_gt_det:
        print(f"\n[1/3] Using GT detections (oracle mode)...")
        from src.detectors.gt_detector import GTDetector
        detector = GTDetector(data_root=args.data)
    else:
        print(f"\n[1/3] Initializing YOLOX detector...")

        # Default setup policy:
        # - COCO pretrained weights (e.g., yolox_x.pth / yolox_l.pth): 80 classes + COCO->nuScenes mapping
        # - nuScenes fine-tuned weights (7 classes): native nuScenes class IDs
        weights_lower = args.detector_weights.lower()
        is_coco_pretrained = (
            "weights/detectors" in weights_lower
            or weights_lower.endswith("yolox_x.pth")
            or weights_lower.endswith("yolox_l.pth")
            or weights_lower.endswith("yolox_m.pth")
            or weights_lower.endswith("yolox_s.pth")
        )

        if is_coco_pretrained:
            detector_test_size = (1280, 1280)  # YOLOX default
            detector_num_classes = None         # auto-detect (expected: 80)
        else:
            detector_test_size = (800, 1440)    # nuScenes fine-tuning resolution used in this project
            detector_num_classes = 7

        if args.detector_test_size is not None:
            detector_test_size = (int(args.detector_test_size[0]), int(args.detector_test_size[1]))

        detector = YOLOXDetector(
            model_path=args.detector_weights,
            conf_thresh=args.conf_thresh,
            nms_thresh=args.nms_thresh,
            device=args.device,
            test_size=detector_test_size,
            num_classes=detector_num_classes,
            fp16=bool(args.detector_fp16),
        )
    
    # Initialize tracker
    print(f"\n[2/3] Initializing {args.tracker.upper()} tracker...")
    
    tracker_config = {
        'device': args.device,
        'track_thresh': args.track_thresh,
        'match_thresh': args.match_thresh,
        'img_width': 1600,
        'img_height': 900,
        'max_age': args.max_age,
        'min_hits': args.min_hits,
        'with_reid': args.with_reid,
        'cmc_method': args.cmc_method,
    }
    
    # Add TrackSSM-specific config
    if args.tracker == 'trackssm':
        # Select checkpoint: MOT17 pretrained (Phase1) or NuScenes fine-tuned (Phase2)
        if args.use_mot17_checkpoint:
            checkpoint_path = 'weights/trackssm/pretrained/MOT17_epoch160.pt'
            checkpoint_type = 'MOT17 pretrained (Phase1)'
            print(f"  Using {checkpoint_type}")
        else:
            checkpoint_path = args.trackssm_checkpoint
            checkpoint_type = 'NuScenes fine-tuned (Phase2)'
            print(f"  Using {checkpoint_type}")
        
        tracker_config['trackssm_weights'] = checkpoint_path  # Fixed: use correct key
        tracker_config['checkpoint_type'] = checkpoint_type
        tracker_config['history_len'] = 5
        tracker_config['oracle_mode'] = args.use_gt_det  # Use GT track IDs in oracle mode
        tracker_config['trackssm_batch'] = bool(args.trackssm_batch)
    
    tracker = TrackerFactory.create(args.tracker, tracker_config)
    
    # Get scenes to process
    print(f"\n[3/3] Processing scenes...")
    scenes = get_scenes(args.data, args.scenes)
    print(f"Found {len(scenes)} scenes to process")
    
    # Process each scene with overall progress bar
    processed_scenes = []
    benchmark_all = []
    for idx, scene_name in enumerate(tqdm(scenes, desc="Overall Progress", unit="scene")):
        print(f"\n[Scene {idx+1}/{len(scenes)}] Processing {scene_name}...")
        scene_out = process_scene(
            scene_name=scene_name,
            data_root=args.data,
            detector=detector,
            tracker=tracker,
            output_dir=args.output,
            use_gt_det=args.use_gt_det,
            save_video=args.save_videos,
            video_fps=args.video_fps,
            benchmark_cfg=(
                {
                    'warmup': args.benchmark_warmup,
                    'max_frames': args.benchmark_max_frames,
                    'sync_cuda': args.benchmark_sync_cuda,
                    'include_io': args.benchmark_include_io,
                } if args.benchmark else None
            ),
            no_save_results=args.no_save_results,
            device=args.device
        )

        # Only count scenes that were actually processed (prevents seqmaps/empty dirs from polluting eval).
        if isinstance(scene_out, dict):
            if not scene_out.get('processed', False):
                continue
            processed_scenes.append(scene_name)
            timing_rows = scene_out.get('timing_rows')
        else:
            # Backwards-compatible fallback (older return type)
            processed_scenes.append(scene_name)
            timing_rows = scene_out

        if args.benchmark and timing_rows:
            benchmark_all.extend(timing_rows)
    
    # Move results to data/ subfolder for TrackEval compatibility
    # (kept for backwards compatibility; in this project we already write into {output}/data/)
    if not args.no_save_results:
        data_dir = os.path.join(args.output, 'data')
        os.makedirs(data_dir, exist_ok=True)
        for scene_name in processed_scenes:
            result_file = os.path.join(args.output, f'{scene_name}.txt')
            if os.path.exists(result_file):
                os.rename(result_file, os.path.join(data_dir, f'{scene_name}.txt'))
    
    print("\n" + "="*80)
    print(f"✓ Tracking complete! Results saved to: {args.output}")
    print("="*80)

    # Benchmark report
    if args.benchmark and benchmark_all:
        bench_df = __import__('pandas').DataFrame(benchmark_all)

        # exclude warmup frames
        bench_df_eval = bench_df[~bench_df['is_warmup']].copy()

        def _summ_series(s):
            if s is None:
                return None
            s = s.dropna().astype(float)
            if s.empty:
                return None
            return {
                'mean_ms': float(s.mean()),
                'median_ms': float(s.median()),
                'p95_ms': float(s.quantile(0.95)),
            }

        total_ms = bench_df_eval['total_ms'].astype(float) if not bench_df_eval.empty else None
        fps = float(1000.0 / total_ms.mean()) if (total_ms is not None and total_ms.mean() > 0) else 0.0

        per_scene = []
        for scene, g in bench_df_eval.groupby('scene'):
            tms = g['total_ms'].astype(float)
            per_scene.append({
                'scene': scene,
                'frames': int(len(g)),
                'fps': float(1000.0 / tms.mean()) if tms.mean() > 0 else 0.0,
                'total_ms': {
                    'mean_ms': float(tms.mean()),
                    'median_ms': float(tms.median()),
                    'p95_ms': float(tms.quantile(0.95)),
                },
                'detect_ms': _summ_series(g.get('detect_ms')),
                'track_ms': _summ_series(g.get('track_ms')),
                'read_ms': _summ_series(g.get('read_ms')) if args.benchmark_include_io else None,
            })

        bench_out = args.benchmark_output or os.path.join(args.output, 'benchmark.json')
        bench_payload = {
            'tracker': args.tracker,
            'device': args.device,
            'include_io': bool(args.benchmark_include_io),
            'sync_cuda': bool(args.benchmark_sync_cuda),
            'warmup_frames_per_scene': int(args.benchmark_warmup),
            'max_frames_per_scene': int(args.benchmark_max_frames),
            'n_rows': int(len(bench_df)),
            'n_rows_eval': int(len(bench_df_eval)),
            'fps_overall': fps,
            'total_ms': _summ_series(bench_df_eval.get('total_ms')),
            'detect_ms': _summ_series(bench_df_eval.get('detect_ms')),
            'track_ms': _summ_series(bench_df_eval.get('track_ms')),
            'read_ms': _summ_series(bench_df_eval.get('read_ms')) if args.benchmark_include_io else None,
            'per_scene': sorted(per_scene, key=lambda x: x['fps']),
        }

        with open(bench_out, 'w') as f:
            json.dump(bench_payload, f, indent=2)
        print(f"\n⏱️  Benchmark saved: {bench_out}")
        print(f"⏱️  FPS overall: {fps:.2f}")
    
    # Print TrackSSM diagnostics (only if counters are available)
    if args.tracker == 'trackssm':
        motion = getattr(tracker, 'trackssm_motion', None)
        has_counters = motion is not None and all(
            hasattr(motion, name)
            for name in (
                'n_predict_calls',
                'n_model_calls',
                'n_successful_predictions',
                'n_fallback_no_history',
                'n_fallback_sanity',
                'n_exceptions',
            )
        )
        if has_counters:
            print("\n" + "="*80)
            print("🔍 TRACKSSM DIAGNOSTICS")
            print("="*80)
            print(f"predict_calls:          {motion.n_predict_calls}")
            print(f"model_calls:            {motion.n_model_calls}")
            print(f"successful_predictions: {motion.n_successful_predictions}")
            print(f"fallback_no_history:    {motion.n_fallback_no_history}")
            print(f"fallback_sanity:        {motion.n_fallback_sanity}")
            print(f"exceptions:             {motion.n_exceptions}")
            print("")
            if motion.n_predict_calls > 0:
                success_rate = 100.0 * motion.n_successful_predictions / motion.n_predict_calls
                model_call_rate = 100.0 * motion.n_model_calls / motion.n_predict_calls
                print(f"Success rate:  {success_rate:.1f}% ({motion.n_successful_predictions}/{motion.n_predict_calls})")
                print(f"Model usage:   {model_call_rate:.1f}% ({motion.n_model_calls}/{motion.n_predict_calls})")
            print("="*80)
    
    # Automatic evaluation if requested
    if args.evaluate:
        print("\n" + "="*80)
        print("Running automatic evaluation...")
        print("="*80)
        
        # Create temporary seqmap for processed scenes
        temp_seqmap = os.path.join(args.output, 'seqmap_temp.txt')
        with open(temp_seqmap, 'w') as f:
            f.write('name\n')
            for scene_name in processed_scenes:
                f.write(f'{scene_name}\n')
        
        # Determine metrics output path
        if args.metrics_output:
            metrics_file = args.metrics_output
        else:
            metrics_file = os.path.join(args.output, 'metrics.json')
        
        # Create experiment configuration file
        config_data = {
            'experiment': {
                'timestamp': __import__('datetime').datetime.now().isoformat(),
                'tracker': args.tracker,
                'detector': 'GT (Oracle)' if args.use_gt_det else 'YOLOX',
                'dataset': args.data,
                'output_dir': args.output,
                'scenes': args.scenes if args.scenes else 'all',
                'num_scenes_processed': len(processed_scenes),
                'save_videos': args.save_videos,
                'video_fps': args.video_fps if args.save_videos else None
            },
            'tracker_config': {
                'track_thresh': args.track_thresh,
                'match_thresh': args.match_thresh,
                'max_age': args.max_age,
                'min_hits': args.min_hits
            },
            'device': args.device
        }
        
        # Add tracker-specific config
        if args.tracker == 'trackssm':
            if args.use_mot17_checkpoint:
                config_data['tracker_config']['checkpoint'] = 'weights/trackssm/phase1/phase1_decoder_best.pth'
                config_data['tracker_config']['checkpoint_type'] = 'MOT17 pretrained (Phase1)'
                config_data['tracker_config']['training_note'] = 'Baseline MOT17 checkpoint before NuScenes fine-tuning'
            else:
                config_data['tracker_config']['checkpoint'] = args.trackssm_checkpoint
                config_data['tracker_config']['checkpoint_type'] = 'NuScenes fine-tuned (Phase2)'
                config_data['tracker_config']['training_note'] = 'Fine-tuned on NuScenes after Phase1'
            config_data['tracker_config']['history_len'] = 5
            config_data['tracker_config']['cmc_method'] = args.cmc_method
        
        # Add detector config
        if not args.use_gt_det:
            detector_model = "YOLOX"
            if "yolox_x" in args.detector_weights.lower():
                detector_model = "YOLOX-X"
            elif "yolox_l" in args.detector_weights.lower():
                detector_model = "YOLOX-L"
            elif "yolox_m" in args.detector_weights.lower():
                detector_model = "YOLOX-M"
            elif "yolox_s" in args.detector_weights.lower():
                detector_model = "YOLOX-S"

            config_data['detector_config'] = {
                'model': detector_model,
                'weights': args.detector_weights,
                'conf_thresh': args.conf_thresh,
                'nms_thresh': args.nms_thresh,
                'test_size': list(detector_test_size),
                'num_classes': detector_num_classes,
                'fp16': bool(args.detector_fp16),
            }
        
        # Save experiment config
        config_file = os.path.join(args.output, 'experiment_config.json')
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Run evaluation with TrackEval (HOTA, DetA, AssA, IDF1, MOTA, CLEAR, Identity - 37 metrics)
        print("\n" + "="*80)
        print("RUNNING TRACKEVAL (37 METRICS)")
        print("="*80)
        
        # Determine GT path (default to 7-class dataset)
        gt_path = args.gt_data if args.gt_data else 'data/nuscenes_mot_front/val'
        
        print(f"GT dataset:     {gt_path}")
        print(f"Predictions:    {args.output}/data/")
        print(f"Output:         {metrics_file}")
        print()
        
        try:
            # Use evaluate_motmetrics.py for complete metrics
            import subprocess
            eval_cmd = [
                sys.executable, 'scripts/evaluation/evaluate_motmetrics.py',
                '--pred-folder', os.path.join(args.output, 'data'),
                '--output', metrics_file
            ]

            # Evaluate only the scenes we actually processed (faster + no missing-pred warnings)
            if processed_scenes:
                eval_cmd += ['--scenes', ','.join(processed_scenes)]
            
            print(f"Running: {' '.join(eval_cmd)}")
            result = subprocess.run(eval_cmd, capture_output=False, text=True)
            
            if result.returncode == 0:
                print("\n✅ Evaluation completed successfully")
                
                # Load and display summary
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        eval_results = json.load(f)
                    
                    # Add experiment config to metrics file
                    eval_results['experiment_config'] = config_data
                    with open(metrics_file, 'w') as f:
                        json.dump(eval_results, f, indent=2)
                    
                    print(f"\n✓ Full metrics saved: {metrics_file}")
                    print(f"  (includes HOTA, DetA, AssA, IDF1, MOTA, MOTP, MT, ML, IDSW, and 28 more metrics)")
                else:
                    print(f"\n⚠️  Metrics file not created: {metrics_file}")
            else:
                print(f"\n⚠️  Evaluation failed with return code {result.returncode}")
                print("   Falling back to basic metrics...")
                
                # Fallback to basic evaluator
                evaluator = NuScenesMultiClassEvaluator(
                    gt_folder=gt_path,
                    pred_folder=os.path.join(args.output, 'data'),
                    iou_threshold=args.iou_threshold
                )
                
                scene_list = []
                with open(temp_seqmap) as f:
                    lines = f.readlines()[1:]
                    scene_list = [line.strip() for line in lines if line.strip()]
                
                results = evaluator.evaluate_all(scene_list=scene_list)
                results['experiment_config'] = config_data
                evaluator.save_results(results, Path(metrics_file))
                
                print(f"\n✓ Basic metrics saved: {metrics_file}")
            
            print("="*80)
        
        except Exception as e:
            print(f"\n⚠️  Evaluation error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()

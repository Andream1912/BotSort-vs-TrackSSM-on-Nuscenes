#!/usr/bin/env python3
"""
Unified Tracking Script - Run any tracker with YOLOX detector

Usage:
    python track.py --tracker trackssm --data data/nuscenes_mot_front/val --output results/trackssm
    python track.py --tracker botsort --data data/nuscenes_mot_front/val --output results/botsort
"""

import os
import sys
import argparse
import cv2
import torch
import numpy as np
import json
import subprocess
from pathlib import Path
from tqdm import tqdm

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'external', 'YOLOX'))

from src.detectors.yolox_detector import YOLOXDetector
from src.trackers.tracker_factory import TrackerFactory


def parse_args():
    parser = argparse.ArgumentParser(description='Unified MOT tracking')
    
    # Tracker selection
    parser.add_argument('--tracker', type=str, required=True,
                       choices=['trackssm', 'botsort'],
                       help='Tracker to use')
    
    # Data paths
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset (e.g., data/nuscenes_mot_front/val)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for tracking results')
    
    # Detector config
    parser.add_argument('--use-gt-det', action='store_true',
                       help='Use ground truth detections instead of YOLOX (oracle mode)')
    parser.add_argument('--detector-weights', type=str,
                       default='weights/detectors/yolox_x.pth',
                       help='Path to YOLOX weights')
    parser.add_argument('--conf-thresh', type=float, default=0.1,
                       help='Detection confidence threshold')
    parser.add_argument('--nms-thresh', type=float, default=0.65,
                       help='NMS threshold')
    
    # Tracker config
    parser.add_argument('--track-thresh', type=float, default=0.2,
                       help='Track confidence threshold')
    parser.add_argument('--match-thresh', type=float, default=0.5,
                       help='Matching IoU threshold')
    parser.add_argument('--trackssm-checkpoint', type=str,
                       default='weights/trackssm/phase2/phase2_full_best.pth',
                       help='TrackSSM checkpoint path (default: Phase2 NuScenes fine-tuned)')
    parser.add_argument('--use-mot17-checkpoint', action='store_true',
                       help='Use MOT17 pretrained checkpoint (Phase1) instead of NuScenes fine-tuned (Phase2)')
    
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
                       help='Automatically evaluate after tracking (compute HOTA, CLEAR, Identity)')
    parser.add_argument('--per-class-metrics', action='store_true',
                       help='Compute per-class metrics (requires seqinfo.ini - usually not needed)')
    parser.add_argument('--metrics-output', type=str, default=None,
                       help='Output file for metrics JSON (default: {output}/metrics.json)')
    
    return parser.parse_args()


def get_scenes(data_root, scene_list=None):
    """Get list of scenes to process"""
    data_path = Path(data_root)
    
    if scene_list:
        scenes = [s.strip() for s in scene_list.split(',')]
    else:
        # Get all scenes
        scenes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    
    return scenes


def process_scene(scene_name, data_root, detector, tracker, output_dir, use_gt_det=False, save_video=False, video_fps=12):
    """Process a single scene"""
    scene_path = Path(data_root) / scene_name
    img_dir = scene_path / 'img1'
    
    if not img_dir.exists():
        print(f"⚠️  Image directory not found: {img_dir}")
        return
    
    # Get all images
    images = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
    
    if len(images) == 0:
        print(f"⚠️  No images found in {img_dir}")
        return
    
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
    
    # Process each frame (no per-frame progress bar for cleaner output)
    for frame_id, img_path in enumerate(images, start=1):
        # Load image
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"⚠️  Failed to load image: {img_path}")
            continue
        
        # Detect objects
        if use_gt_det:
            detections = detector.detect(scene_name, frame_id)
        else:
            detections = detector.detect(img)
        
        # Update tracker
        tracks = tracker.update(detections, frame=img)
        
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
    
    # Write results in two formats:
    # 1. With original classes (for analysis)
    # 2. With unified class=1 (for TrackEval)
    
    # Format 1: With classes
    output_with_classes = Path(output_dir) / 'with_classes' / f"{scene_name}.txt"
    output_with_classes.parent.mkdir(parents=True, exist_ok=True)
    with open(output_with_classes, 'w') as f:
        for r in results:
            f.write(f"{r['frame']},{r['track_id']},{r['bbox'][0]:.2f},{r['bbox'][1]:.2f},"
                   f"{r['bbox'][2]:.2f},{r['bbox'][3]:.2f},{r['confidence']:.4f},{r['class_id']},1\n")
    
    # Format 2: Unified class (for TrackEval) - save in data/ subdirectory
    output_file_data = Path(output_dir) / 'data' / f"{scene_name}.txt"
    output_file_data.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file_data, 'w') as f:
        for r in results:
            f.write(f"{r['frame']},{r['track_id']},{r['bbox'][0]:.2f},{r['bbox'][1]:.2f},"
                   f"{r['bbox'][2]:.2f},{r['bbox'][3]:.2f},{r['confidence']:.4f},1,1\n")
    
    # Release video writer if used
    if video_writer is not None:
        video_writer.release()
        print(f"  ✓ Video saved: {len(images)} frames")
    
    num_unique_tracks = len(set(r['track_id'] for r in results))
    print(f"  ✓ Saved {num_unique_tracks} unique tracks ({len(results)} total detections)")
    print(f"  ✓ Standard MOT: {output_file_data}")
    print(f"  ✓ With classes: {output_with_classes}")



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
        detector = YOLOXDetector(
            model_path=args.detector_weights,
            conf_thresh=args.conf_thresh,
            nms_thresh=args.nms_thresh,
            device=args.device
        )
    
    # Initialize tracker
    print(f"\n[2/3] Initializing {args.tracker.upper()} tracker...")
    
    tracker_config = {
        'device': args.device,
        'track_thresh': args.track_thresh,
        'match_thresh': args.match_thresh,
        'img_width': 1600,
        'img_height': 900,
        'max_age': 30,
        'min_hits': 3
    }
    
    # Add TrackSSM-specific config
    if args.tracker == 'trackssm':
        # Select checkpoint: MOT17 pretrained (Phase1) or NuScenes fine-tuned (Phase2)
        if args.use_mot17_checkpoint:
            checkpoint_path = 'weights/trackssm/phase1/phase1_decoder_best.pth'
            checkpoint_type = 'MOT17 pretrained (Phase1)'
            print(f"  Using {checkpoint_type}")
        else:
            checkpoint_path = args.trackssm_checkpoint
            checkpoint_type = 'NuScenes fine-tuned (Phase2)'
            print(f"  Using {checkpoint_type}")
        
        tracker_config['checkpoint_path'] = checkpoint_path
        tracker_config['checkpoint_type'] = checkpoint_type
        tracker_config['history_len'] = 5
    
    tracker = TrackerFactory.create(args.tracker, tracker_config)
    
    # Get scenes to process
    print(f"\n[3/3] Processing scenes...")
    scenes = get_scenes(args.data, args.scenes)
    print(f"Found {len(scenes)} scenes to process")
    
    # Process each scene with overall progress bar
    processed_scenes = []
    for idx, scene_name in enumerate(tqdm(scenes, desc="Overall Progress", unit="scene")):
        print(f"\n[Scene {idx+1}/{len(scenes)}] Processing {scene_name}...")
        process_scene(
            scene_name=scene_name,
            data_root=args.data,
            detector=detector,
            tracker=tracker,
            output_dir=args.output,
            use_gt_det=args.use_gt_det,
            save_video=args.save_videos,
            video_fps=args.video_fps
        )
        processed_scenes.append(scene_name)
    
    # Move results to data/ subfolder for TrackEval compatibility
    data_dir = os.path.join(args.output, 'data')
    os.makedirs(data_dir, exist_ok=True)
    for scene_name in processed_scenes:
        result_file = os.path.join(args.output, f'{scene_name}.txt')
        if os.path.exists(result_file):
            os.rename(result_file, os.path.join(data_dir, f'{scene_name}.txt'))
    
    print("\n" + "="*80)
    print(f"✓ Tracking complete! Results saved to: {args.output}")
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
                'max_age': 30,
                'min_hits': 3
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
        
        # Add detector config
        if not args.use_gt_det:
            config_data['detector_config'] = {
                'model': 'YOLOX-X',
                'weights': args.detector_weights,
                'conf_thresh': args.conf_thresh,
                'nms_thresh': args.nms_thresh
            }
        
        # Save experiment config
        config_file = os.path.join(args.output, 'experiment_config.json')
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Run evaluation
        eval_cmd = [
            sys.executable,
            os.path.join(project_root, 'evaluate.py'),
            '--gt', args.data,
            '--results', args.output,
            '--output', metrics_file,
            '--seqmap', temp_seqmap,
            '--config', config_file  # Pass config to evaluation
        ]
        
        # Add per-class flag only if requested
        if args.per_class_metrics:
            eval_cmd.append('--per-class')
        
        try:
            result = subprocess.run(eval_cmd, check=True, capture_output=False)
            
            # Load and display summary from summary file
            summary_file = os.path.join(os.path.dirname(metrics_file), 
                                       f"{os.path.splitext(os.path.basename(metrics_file))[0]}_summary.json")
            
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                
                print("\n" + "="*80)
                print("EVALUATION SUMMARY")
                print("="*80)
                
                if 'HOTA' in summary:
                    print(f"\n✓ HOTA: {summary['HOTA']:.4f}")
                
                if 'MOTA' in summary:
                    print(f"✓ MOTA: {summary['MOTA']:.4f}")
                
                if 'IDF1' in summary:
                    print(f"✓ IDF1: {summary['IDF1']:.4f}")
                
                if 'IDSW' in summary:
                    print(f"✓ ID Switches: {summary['IDSW']}")
                
                if 'FP' in summary and 'FN' in summary:
                    print(f"\nFP: {summary['FP']} | FN: {summary['FN']}")
                
                if 'Precision' in summary and 'Recall' in summary:
                    print(f"Precision: {summary['Precision']:.4f} | Recall: {summary['Recall']:.4f}")
                
                if 'Total_GT_IDs' in summary and 'Total_Predicted_IDs' in summary:
                    print(f"\nTracked IDs: {summary['Total_Predicted_IDs']} / {summary['Total_GT_IDs']} GT")
                
                print(f"\n✓ Full metrics: {metrics_file}")
                print(f"✓ Summary: {summary_file}")
                
                # Check per-class metrics
                per_class_file = os.path.join(os.path.dirname(metrics_file), 
                                             f"{os.path.splitext(os.path.basename(metrics_file))[0]}_per_class.json")
                if os.path.exists(per_class_file):
                    with open(per_class_file, 'r') as f:
                        per_class = json.load(f)
                    if per_class:
                        print(f"✓ Per-class metrics: {per_class_file}")
                        print(f"  Classes evaluated: {', '.join(per_class.keys())}")
                
                print("="*80)
        
        except subprocess.CalledProcessError as e:
            print(f"\n⚠️  Evaluation failed with error code {e.returncode}")
        except Exception as e:
            print(f"\n⚠️  Evaluation error: {e}")


if __name__ == '__main__':
    main()

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
from pathlib import Path
from tqdm import tqdm

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
                       help='Path to YOLOX weights')
    parser.add_argument('--conf-thresh', type=float, default=0.7,
                       help='Detection confidence threshold')
    parser.add_argument('--nms-thresh', type=float, default=0.65,
                       help='NMS threshold')
    
    # Tracker config
    parser.add_argument('--track-thresh', type=float, default=0.7,
                       help='Track confidence threshold')
    parser.add_argument('--match-thresh', type=float, default=0.7,
                       help='Matching IoU threshold')
    parser.add_argument('--max-age', type=int, default=30,
                       help='Maximum frames to keep lost track (default: 30)')
    parser.add_argument('--min-hits', type=int, default=1,
                       help='Minimum hits before track is activated (default: 3)')
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
                       help='Run evaluation after tracking (uses 7-class GT automatically)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for evaluation matching (default: 0.5)')
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
    print(f"  ✓ Saved {num_unique_tracks} unique tracks ({len(results)} total detections)")
    print(f"  ✓ Results: {output_file}")



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
            device=args.device,
            test_size=(896, 1600)  # Multiple of 32 for YOLOX (H, W)
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
        'min_hits': args.min_hits
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
    
    # Print TrackSSM diagnostics if using trackssm tracker
    if args.tracker == 'trackssm':
        try:
            motion = tracker.trackssm_motion
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
        except Exception as e:
            print(f"\n⚠️  Could not print TrackSSM diagnostics: {e}")
    
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
            # Use evaluate.py with TrackEval for complete metrics
            import subprocess
            eval_cmd = [
                sys.executable, 'evaluate.py',
                '--gt-folder', gt_path,
                '--results-folder', os.path.join(args.output, 'data'),
                '--output-file', metrics_file,
                '--seqmap-file', temp_seqmap
            ]
            
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

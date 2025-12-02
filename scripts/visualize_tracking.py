#!/usr/bin/env python3
"""
Interactive Tracking Visualization Tool

Visualize tracking results for any scene with any model.

Usage:
    python scripts/visualize_tracking.py
    python scripts/visualize_tracking.py --scene scene-0003_CAM_FRONT --model kalman
    python scripts/visualize_tracking.py --list-scenes
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import deque
import random

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Color palette for tracks (distinct colors)
COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), 
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128)
]

def get_color(track_id):
    """Get consistent color for track ID"""
    idx = track_id % len(COLORS)
    return COLORS[idx]

def parse_args():
    parser = argparse.ArgumentParser(description='Interactive Tracking Visualization')
    
    parser.add_argument('--results-dir', type=str, 
                       default='results/results_prima_del_meeting/last_minute',
                       help='Results directory (default: results/results_prima_del_meeting/last_minute)')
    parser.add_argument('--data-dir', type=str,
                       default='data/nuscenes_mot_front/val',
                       help='Data directory with images')
    
    parser.add_argument('--scene', type=str, default=None,
                       help='Scene name (e.g., scene-0003_CAM_FRONT)')
    parser.add_argument('--model', type=str, default=None,
                       choices=['kalman', 'trackssm_mot17', 'trackssm_finetuned'],
                       help='Model to visualize')
    parser.add_argument('--compare-all', action='store_true',
                       help='Compare all 3 models side-by-side in one video')
    parser.add_argument('--compare-best', action='store_true',
                       help='Compare only Kalman vs TrackSSM Finetuned (2 best models)')
    
    parser.add_argument('--list-scenes', action='store_true',
                       help='List available scenes and exit')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models and exit')
    
    parser.add_argument('--output', type=str, default=None,
                       help='Save video to file (auto-detected in headless mode)')
    parser.add_argument('--fps', type=int, default=12,
                       help='Output video FPS (default: 12)')
    parser.add_argument('--show-labels', action='store_true', default=True,
                       help='Show track IDs on bboxes')
    parser.add_argument('--show-trail', action='store_true',
                       help='Show track trails (last 10 frames)')
    parser.add_argument('--bbox-thickness', type=int, default=2,
                       help='Bounding box line thickness')
    parser.add_argument('--save-frames', type=str, default=None,
                       help='Save individual frames to directory (e.g., frames_output/)')
    parser.add_argument('--frame-interval', type=int, default=1,
                       help='Save every N frames (default: 1 = all frames)')
    
    return parser.parse_args()

def get_available_models(results_dir):
    """Get list of available models"""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"⚠️  Results directory not found: {results_path}")
        return []
    
    models = []
    
    # New structure: results/results_prima_del_meeting/last_minute/{kalman,trackssm_finetuned,trackssm_mot17}/data/
    model_mapping = {
        'kalman': ('kalman', 'Kalman Filter (BoT-SORT)', 'data'),
        'trackssm_mot17': ('trackssm_mot17', 'TrackSSM + MOT17 Pretrained', 'data'),
        'trackssm_finetuned': ('trackssm_finetuned', 'TrackSSM + NuScenes Fine-tuned', 'data')
    }
    
    for model_dir in sorted(results_path.iterdir()):
        if not model_dir.is_dir():
            continue
            
        if model_dir.name in model_mapping:
            short_name, display_name, data_subpath = model_mapping[model_dir.name]
            data_path = model_dir / data_subpath
            if data_path.exists():
                models.append({
                    'name': short_name,
                    'display_name': display_name,
                    'path': data_path
                })
    
    return models

def get_available_scenes(data_dir):
    """Get list of available scenes"""
    data_path = Path(data_dir)
    scenes = []
    
    for scene_dir in sorted(data_path.iterdir()):
        if scene_dir.is_dir() and not scene_dir.name.startswith('.'):
            # Check if it has images
            img_dir = scene_dir / 'img1'
            if img_dir.exists() and list(img_dir.glob('*.jpg')):
                scenes.append(scene_dir.name)
    
    return scenes

def load_tracking_results(results_path, scene_name):
    """Load tracking results for a scene"""
    # Try with full name first (scene-0003_CAM_FRONT.txt)
    tracking_file = results_path / f"{scene_name}.txt"
    
    # If not found, try without _CAM_FRONT suffix
    if not tracking_file.exists() and scene_name.endswith('_CAM_FRONT'):
        scene_name_short = scene_name.replace('_CAM_FRONT', '')
        tracking_file = results_path / f"{scene_name_short}.txt"
    
    if not tracking_file.exists():
        return None
    
    tracks = {}
    with open(tracking_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            class_id = int(parts[7]) if len(parts) > 7 else 1
            
            if frame_id not in tracks:
                tracks[frame_id] = []
            
            tracks[frame_id].append({
                'id': track_id,
                'bbox': [x, y, w, h],
                'conf': conf,
                'class_id': class_id
            })
    
    return tracks

def draw_bbox(img, bbox, track_id, conf, color, thickness=2, show_label=True):
    """Draw bounding box with track ID"""
    x, y, w, h = bbox
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    
    # Draw bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    if show_label:
        # Draw label background
        label = f"ID:{track_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
        cv2.putText(img, label, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)

def draw_trail(img, trail_positions, color):
    """Draw track trail"""
    if len(trail_positions) < 2:
        return
    
    for i in range(len(trail_positions) - 1):
        pt1 = trail_positions[i]
        pt2 = trail_positions[i + 1]
        alpha = (i + 1) / len(trail_positions)
        thickness = max(1, int(3 * alpha))
        cv2.line(img, pt1, pt2, color, thickness)

def visualize_comparison(scene_name, models, data_dir, args, comparison_type="all"):
    """Visualize models side-by-side
    
    Args:
        comparison_type: "all" (3 models) or "best" (2 models: Kalman vs TrackSSM Finetuned)
    """
    num_models = len(models)
    
    if comparison_type == "all":
        print(f"\n🎬 Comparing all {num_models} models on: {scene_name}")
    else:
        print(f"\n🎬 Comparing Kalman vs TrackSSM Finetuned on: {scene_name}")
    
    # Load tracking results for all models
    all_tracks = {}
    for model in models:
        tracks = load_tracking_results(model['path'], scene_name)
        if tracks is None:
            print(f"⚠️  No tracking results for {model['display_name']}")
            all_tracks[model['name']] = {}
        else:
            all_tracks[model['name']] = tracks
            print(f"✓ Loaded {model['display_name']}: {len(tracks)} frames")
    
    # Get image directory
    img_dir = Path(data_dir) / scene_name / 'img1'
    if not img_dir.exists():
        print(f"❌ Image directory not found: {img_dir}")
        return
    
    img_files = sorted(img_dir.glob('*.jpg'))
    if not img_files:
        print(f"❌ No images found in {img_dir}")
        return
    
    # Read first frame to get dimensions
    first_img = cv2.imread(str(img_files[0]))
    h, w = first_img.shape[:2]
    
    # Calculate dimensions for side-by-side
    scale = 0.6 if num_models == 2 else 0.5  # Larger frames for 2-model comparison
    new_w = int(w * scale)
    new_h = int(h * scale)
    canvas_w = new_w * num_models + (num_models + 1) * 20  # spacing
    canvas_h = new_h + 60  # Extra space for labels
    
    # Setup video writer
    if args.output is None:
        if comparison_type == "best":
            args.output = f"comparison_best_{scene_name}.mp4"
        else:
            args.output = f"comparison_all_{scene_name}.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (canvas_w, canvas_h))
    print(f"💾 Saving comparison video to: {args.output}")
    
    # Track trails for each model
    track_trails = [{} for _ in range(num_models)]  # One dict per model
    
    # Process frames
    print("\n🎥 Processing frames...")
    for frame_idx, img_file in enumerate(tqdm(img_files, desc="Rendering comparison")):
        frame_id = frame_idx + 1
        
        # Create canvas
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 50
        
        # Process each model
        for col_idx, (model, model_tracks) in enumerate(zip(models, [all_tracks[m['name']] for m in models])):
            # Load image
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            # Get tracks for this frame
            frame_tracks = model_tracks.get(frame_id, [])
            
            # Draw tracks
            for track in frame_tracks:
                track_id = track['id']
                bbox = track['bbox']
                conf = track['conf']
                
                color = get_color(track_id)
                draw_bbox(img, bbox, track_id, conf, color, args.bbox_thickness, args.show_labels)
                
                # Update trails
                if args.show_trail:
                    cx = int(bbox[0] + bbox[2] / 2)
                    cy = int(bbox[1] + bbox[3] / 2)
                    if track_id not in track_trails[col_idx]:
                        track_trails[col_idx][track_id] = deque(maxlen=10)
                    track_trails[col_idx][track_id].append((cx, cy))
                    draw_trail(img, list(track_trails[col_idx][track_id]), color)
            
            # Add frame info
            info_text = f"Frame {frame_id}/{len(img_files)} | Tracks: {len(frame_tracks)}"
            cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            
            # Resize and place on canvas
            img_resized = cv2.resize(img, (new_w, new_h))
            x_offset = col_idx * (new_w + 20) + 10
            y_offset = 50
            
            # Ensure dimensions match (handle rounding issues)
            h_canvas = min(new_h, canvas.shape[0] - y_offset)
            w_canvas = min(new_w, canvas.shape[1] - x_offset)
            canvas[y_offset:y_offset+h_canvas, x_offset:x_offset+w_canvas] = img_resized[:h_canvas, :w_canvas]
            
            # Add model label on top
            label = model['display_name'].split(' (')[0]  # Short name
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_x = x_offset + (new_w - label_size[0]) // 2
            cv2.putText(canvas, label, (label_x, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 2)
        
        # Write frame
        writer.write(canvas)
    
    writer.release()
    print(f"\n✅ Comparison video saved: {args.output}")

def visualize_scene(scene_name, model_info, data_dir, args):
    """Visualize tracking for a scene"""
    print(f"\n🎬 Visualizing: {scene_name}")
    print(f"📊 Model: {model_info['display_name']}")
    
    # Load tracking results
    tracks = load_tracking_results(model_info['path'], scene_name)
    if tracks is None:
        print(f"❌ No tracking results found for {scene_name}")
        return
    
    # Get image directory
    img_dir = Path(data_dir) / scene_name / 'img1'
    if not img_dir.exists():
        print(f"❌ Image directory not found: {img_dir}")
        return
    
    # Get all images
    img_files = sorted(img_dir.glob('*.jpg'))
    if not img_files:
        print(f"❌ No images found in {img_dir}")
        return
    
    print(f"📁 Found {len(img_files)} frames")
    print(f"📦 Found {len(tracks)} frames with tracks")
    
    # In headless mode (no display), force video output
    if args.output is None and os.environ.get('DISPLAY') is None:
        args.output = f"visualization_{scene_name}_{model_info['name']}.mp4"
        print(f"⚠️  No display detected (headless mode)")
        print(f"💾 Automatically saving to: {args.output}")
    
    # Setup video writer if output specified
    writer = None
    if args.output:
        first_img = cv2.imread(str(img_files[0]))
        h, w = first_img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))
        print(f"💾 Saving video to: {args.output}")
    
    # Setup frame saving directory if specified
    frames_dir = None
    if args.save_frames:
        frames_dir = Path(args.save_frames)
        frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"💾 Saving frames to: {frames_dir}")
    
    # Track trails for visualization
    track_trails = {}  # track_id -> list of (cx, cy) positions
    
    # Process frames
    print("\n🎥 Processing frames...")
    for frame_idx, img_file in enumerate(tqdm(img_files, desc="Rendering")):
        frame_id = frame_idx + 1
        
        # Load image
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        
        # Get tracks for this frame
        frame_tracks = tracks.get(frame_id, [])
        
        # Draw trails first (so they appear behind bboxes)
        if args.show_trail:
            for track_id, positions in track_trails.items():
                if len(positions) > 1:
                    color = get_color(track_id)
                    draw_trail(img, positions[-10:], color)  # Last 10 positions
        
        # Draw tracks
        for track in frame_tracks:
            track_id = track['id']
            bbox = track['bbox']
            conf = track['conf']
            
            # Get color for this track
            color = get_color(track_id)
            
            # Draw bbox
            draw_bbox(img, bbox, track_id, conf, color, 
                     thickness=args.bbox_thickness, show_label=args.show_labels)
            
            # Update trail
            if args.show_trail:
                x, y, w, h = bbox
                cx, cy = int(x + w/2), int(y + h/2)
                if track_id not in track_trails:
                    track_trails[track_id] = []
                track_trails[track_id].append((cx, cy))
        
        # Add info overlay
        info_text = f"Frame: {frame_id}/{len(img_files)} | Tracks: {len(frame_tracks)} | Model: {model_info['display_name']}"
        cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Save frame if requested
        if frames_dir and frame_idx % args.frame_interval == 0:
            frame_filename = frames_dir / f"frame_{frame_id:06d}.jpg"
            cv2.imwrite(str(frame_filename), img)
        
        # Show or save
        if writer:
            writer.write(img)
        else:
            # Try to display, but handle headless environment
            try:
                cv2.imshow(f'{scene_name} - {model_info["display_name"]}', img)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Pause
                    cv2.waitKey(0)
            except cv2.error:
                print("⚠️  Cannot display (no X server). Use --output to save video.")
                break
    
    # Cleanup
    if writer:
        writer.release()
        print(f"\n✅ Video saved: {args.output}")
        
        # Show video info
        import subprocess
        try:
            result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                   '-show_entries', 'stream=width,height,duration',
                                   '-of', 'default=noprint_wrappers=1', args.output],
                                  capture_output=True, text=True, timeout=5)
            print(f"📹 Video info: {result.stdout.strip()}")
        except:
            pass
    else:
        try:
            cv2.destroyAllWindows()
        except:
            pass

def interactive_mode(results_dir, data_dir, args):
    """Interactive selection mode"""
    # Get available options
    models = get_available_models(results_dir)
    scenes = get_available_scenes(data_dir)
    
    if not models:
        print(f"❌ No models found in {results_dir}")
        return
    
    if not scenes:
        print(f"❌ No scenes found in {data_dir}")
        return
    
    print("\n" + "="*80)
    print("🎬 INTERACTIVE TRACKING VISUALIZATION")
    print("="*80)
    
    # Select model
    print("\n📊 Available Models:")
    for i, model in enumerate(models):
        print(f"  [{i+1}] {model['display_name']}")
    print(f"  [{len(models)+1}] 🎯 Compare Kalman vs TrackSSM Finetuned (Best 2)")
    print(f"  [{len(models)+2}] 🎬 Compare All {len(models)} Models Side-by-Side")
    
    selected_model = None
    compare_mode = None  # None, "best", or "all"
    
    while True:
        try:
            model_choice = input(f"\nSelect model (1-{len(models)+2}) or 'q' to quit: ")
            if model_choice.lower() == 'q':
                return
            model_idx = int(model_choice) - 1
            if model_idx == len(models):
                # Compare best 2 (Kalman vs TrackSSM Finetuned)
                compare_mode = "best"
                break
            elif model_idx == len(models) + 1:
                # Compare all models
                compare_mode = "all"
                break
            elif 0 <= model_idx < len(models):
                selected_model = models[model_idx]
                break
            else:
                print(f"❌ Invalid choice. Please enter 1-{len(models)+2}")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
    
    # Select scene
    print(f"\n🎬 Available Scenes (total: {len(scenes)}):")
    for i, scene in enumerate(scenes[:20]):  # Show first 20
        print(f"  [{i+1}] {scene}")
    if len(scenes) > 20:
        print(f"  ... and {len(scenes) - 20} more")
    
    while True:
        scene_input = input(f"\nEnter scene number (1-{len(scenes)}), name, or 'list' to see all: ")
        
        if scene_input.lower() == 'list':
            print("\n📋 All Scenes:")
            for i, scene in enumerate(scenes):
                print(f"  [{i+1}] {scene}")
            continue
        
        try:
            # Try as number
            scene_idx = int(scene_input) - 1
            if 0 <= scene_idx < len(scenes):
                selected_scene = scenes[scene_idx]
                break
            else:
                print(f"❌ Invalid choice. Please enter 1-{len(scenes)}")
        except ValueError:
            # Try as scene name
            if scene_input in scenes:
                selected_scene = scene_input
                break
            else:
                # Try partial match
                matches = [s for s in scenes if scene_input.lower() in s.lower()]
                if len(matches) == 1:
                    selected_scene = matches[0]
                    print(f"✓ Found: {selected_scene}")
                    break
                elif len(matches) > 1:
                    print(f"❌ Multiple matches found:")
                    for m in matches[:10]:
                        print(f"  - {m}")
                    print("Please be more specific.")
                else:
                    print(f"❌ Scene not found: {scene_input}")
    
    # Ask for output (check if headless first)
    is_headless = os.environ.get('DISPLAY') is None
    
    if compare_mode:
        # Comparison mode always saves to file
        if compare_mode == "best":
            default_output = f"comparison_best_{selected_scene}.mp4"
            print(f"\n💾 Kalman vs TrackSSM comparison - will save to video file")
        else:
            default_output = f"comparison_all_{selected_scene}.mp4"
            print(f"\n💾 Full comparison mode - will save to video file")
        
        output_path = input(f"Output filename (default: {default_output}): ").strip()
        if not output_path:
            output_path = default_output
        args.output = output_path
    elif is_headless:
        print("\n⚠️  No display detected (headless mode)")
        print("💾 Will save video file automatically")
        default_output = f"visualization_{selected_scene}_{selected_model['name']}.mp4"
        output_path = input(f"Output filename (default: {default_output}): ").strip()
        if not output_path:
            output_path = default_output
        args.output = output_path
    else:
        print("\n💾 Output Options:")
        print("  [1] Display only (press 'q' to quit, SPACE to pause)")
        print("  [2] Save to video file")
        
        save_video = input("\nChoice (1-2, default: 1): ").strip()
        
        if save_video == '2':
            default_output = f"visualization_{selected_scene}_{selected_model['name']}.mp4"
            output_path = input(f"Output filename (default: {default_output}): ").strip()
            if not output_path:
                output_path = default_output
            args.output = output_path
    
    # Visualize
    if compare_mode == "best":
        # Filter only Kalman and TrackSSM Finetuned
        best_models = [m for m in models if m['name'] in ['kalman', 'trackssm_finetuned']]
        if len(best_models) < 2:
            print(f"❌ Need both Kalman and TrackSSM Finetuned for comparison")
            return
        visualize_comparison(selected_scene, best_models, data_dir, args, comparison_type="best")
    elif compare_mode == "all":
        visualize_comparison(selected_scene, models[:3], data_dir, args, comparison_type="all")
    else:
        visualize_scene(selected_scene, selected_model, data_dir, args)
    
    # Ask to continue
    print("\n" + "="*80)
    continue_choice = input("Visualize another scene? (y/n): ").strip().lower()
    if continue_choice == 'y':
        interactive_mode(results_dir, data_dir, args)

def main():
    args = parse_args()
    
    # List modes
    if args.list_scenes:
        scenes = get_available_scenes(args.data_dir)
        print(f"\n📋 Available Scenes ({len(scenes)}):")
        for scene in scenes:
            print(f"  - {scene}")
        return
    
    if args.list_models:
        models = get_available_models(args.results_dir)
        print(f"\n📊 Available Models ({len(models)}):")
        for model in models:
            print(f"  - {model['name']}: {model['display_name']}")
            print(f"    Path: {model['path']}")
        return
    
    # Direct visualization mode
    if args.scene and (args.model or args.compare_all or args.compare_best):
        models = get_available_models(args.results_dir)
        
        if args.compare_best:
            # Compare Kalman vs TrackSSM Finetuned
            best_models = [m for m in models if m['name'] in ['kalman', 'trackssm_finetuned']]
            if len(best_models) < 2:
                print(f"❌ Need both Kalman and TrackSSM Finetuned for comparison")
                return
            visualize_comparison(args.scene, best_models, args.data_dir, args, comparison_type="best")
        elif args.compare_all:
            # Compare all models side-by-side
            if len(models) < 3:
                print(f"❌ Need at least 3 models for comparison, found {len(models)}")
                return
            visualize_comparison(args.scene, models[:3], args.data_dir, args, comparison_type="all")
        else:
            # Single model visualization
            model_dict = {m['name']: m for m in models}
            
            if args.model not in model_dict:
                print(f"❌ Model not found: {args.model}")
                print(f"Available models: {list(model_dict.keys())}")
                return
            
            selected_model = model_dict[args.model]
            visualize_scene(args.scene, selected_model, args.data_dir, args)
    else:
        # Interactive mode
        interactive_mode(args.results_dir, args.data_dir, args)

if __name__ == '__main__':
    main()

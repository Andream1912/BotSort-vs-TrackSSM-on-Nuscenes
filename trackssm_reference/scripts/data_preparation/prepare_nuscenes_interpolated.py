#!/usr/bin/env python3
"""
Prepare NuScenes dataset with temporal interpolation for TrackSSM fine-tuning.

This script:
1. Loads NuScenes annotations (2 Hz keyframes)
2. Projects 3D boxes to 2D image coordinates
3. Interpolates linearly between keyframes to generate 12 FPS annotations
4. Exports to MOT format for all 6 cameras and 7 classes

Usage:
    python prepare_nuscenes_interpolated.py \\
        --dataroot /path/to/nuscenes \\
        --version v1.0-trainval \\
        --output_dir ./data/nuscenes_mot_6cams_interpolated \\
        --cameras CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT \\
                  CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT \\
        --split train
"""

import os
import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm

# Add NuScenes devkit to path
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
    from pyquaternion import Quaternion
except ImportError:
    print("ERROR: NuScenes devkit not found!")
    print("Install with: pip install nuscenes-devkit")
    sys.exit(1)


# NuScenes class mapping to 7 classes
NUSCENES_CLASSES = {
    'vehicle.car': 0,
    'vehicle.truck': 1,
    'vehicle.bus': 2,
    'vehicle.trailer': 3,
    'human.pedestrian': 4,
    'vehicle.motorcycle': 5,
    'vehicle.bicycle': 6
}

CLASS_NAMES = ['car', 'truck', 'bus', 'trailer', 'pedestrian', 'motorcycle', 'bicycle']


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare interpolated NuScenes dataset for TrackSSM')
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Path to NuScenes dataset root (e.g., /mnt/datasets/Nuscense)')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                       choices=['v1.0-trainval', 'v1.0-test', 'v1.0-mini'],
                       help='NuScenes version')
    parser.add_argument('--output_dir', type=str, 
                       default='./data/nuscenes_mot_6cams_interpolated',
                       help='Output directory for MOT format data')
    parser.add_argument('--cameras', nargs='+', 
                       default=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                               'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
                       help='Cameras to process')
    parser.add_argument('--split', type=str, required=True,
                       choices=['train', 'val', 'test'],
                       help='Data split to process')
    parser.add_argument('--target_fps', type=int, default=12,
                       help='Target FPS for interpolation (default: 12)')
    parser.add_argument('--min_visibility', type=int, default=1,
                       choices=[1, 2, 3, 4],
                       help='Minimum visibility level (1-4, default: 1=any)')
    
    return parser.parse_args()


def get_deterministic_track_id(scene_token: str, instance_token: str) -> int:
    """
    Generate deterministic track ID from scene and instance tokens.
    
    Uses string concatenation instead of hash() for reproducibility.
    
    Args:
        scene_token: Scene identifier
        instance_token: Instance (track) identifier
        
    Returns:
        Deterministic integer track ID
    """
    # Combine tokens and take last 10 digits for uniqueness
    combined = scene_token + instance_token
    # Use last 10 hex characters and convert to int
    unique_id = int(combined[-10:], 16)
    return unique_id


def project_box_to_2d(nusc: NuScenes, 
                     sample_data_token: str,
                     box_3d) -> Optional[Tuple[float, float, float, float]]:
    """
    Project 3D box to 2D image coordinates.
    
    Args:
        nusc: NuScenes instance
        sample_data_token: Sample data token for camera
        box_3d: 3D box in global coordinates
        
    Returns:
        Tuple of (x, y, w, h) or None if not visible
    """
    # Get sample data, calibration, and ego pose
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    
    # Transform box: global → ego → camera
    box = box_3d.copy()
    
    # Step 1: global → ego
    box.translate(-np.array(pose_record['translation']))
    box.rotate(Quaternion(pose_record['rotation']).inverse)
    
    # Step 2: ego → camera
    box.translate(-np.array(cs_record['translation']))
    box.rotate(Quaternion(cs_record['rotation']).inverse)
    
    # Check if box is in front of camera
    if box.center[2] < 0.1:
        return None
    
    # Get camera intrinsics
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])
    
    # Project 3D box corners to 2D
    corners_3d = box.corners()
    corners_2d = view_points(corners_3d, cam_intrinsic, normalize=True)[:2, :]
    
    # Get image dimensions
    imsize = (sd_record['width'], sd_record['height'])
    
    # Check if box is visible in image (geometric check only)
    if not box_in_image(box, cam_intrinsic, imsize, vis_level=BoxVisibility.ANY):
        return None
    
    # Calculate 2D bounding box
    x_min = max(0, float(np.min(corners_2d[0, :])))
    x_max = min(imsize[0], float(np.max(corners_2d[0, :])))
    y_min = max(0, float(np.min(corners_2d[1, :])))
    y_max = min(imsize[1], float(np.max(corners_2d[1, :])))
    
    # Convert to MOT format (x, y, w, h)
    x = x_min
    y = y_min
    w = x_max - x_min
    h = y_max - y_min
    
    # Skip tiny boxes
    if w < 1 or h < 1:
        return None
    
    return (x, y, w, h)


def interpolate_boxes(box0: np.ndarray, box1: np.ndarray, alpha: float) -> np.ndarray:
    """
    Linear interpolation between two boxes.
    
    Args:
        box0: First box [x, y, w, h]
        box1: Second box [x, y, w, h]
        alpha: Interpolation factor (0 = box0, 1 = box1)
        
    Returns:
        Interpolated box [x, y, w, h]
    """
    return box0 * (1 - alpha) + box1 * alpha


def process_scene_camera(nusc: NuScenes,
                         scene_token: str,
                         camera_name: str,
                         output_path: Path,
                         target_fps: int = 12,
                         min_visibility: int = 1) -> Dict:
    """
    Process single scene-camera with interpolation.
    
    Args:
        nusc: NuScenes instance
        scene_token: Scene identifier
        camera_name: Camera channel name
        output_path: Output file path
        target_fps: Target FPS for interpolation
        min_visibility: Minimum visibility level (1-4, from NuScenes visibility_token)
        
    Returns:
        Statistics dictionary
    """
    scene = nusc.get('scene', scene_token)
    
    # Calculate frames per keyframe based on target_fps
    # NuScenes keyframes are at 2 Hz (every 0.5s)
    frames_per_keyframe = target_fps // 2  # For 12fps: 12 // 2 = 6
    
    # Collect all keyframe annotations
    keyframe_data = []  # List of (frame_idx, sample_token, timestamp)
    sample_token = scene['first_sample_token']
    frame_idx = 0
    
    while sample_token:
        sample = nusc.get('sample', sample_token)
        keyframe_data.append((frame_idx, sample_token, sample['timestamp']))
        
        # Increment frame by keyframe interval
        frame_idx += frames_per_keyframe
        sample_token = sample['next']
    
    # Dictionary to store tracks: instance_token -> list of (frame_idx, box_2d, class_id, visibility)
    tracks = defaultdict(list)
    
    # Process each keyframe
    for frame_idx, sample_token, timestamp in keyframe_data:
        sample = nusc.get('sample', sample_token)
        
        # Skip if camera not in sample
        if camera_name not in sample['data']:
            continue
            
        sample_data_token = sample['data'][camera_name]
        
        # Get all annotations for this sample
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            
            # Filter by visibility (1-4 scale)
            vis_token = int(ann['visibility_token'])
            if vis_token < min_visibility:
                continue
            
            # Filter by class
            category = ann['category_name']
            class_id = None
            for prefix, cid in NUSCENES_CLASSES.items():
                if category.startswith(prefix):
                    class_id = cid
                    break
            
            if class_id is None:
                continue
            
            # Project to 2D
            box_3d = nusc.get_box(ann_token)
            box_2d = project_box_to_2d(nusc, sample_data_token, box_3d)
            
            if box_2d is None:
                continue
            
            x, y, w, h = box_2d
            instance_token = ann['instance_token']
            
            tracks[instance_token].append({
                'frame_idx': frame_idx,
                'box': np.array([x, y, w, h]),
                'class_id': class_id,
                'visibility': vis_token
            })
    
    # Interpolate between keyframes
    interpolated_annotations = []
    
    for instance_token, detections in tracks.items():
        # Sort by frame
        detections = sorted(detections, key=lambda d: d['frame_idx'])
        
        # Generate deterministic track ID
        track_id = get_deterministic_track_id(scene_token, instance_token)
        
        # Interpolate between consecutive keyframes
        for i in range(len(detections) - 1):
            det0 = detections[i]
            det1 = detections[i + 1]
            
            frame0 = det0['frame_idx']
            frame1 = det1['frame_idx']
            
            # Number of frames to interpolate (should be 6 for 2Hz to 12fps)
            num_interpolated = frame1 - frame0
            
            for j in range(num_interpolated):
                frame_id = frame0 + j
                alpha = j / num_interpolated
                
                # Interpolate box
                box_interp = interpolate_boxes(det0['box'], det1['box'], alpha)
                
                # Interpolate visibility (use minimum)
                vis_interp = min(det0['visibility'], det1['visibility'])
                
                interpolated_annotations.append({
                    'frame_id': frame_id + 1,  # MOT format is 1-indexed
                    'track_id': track_id,
                    'x': box_interp[0],
                    'y': box_interp[1],
                    'w': box_interp[2],
                    'h': box_interp[3],
                    'conf': 1.0,
                    'class_id': det0['class_id'],
                    'visibility': vis_interp
                })
        
        # Add last keyframe
        if detections:
            det_last = detections[-1]
            track_id = get_deterministic_track_id(scene_token, instance_token)
            interpolated_annotations.append({
                'frame_id': det_last['frame_idx'] + 1,
                'track_id': track_id,
                'x': det_last['box'][0],
                'y': det_last['box'][1],
                'w': det_last['box'][2],
                'h': det_last['box'][3],
                'conf': 1.0,
                'class_id': det_last['class_id'],
                'visibility': det_last['visibility']
            })
    
    # Sort by frame_id and track_id
    interpolated_annotations.sort(key=lambda a: (a['frame_id'], a['track_id']))
    
    # Write to MOT format
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for ann in interpolated_annotations:
            # MOT format: frame,id,x,y,w,h,conf,class,visibility,-1
            f.write(f"{ann['frame_id']},{ann['track_id']},"
                   f"{ann['x']:.2f},{ann['y']:.2f},"
                   f"{ann['w']:.2f},{ann['h']:.2f},"
                   f"{ann['conf']:.6f},{ann['class_id']},"
                   f"{ann['visibility']:.2f},-1\n")
    
    # Statistics
    unique_tracks = len(tracks)
    total_bboxes = len(interpolated_annotations)
    keyframe_bboxes = sum(len(dets) for dets in tracks.values())
    interpolated_bboxes = total_bboxes - keyframe_bboxes
    
    return {
        'scene': scene['name'],
        'camera': camera_name,
        'tracks': unique_tracks,
        'keyframe_bboxes': keyframe_bboxes,
        'interpolated_bboxes': interpolated_bboxes,
        'total_bboxes': total_bboxes
    }


def main():
    args = parse_args()
    
    print("=" * 80)
    print("NuScenes Interpolated Dataset Preparation for TrackSSM Fine-tuning")
    print("=" * 80)
    print(f"Dataroot: {args.dataroot}")
    print(f"Version: {args.version}")
    print(f"Split: {args.split}")
    print(f"Output: {args.output_dir}")
    print(f"Cameras: {', '.join(args.cameras)}")
    print(f"Target FPS: {args.target_fps}")
    print(f"Min visibility: {args.min_visibility}")
    print("=" * 80)
    
    # Load NuScenes
    print("\nLoading NuScenes dataset...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    
    # Get scenes for split using official splits
    from nuscenes.utils import splits as nusc_splits
    
    if args.version.startswith('v1.0-trainval'):
        if args.split == 'train':
            scene_names = set(nusc_splits.train)
        elif args.split == 'val':
            scene_names = set(nusc_splits.val)
        else:  # test - use a custom test split or validation
            scene_names = set(nusc_splits.val)  # Or define your own test scenes
    elif args.version.startswith('v1.0-mini'):
        # For mini, just use all scenes for any split
        scene_names = set([s['name'] for s in nusc.scene])
    else:
        raise ValueError(f"Unknown version: {args.version}")
    
    scenes = [s for s in nusc.scene if s['name'] in scene_names]
    
    print(f"\nProcessing {len(scenes)} scenes for {args.split} split")
    
    # Output directory
    output_dir = Path(args.output_dir) / args.split
    
    # Statistics
    total_stats = defaultdict(int)
    scene_stats = []
    
    # Process each scene-camera combination
    total_combinations = len(scenes) * len(args.cameras)
    
    with tqdm(total=total_combinations, desc="Processing scene-cameras") as pbar:
        for scene in scenes:
            scene_token = scene['token']
            scene_name = scene['name']
            
            for camera_name in args.cameras:
                # Output path: {split}/{scene}-{camera}/gt/gt.txt
                scene_cam_name = f"{scene_name}-{camera_name}"
                output_path = output_dir / scene_cam_name / 'gt' / 'gt.txt'
                
                # Process
                stats = process_scene_camera(
                    nusc, scene_token, camera_name, output_path,
                    args.target_fps, args.min_visibility
                )
                
                # Accumulate stats
                total_stats['scene_cameras'] += 1
                total_stats['tracks'] += stats['tracks']
                total_stats['keyframe_bboxes'] += stats['keyframe_bboxes']
                total_stats['interpolated_bboxes'] += stats['interpolated_bboxes']
                total_stats['total_bboxes'] += stats['total_bboxes']
                
                scene_stats.append(stats)
                
                pbar.update(1)
    
    # Create seqinfo.ini for each scene-camera
    print("\nCreating seqinfo.ini files...")
    for scene in tqdm(scenes, desc="Writing seqinfo"):
        for camera_name in args.cameras:
            scene_name = scene['name']
            scene_cam_name = f"{scene_name}-{camera_name}"
            
            # Get first sample to get image dimensions
            sample_token = scene['first_sample_token']
            sample = nusc.get('sample', sample_token)
            
            if camera_name in sample['data']:
                sd_token = sample['data'][camera_name]
                sd_record = nusc.get('sample_data', sd_token)
                
                img_width = sd_record['width']
                img_height = sd_record['height']
                
                # Count frames (last frame index + 1)
                gt_file = output_dir / scene_cam_name / 'gt' / 'gt.txt'
                if gt_file.exists():
                    with open(gt_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            seq_length = max(int(line.split(',')[0]) for line in lines)
                        else:
                            seq_length = 0
                else:
                    seq_length = 0
                
                # Write seqinfo.ini
                seqinfo_path = output_dir / scene_cam_name / 'seqinfo.ini'
                with open(seqinfo_path, 'w') as f:
                    f.write(f"[Sequence]\n")
                    f.write(f"name={scene_cam_name}\n")
                    f.write(f"imDir=img1\n")
                    f.write(f"frameRate={args.target_fps}\n")
                    f.write(f"seqLength={seq_length}\n")
                    f.write(f"imWidth={img_width}\n")
                    f.write(f"imHeight={img_height}\n")
                    f.write(f"imExt=.jpg\n")
    
    # Save statistics
    stats_file = output_dir / 'dataset_stats.json'
    with open(stats_file, 'w') as f:
        json.dump({
            'split': args.split,
            'cameras': args.cameras,
            'target_fps': args.target_fps,
            'total_stats': dict(total_stats),
            'per_scene': scene_stats
        }, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Split: {args.split}")
    print(f"Total scene-cameras: {total_stats['scene_cameras']}")
    print(f"Total tracks: {total_stats['tracks']:,}")
    print(f"Keyframe bboxes: {total_stats['keyframe_bboxes']:,}")
    print(f"Interpolated bboxes: {total_stats['interpolated_bboxes']:,}")
    print(f"Total bboxes: {total_stats['total_bboxes']:,}")
    print(f"Interpolation ratio: {total_stats['interpolated_bboxes']/total_stats['total_bboxes']*100:.1f}%")
    print(f"\nOutput directory: {output_dir}")
    print(f"Statistics saved to: {stats_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()

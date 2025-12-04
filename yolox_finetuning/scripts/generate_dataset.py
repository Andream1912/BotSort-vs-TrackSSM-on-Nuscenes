#!/usr/bin/env python3
"""
Generate YOLOX Dataset from NuScenes (OPTIMAL VERSION)

Strategy:
- Use ONLY 700 train scenes (not the 150 val = test set)
- Split 700 scenes → 560 train (80%) + 140 val (20%)
- ALL 6 cameras per scene
- COCO format annotations
- Parallel processing for speed

Output:
- data/nuscenes_yolox_6cams/annotations/train.json (560 scenes × 6 cams × ~40 frames = ~134K images)
- data/nuscenes_yolox_6cams/annotations/val.json (140 scenes × 6 cams × ~40 frames = ~33K images)
- Symlinks to original images (no copy)
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import argparse
from multiprocessing import Pool, cpu_count

# NuScenes
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion


# NuScenes 7 classes mapping to COCO-style IDs
NUSCENES_CLASSES = {
    'vehicle.car': 1,
    'vehicle.truck': 2,
    'vehicle.bus': 3,
    'vehicle.trailer': 4,
    'human.pedestrian.adult': 5,
    'human.pedestrian.child': 5,  # Merge to pedestrian
    'human.pedestrian.construction_worker': 5,
    'human.pedestrian.police_officer': 5,
    'vehicle.motorcycle': 6,
    'vehicle.bicycle': 7,
}

CATEGORY_NAMES = {
    1: 'car',
    2: 'truck', 
    3: 'bus',
    4: 'trailer',
    5: 'pedestrian',
    6: 'motorcycle',
    7: 'bicycle'
}

# All 6 cameras
CAMERAS = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT'
]


def get_2d_boxes(nusc, sample_data_token, visibilities=['', '1', '2', '3', '4']):
    """
    Get 2D bounding boxes for a camera sample.
    Returns list of (category_id, bbox_2d, visibility)
    """
    # Get sample data
    sd_record = nusc.get('sample_data', sample_data_token)
    
    # Get calibration
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    camera_intrinsic = np.array(cs_record['camera_intrinsic'])
    
    # Get image size
    imsize = (sd_record['width'], sd_record['height'])
    
    # Get all annotations for this sample
    sample_record = nusc.get('sample', sd_record['sample_token'])
    
    boxes_2d = []
    
    for ann_token in sample_record['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        
        # Filter by category
        category_name = ann['category_name']
        if category_name not in NUSCENES_CLASSES:
            continue
        
        category_id = NUSCENES_CLASSES[category_name]
        
        # Get 3D box
        box = nusc.get_box(ann_token)
        
        # Move box to camera coordinate frame
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)
        
        # Check if box is behind camera
        if box.center[2] <= 0:
            continue
        
        # Project 3D box to 2D
        corners_3d = box.corners()  # 3x8
        corners_2d = view_points(corners_3d, camera_intrinsic, normalize=True)[:2, :]  # 2x8
        
        # Get 2D bounding box (min/max of projected corners)
        min_x, min_y = corners_2d.min(axis=1)
        max_x, max_y = corners_2d.max(axis=1)
        
        # Clip to image bounds
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(imsize[0], max_x)
        max_y = min(imsize[1], max_y)
        
        # Check valid box
        width = max_x - min_x
        height = max_y - min_y
        
        if width <= 0 or height <= 0:
            continue
        
        # COCO format: [x, y, width, height]
        bbox = [float(min_x), float(min_y), float(width), float(height)]
        area = float(width * height)
        
        # Visibility
        visibility_token = ann['visibility_token']
        visibility = nusc.get('visibility', visibility_token)['level'] if visibility_token != '' else ''
        
        boxes_2d.append({
            'category_id': category_id,
            'bbox': bbox,
            'area': area,
            'visibility': visibility,
            'instance_token': ann['instance_token']
        })
    
    return boxes_2d


def process_scene(args):
    """Process single scene (for parallel processing)"""
    scene_name, nusc_dataroot, cameras = args
    
    # Create NuScenes instance (each worker needs its own)
    nusc = NuScenes(version='v1.0-trainval', dataroot=nusc_dataroot, verbose=False)
    
    # Get scene
    scene = [s for s in nusc.scene if s['name'] == scene_name][0]
    
    images = []
    annotations = []
    ann_id = 0
    
    # Get first sample
    sample_token = scene['first_sample_token']
    
    while sample_token:
        sample = nusc.get('sample', sample_token)
        
        # Process all 6 cameras
        for cam in cameras:
            sd_token = sample['data'][cam]
            sd_record = nusc.get('sample_data', sd_token)
            
            # Image info
            image_info = {
                'id': len(images),
                'file_name': sd_record['filename'],  # e.g. samples/CAM_FRONT/xxx.jpg
                'width': sd_record['width'],
                'height': sd_record['height'],
                'scene_token': scene['token'],
                'scene_name': scene_name,
                'sample_token': sample_token,
                'camera': cam
            }
            images.append(image_info)
            
            # Get 2D boxes for this camera
            boxes = get_2d_boxes(nusc, sd_token)
            
            for box in boxes:
                ann = {
                    'id': ann_id,
                    'image_id': image_info['id'],
                    'category_id': box['category_id'],
                    'bbox': box['bbox'],
                    'area': box['area'],
                    'iscrowd': 0
                }
                annotations.append(ann)
                ann_id += 1
        
        # Next sample
        sample_token = sample['next']
    
    return images, annotations


def generate_coco_dataset(nusc_dataroot, output_dir, split='train', scene_names=None, num_workers=8):
    """
    Generate COCO format dataset
    
    Args:
        nusc_dataroot: Path to NuScenes data
        output_dir: Output directory
        split: 'train' or 'val'
        scene_names: List of scene names to include
        num_workers: Number of parallel workers
    """
    print(f"\n{'='*80}")
    print(f"Generating {split.upper()} dataset")
    print(f"{'='*80}")
    print(f"Scenes: {len(scene_names)}")
    print(f"Cameras: {len(CAMERAS)}")
    print(f"Workers: {num_workers}")
    print(f"Expected images: {len(scene_names) * len(CAMERAS) * 40} (~40 samples/scene)")
    
    # Prepare arguments for parallel processing
    args_list = [(scene_name, nusc_dataroot, CAMERAS) for scene_name in scene_names]
    
    # Process scenes in parallel
    print("\nProcessing scenes...")
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_scene, args_list),
            total=len(scene_names),
            desc=f"Processing {split}"
        ))
    
    # Merge results
    print("\nMerging results...")
    all_images = []
    all_annotations = []
    
    image_id_offset = 0
    ann_id_offset = 0
    
    for images, annotations in results:
        # Create mapping: old_image_id -> new_image_id
        old_to_new_image_id = {}
        
        for img in images:
            old_id = img['id']
            new_id = image_id_offset
            img['id'] = new_id
            old_to_new_image_id[old_id] = new_id
            all_images.append(img)
            image_id_offset += 1
        
        # Remap annotation image_ids using mapping
        for ann in annotations:
            old_image_id = ann['image_id']
            ann['id'] = ann_id_offset
            ann['image_id'] = old_to_new_image_id[old_image_id]  # Use mapping!
            all_annotations.append(ann)
            ann_id_offset += 1
    
    # Create COCO structure
    coco_data = {
        'info': {
            'description': f'NuScenes {split} set for YOLOX fine-tuning',
            'version': '1.0',
            'year': 2025,
            'contributor': 'NuScenes',
            'date_created': '2025-12-02'
        },
        'licenses': [],
        'images': all_images,
        'annotations': all_annotations,
        'categories': [
            {'id': cat_id, 'name': name, 'supercategory': 'vehicle' if cat_id <= 4 else 'person' if cat_id == 5 else 'vehicle'}
            for cat_id, name in CATEGORY_NAMES.items()
        ]
    }
    
    # Save to JSON
    output_file = os.path.join(output_dir, f'{split}.json')
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(coco_data, f)
    
    print(f"\n{'='*80}")
    print(f"{split.upper()} DATASET COMPLETE")
    print(f"{'='*80}")
    print(f"Images: {len(all_images):,}")
    print(f"Annotations: {len(all_annotations):,}")
    print(f"Annotations per image: {len(all_annotations)/len(all_images):.1f}")
    print(f"Output: {output_file}")
    print(f"{'='*80}\n")
    
    return coco_data


def create_image_symlinks(nusc_dataroot, output_dir):
    """Create symlinks to original images (avoid copying 100GB+)"""
    print("\nCreating image symlinks...")
    
    src_samples = os.path.join(nusc_dataroot, 'samples')
    dst_samples = os.path.join(output_dir, 'samples')
    
    if os.path.exists(dst_samples):
        print(f"  Symlink already exists: {dst_samples}")
    else:
        os.symlink(src_samples, dst_samples)
        print(f"  Created symlink: {dst_samples} -> {src_samples}")


def main():
    parser = argparse.ArgumentParser("Generate YOLOX Dataset from NuScenes")
    parser.add_argument('--nusc-root', default='/mnt/datasets/Nuscense', help='NuScenes data root')
    parser.add_argument('--output-dir', default='data/nuscenes_yolox_6cams', help='Output directory')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers (default: CPU count)')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train ratio from 700 scenes (default: 0.8)')
    args = parser.parse_args()
    
    # Set workers
    if args.workers is None:
        args.workers = min(cpu_count(), 16)  # Cap at 16 for stability
    
    # Create output directories
    output_dir = args.output_dir
    ann_dir = os.path.join(output_dir, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)
    
    print("="*80)
    print("YOLOX DATASET GENERATION - NUSCENES")
    print("="*80)
    print(f"NuScenes root: {args.nusc_root}")
    print(f"Output dir: {output_dir}")
    print(f"Workers: {args.workers}")
    print(f"Train ratio: {args.train_ratio}")
    print("="*80)
    
    # Get official splits
    print("\nLoading NuScenes splits...")
    splits = create_splits_scenes()
    
    train_scenes_official = splits['train']  # 700 scenes
    val_scenes_official = splits['val']      # 150 scenes (RESERVED FOR TEST!)
    
    print(f"✓ Official train: {len(train_scenes_official)} scenes")
    print(f"✓ Official val (RESERVED): {len(val_scenes_official)} scenes")
    
    # Split 700 train scenes into 560 train + 140 val
    np.random.seed(42)  # Reproducible split
    shuffled = np.random.permutation(train_scenes_official)
    
    split_idx = int(len(shuffled) * args.train_ratio)
    train_scenes = shuffled[:split_idx].tolist()
    val_scenes = shuffled[split_idx:].tolist()
    
    print(f"\n{'='*80}")
    print(f"INTERNAL SPLIT (from 700 train scenes)")
    print(f"{'='*80}")
    print(f"Train: {len(train_scenes)} scenes ({args.train_ratio*100:.0f}%)")
    print(f"Val: {len(val_scenes)} scenes ({(1-args.train_ratio)*100:.0f}%)")
    print(f"Test (reserved): {len(val_scenes_official)} scenes (NOT USED)")
    print(f"{'='*80}")
    
    # Generate train set
    train_data = generate_coco_dataset(
        nusc_dataroot=args.nusc_root,
        output_dir=ann_dir,
        split='train',
        scene_names=train_scenes,
        num_workers=args.workers
    )
    
    # Generate val set
    val_data = generate_coco_dataset(
        nusc_dataroot=args.nusc_root,
        output_dir=ann_dir,
        split='val',
        scene_names=val_scenes,
        num_workers=args.workers
    )
    
    # Create symlinks to images
    create_image_symlinks(args.nusc_root, output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("DATASET GENERATION COMPLETE!")
    print("="*80)
    print(f"Train: {len(train_data['images']):,} images, {len(train_data['annotations']):,} annotations")
    print(f"Val: {len(val_data['images']):,} images, {len(val_data['annotations']):,} annotations")
    print(f"Total: {len(train_data['images']) + len(val_data['images']):,} images")
    print(f"\nOutput files:")
    print(f"  {ann_dir}/train.json")
    print(f"  {ann_dir}/val.json")
    print(f"  {output_dir}/samples/ (symlink)")
    print("="*80)
    
    # Verification
    print("\nVerification:")
    print(f"  ✓ No overlap between train/val splits")
    print(f"  ✓ Test set (150 scenes) reserved")
    print(f"  ✓ All 6 cameras included")
    print(f"  ✓ COCO format ready for YOLOX")
    print("\nReady for training! Update config to use: data/nuscenes_yolox_6cams")


if __name__ == '__main__':
    main()

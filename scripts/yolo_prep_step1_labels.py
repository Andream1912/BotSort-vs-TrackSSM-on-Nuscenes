#!/usr/bin/env python3
"""
Create YOLO dataset for nuScenes fine-tuning.
Strategy: Convert GT annotations + create image list pointing to /mnt/datasets/Nuscense/samples/
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
import json

# Check if we have the cache files from your preprocessing
CACHE_TRAIN = Path('/mnt/datasets/Nuscense/scene_frame_mapping_train.json')
CACHE_VAL = Path('/mnt/datasets/Nuscense/scene_frame_mapping_val.json')

# Class mapping
CLASS_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
IMG_W, IMG_H = 1600, 900


def convert_bbox(x, y, w, h):
    """MOT → YOLO normalized."""
    xc = (x + w/2) / IMG_W
    yc = (y + h/2) / IMG_H
    wn = w / IMG_W
    hn = h / IMG_H
    return max(0, min(1, xc)), max(0, min(1, yc)), max(0, min(1, wn)), max(0, min(1, hn))


def main():
    base = Path('/user/amarino/tesi_project_amarino')
    mot_dir = base / 'data' / 'nuscenes_mot_6cams_interpolated'
    output_dir = base / 'data' / 'nuscenes_yolo_7classes'
    nusc_samples = Path('/mnt/datasets/Nuscense/samples')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Creating YOLO dataset from nuScenes MOT annotations")
    print("="*70)
    
    # We'll create a simpler structure:
    # - Convert all GT to YOLO labels
    # - Create train.txt and val.txt with absolute paths to images
    # - Images stay in /mnt/datasets/Nuscense/samples/CAM_*/
    
    for split in ['train', 'val']:
        split_dir = mot_dir / split
        if not split_dir.exists():
            print(f"Skipping {split} (not found)")
            continue
        
        # Output directories
        label_dir = output_dir / 'labels' / split
        label_dir.mkdir(parents=True, exist_ok=True)
        
        scenes = sorted([d for d in split_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('scene-')])
        
        print(f"\n{split.upper()}: {len(scenes)} scenes")
        
        image_paths = []
        label_count = 0
        
        for scene_dir in tqdm(scenes, desc=f"Converting {split}"):
            scene_name = scene_dir.name
            gt_file = scene_dir / 'gt' / 'gt.txt'
            
            if not gt_file.exists():
                continue
            
            # Extract camera
            if '-CAM_' in scene_name:
                camera = 'CAM_' + scene_name.split('-CAM_')[-1]
            else:
                camera = 'CAM_FRONT'
            
            # Read GT and group by frame
            frame_annots = {}
            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 10:
                        continue
                    
                    frame_id = int(parts[0])
                    x, y, w, h = map(float, parts[2:6])
                    cls = int(float(parts[7]))  # Class at index 7
                    
                    if cls not in CLASS_MAP:
                        continue
                    
                    if frame_id not in frame_annots:
                        frame_annots[frame_id] = []
                    
                    xc, yc, wn, hn = convert_bbox(x, y, w, h)
                    frame_annots[frame_id].append(f"{CLASS_MAP[cls]} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
            
            # Write labels
            for frame_id, annots in frame_annots.items():
                # Unique label name
                label_file = label_dir / f"{scene_name}_{frame_id:06d}.txt"
                
                with open(label_file, 'w') as f:
                    f.write('\n'.join(annots) + '\n')
                
                label_count += 1
                
                # For now, we'll add a placeholder image path
                # The actual mapping scene+frame → sample_data requires NuScenes API
                # But for a quick start, we can list all CAM images and match during training
                image_paths.append(f"{scene_name}_{frame_id:06d}")  # Just the ID for now
        
        print(f"  → {label_count} labels created")
        
        # Write image list (placeholder for now - you'll need to map properly)
        list_file = output_dir / f'{split}_placeholder.txt'
        with open(list_file, 'w') as f:
            f.write('\n'.join(image_paths))
        
        print(f"  → Image list: {list_file}")
    
    # Create data.yaml
    yaml_path = output_dir / 'data.yaml'
    yaml_content = f"""# nuScenes YOLO dataset
path: {output_dir}
train: labels/train
val: labels/val

nc: 7
names: ['car', 'truck', 'bus', 'trailer', 'pedestrian', 'motorcycle', 'bicycle']

# WARNING: Image paths need proper mapping via NuScenes API
# Labels are ready, but you need to create train.txt / val.txt
# with actual image paths from /mnt/datasets/Nuscense/samples/CAM_*/
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print("\n" + "="*70)
    print(f"Labels ready at: {output_dir}/labels/")
    print(f"Config: {yaml_path}")
    print("\nNEXT: Run the image mapping script to create proper train.txt / val.txt")
    print("="*70)


if __name__ == '__main__':
    main()

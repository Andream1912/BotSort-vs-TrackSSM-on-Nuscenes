#!/usr/bin/env python3
"""
Script per preparare NuScenes in formato compatibile con TrackSSM evaluation.
VERSIONE AGGIORNATA: Include tutte le 7 classi NuScenes nel GT.

Genera:
1. Detection files in formato MOT (frame, x, y, w, h, score)
2. GT files con class_id (frame, id, x, y, w, h, conf, class_id, visible, -1)
3. seqinfo.ini per ogni sequenza

Usage:
    python prepare_nuscenes_all_classes.py \
        --nusc_root /mnt/datasets/Nuscense \
        --out_root ./data/nuscenes_mot_front_7classes \
        --split val \
        --use_gt_as_det
"""

import os
import os.path as osp
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import json

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits as nusc_splits
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

# Parametri camera
IMG_W, IMG_H = 1600, 900
CAM = "CAM_FRONT"

# Mapping categorie NuScenes -> 7 classi standard (allineato a BotSort)
CATEGORY_TO_CLASS = {
    # Vehicles
    "vehicle.car": 1,
    
    "vehicle.truck": 2,
    
    "vehicle.bus.bendy": 3,
    "vehicle.bus.rigid": 3,
    
    "vehicle.trailer": 4,
    
    "vehicle.motorcycle": 6,
    
    "vehicle.bicycle": 7,
    
    # Pedestrians (tutte mappate a classe 5)
    "human.pedestrian.adult": 5,
    "human.pedestrian.child": 5,
    "human.pedestrian.construction_worker": 5,
    "human.pedestrian.police_officer": 5,
    "human.pedestrian.personal_mobility": 5,
    "human.pedestrian.stroller": 5,
    "human.pedestrian.wheelchair": 5,
}

CLASS_NAMES = {
    1: 'car',
    2: 'truck',
    3: 'bus',
    4: 'trailer',
    5: 'pedestrian',
    6: 'motorcycle',
    7: 'bicycle'
}

# Categorie valide
VALID_CATS = set(CATEGORY_TO_CLASS.keys())

def project_box_3d_to_2d(nusc: NuScenes, ann, sd):
    """Proietta bbox 3D in 2D su CAM_FRONT"""
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    pose = nusc.get('ego_pose', sd['ego_pose_token'])
    K = np.array(cs['camera_intrinsic'])

    box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))

    # global -> ego
    box.translate(-np.array(pose['translation']))
    box.rotate(Quaternion(pose['rotation']).inverse)
    # ego -> cam
    box.translate(-np.array(cs['translation']))
    box.rotate(Quaternion(cs['rotation']).inverse)

    corners = box.corners()
    if not np.any(corners[2, :] > 0):
        return None

    pts = view_points(corners, K, normalize=True)[:2, :]
    x1, y1 = pts[0].min(), pts[1].min()
    x2, y2 = pts[0].max(), pts[1].max()

    # scarta box fuori frame
    if x2 <= 0 or y2 <= 0 or x1 >= IMG_W or y1 >= IMG_H:
        return None

    # clamp
    x1 = max(0, min(IMG_W - 1, x1))
    y1 = max(0, min(IMG_H - 1, y1))
    x2 = max(0, min(IMG_W - 1, x2))
    y2 = max(0, min(IMG_H - 1, y2))
    
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None

    return [x1, y1, w, h]


def create_seqinfo(out_path, scene_name, num_frames, fps=2.0):
    """Crea file seqinfo.ini compatibile con TrackSSM"""
    content = f"""[Sequence]
name={scene_name}
imDir=img1
frameRate={fps}
seqLength={num_frames}
imWidth={IMG_W}
imHeight={IMG_H}
imExt=.jpg
"""
    with open(out_path, 'w') as f:
        f.write(content)


def process_split(nusc_root, version, split, out_root, use_gt_as_det=True, min_vis=2):
    """
    Processa uno split di NuScenes e genera structure per TrackSSM con tutte le 7 classi.
    
    Args:
        nusc_root: Path to NuScenes dataset
        version: v1.0-trainval or v1.0-test
        split: 'train' or 'val'
        out_root: Output directory
        use_gt_as_det: Se True, usa GT annotations come pseudo-detections (con score=1.0)
        min_vis: Visibility minima (0-4, 2 = ~40% visibile)
    """
    print(f"\n{'='*60}")
    print(f"Processing NuScenes {split} split (ALL 7 CLASSES)")
    print(f"{'='*60}")
    
    nusc = NuScenes(version=version, dataroot=nusc_root, verbose=True)
    
    # Seleziona scene
    if split == "val":
        scene_names = set(nusc_splits.val)
        scenes = [s for s in nusc.scene if s["name"] in scene_names]
    elif split == "train":
        val_names = set(nusc_splits.val)
        scenes = [s for s in nusc.scene if s["name"] not in val_names]
    else:
        raise ValueError("split must be 'train' or 'val'")

    # Crea directory di output
    out_info = Path(out_root) / split
    out_det_dir = Path(out_root) / "detections" / split
    out_det_dir.mkdir(parents=True, exist_ok=True)
    
    total_scenes = 0
    total_detections = 0
    class_counts = defaultdict(int)

    for scene in tqdm(scenes, desc=f"Processing {split} scenes"):
        scene_name = scene["name"]
        
        # Struttura directory per ogni sequenza
        seq_dir = out_info / scene_name
        seq_img_dir = seq_dir / "img1"
        seq_det_dir = seq_dir / "det"
        seq_gt_dir = seq_dir / "gt"
        
        seq_img_dir.mkdir(parents=True, exist_ok=True)
        seq_det_dir.mkdir(parents=True, exist_ok=True)
        seq_gt_dir.mkdir(parents=True, exist_ok=True)
        
        # Raccogli detections per frame
        frame_detections = defaultdict(list)  # frame_id -> list of [x,y,w,h,score,class_id]
        frame_gt = defaultdict(list)  # frame_id -> list of [id,x,y,w,h,class_id,visible]
        
        token = scene["first_sample_token"]
        frame_id = 1
        frame_mapping = {}  # token -> frame_id
        
        # Prima passata: raccolta detections
        while token:
            sample = nusc.get('sample', token)
            if CAM not in sample['data']:
                token = sample['next']
                frame_id += 1
                continue
                
            sd = nusc.get('sample_data', sample['data'][CAM])
            frame_mapping[token] = frame_id
            
            # Processa annotations
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                cat = ann['category_name']
                
                if cat not in VALID_CATS:
                    continue
                
                # Get class_id from category
                class_id = CATEGORY_TO_CLASS[cat]
                    
                vis = int(ann['visibility_token']) if ann['visibility_token'] else 0
                if vis < min_vis:
                    continue
                    
                bb = project_box_3d_to_2d(nusc, ann, sd)
                if bb is None:
                    continue
                    
                x, y, w, h = bb
                if w < 5 or h < 5:  # Filtro bbox troppo piccole
                    continue
                
                # Per detections: usa score=1.0 se usiamo GT
                if use_gt_as_det:
                    score = 1.0
                    frame_detections[frame_id].append([x, y, w, h, score, class_id])
                    total_detections += 1
                    class_counts[class_id] += 1
                
                # Per GT: salva con instance_token come ID + class_id
                inst_id = abs(hash(ann['instance_token'])) % 1000000
                frame_gt[frame_id].append([inst_id, x, y, w, h, class_id, vis])
            
            token = sample['next']
            frame_id += 1
        
        num_frames = frame_id - 1
        
        if num_frames == 0:
            continue
        
        # Crea seqinfo.ini
        create_seqinfo(seq_dir / "seqinfo.ini", scene_name, num_frames)
        
        # Scrivi detection files
        # Formato 1: det/det.txt (una detection per riga)
        det_file = seq_det_dir / "det.txt"
        with open(det_file, 'w') as f:
            for fid in sorted(frame_detections.keys()):
                for det in frame_detections[fid]:
                    x, y, w, h, score, cls_id = det
                    f.write(f"{fid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.6f},{cls_id}\n")
        
        # Formato 2: detections/val/scene-xxxx.txt (per frame, con class_id)
        det_frame_file = out_det_dir / f"{scene_name}.txt"
        with open(det_frame_file, 'w') as f:
            for fid in range(1, num_frames + 1):
                if fid in frame_detections:
                    dets = frame_detections[fid]
                    for det in dets:
                        x, y, w, h, score, cls_id = det
                        # Formato: frame,x,y,w,h,score,class_id
                        f.write(f"{fid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.6f},{cls_id}\n")
        
        # Scrivi GT con class_id
        # Formato MOT esteso: frame,id,x,y,w,h,conf,class_id,visible,-1
        gt_file = seq_gt_dir / "gt.txt"
        with open(gt_file, 'w') as f:
            for fid in sorted(frame_gt.keys()):
                for gt in frame_gt[fid]:
                    tid, x, y, w, h, cls_id, vis = gt
                    f.write(f"{fid},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,{cls_id},{vis},-1\n")
        
        total_scenes += 1
    
    print(f"\n[OK] Processed {total_scenes} scenes, {total_detections} total detections")
    print(f"\nClass distribution:")
    for cls_id in sorted(class_counts.keys()):
        cls_name = CLASS_NAMES[cls_id]
        count = class_counts[cls_id]
        print(f"  {cls_id}. {cls_name:12s}: {count:6d} instances")
    
    print(f"\nOutput structure:")
    print(f"  - Info dir: {out_info}")
    print(f"  - Detection files: {out_det_dir}")
    

def main():
    parser = argparse.ArgumentParser(description="Prepare NuScenes for TrackSSM evaluation (ALL 7 CLASSES)")
    parser.add_argument("--nusc_root", required=True, help="Path to NuScenes root (e.g., /mnt/datasets/Nuscense)")
    parser.add_argument("--version", default="v1.0-trainval", help="NuScenes version")
    parser.add_argument("--out_root", required=True, help="Output directory (e.g., ./data/nuscenes_mot_front_7classes)")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="Split to process")
    parser.add_argument("--use_gt_as_det", action="store_true", help="Use GT annotations as detections")
    parser.add_argument("--min_vis", type=int, default=2, help="Minimum visibility level (0-4)")
    
    args = parser.parse_args()
    
    print(f"\nConfiguration:")
    print(f"  NuScenes root: {args.nusc_root}")
    print(f"  Version: {args.version}")
    print(f"  Output root: {args.out_root}")
    print(f"  Split: {args.split}")
    print(f"  Use GT as detections: {args.use_gt_as_det}")
    print(f"  Min visibility: {args.min_vis}")
    print(f"  Classes: 7 (car, truck, bus, trailer, pedestrian, motorcycle, bicycle)")
    
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    
    process_split(
        nusc_root=args.nusc_root,
        version=args.version,
        split=args.split,
        out_root=args.out_root,
        use_gt_as_det=args.use_gt_as_det,
        min_vis=args.min_vis
    )
    
    print("\n" + "="*60)
    print("DONE! Dataset prepared with all 7 classes.")
    print("Next steps:")
    print("1. Update config to point to new dataset")
    print("2. Run tracking on all 7 classes")
    print("3. Compute per-class metrics")
    print("="*60)


if __name__ == "__main__":
    main()

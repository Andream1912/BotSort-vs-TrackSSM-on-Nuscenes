#!/usr/bin/env python3
"""
Compare TrackSSM vs BoT-SORT focusing on a SINGLE OBJECT with IDSW

This script highlights ONE specific object that:
- TrackSSM tracks continuously with same ID
- BoT-SORT loses and re-assigns ID multiple times

Usage:
    python scripts/compare_single_object_idsw.py --scene scene-0802_CAM_FRONT --ts-id 2
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Color for highlighted object (bright yellow)
HIGHLIGHT_COLOR = (0, 255, 255)  # Cyan/Yellow in BGR
OTHER_COLOR = (128, 128, 128)    # Gray for other objects

def bbox_overlap(bbox1, bbox2):
    """Calculate IoU between two bboxes"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1+w1, x2+w2)
    yi2 = min(y1+h1, y2+h2)
    
    inter_area = max(0, xi2-xi1) * max(0, yi2-yi1)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def load_tracking(file_path):
    """Load tracking results"""
    tracks = {}
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split(',')
            frame = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            conf = float(parts[6])
            
            if frame not in tracks:
                tracks[frame] = []
            tracks[frame].append({
                'id': track_id,
                'bbox': (int(x), int(y), int(w), int(h)),
                'conf': conf
            })
    return tracks

def find_matching_botsort_id(ts_bbox, bs_tracks):
    """Find BoT-SORT track that matches TrackSSM bbox"""
    best_id = None
    best_iou = 0
    
    for bs_track in bs_tracks:
        iou = bbox_overlap(ts_bbox, bs_track['bbox'])
        if iou > best_iou:
            best_iou = iou
            best_id = bs_track['id']
    
    return best_id if best_iou > 0.5 else None

def draw_frame(img, ts_tracks, bs_tracks, target_ts_id, matching_bs_id, frame_num, total_frames, idsw_count):
    """Draw comparison frame with highlighted object"""
    h, w = img.shape[:2]
    
    # Create canvases for both sides
    ts_canvas = img.copy()
    bs_canvas = img.copy()
    
    # Draw title bars
    cv2.rectangle(ts_canvas, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.rectangle(bs_canvas, (0, 0), (w, 80), (0, 0, 0), -1)
    
    cv2.putText(ts_canvas, "TrackSSM (stable)", (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(bs_canvas, "BoT-SORT (unstable)", (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    # Draw TrackSSM tracks
    target_ts_bbox = None
    for track in ts_tracks:
        is_target = track['id'] == target_ts_id
        color = HIGHLIGHT_COLOR if is_target else OTHER_COLOR
        thickness = 5 if is_target else 2
        
        x, y, w_box, h_box = track['bbox']
        cv2.rectangle(ts_canvas, (x, y), (x+w_box, y+h_box), color, thickness)
        
        # Label
        label = f"ID:{track['id']}"
        if is_target:
            label = f">>> ID:{track['id']} <<<"
            target_ts_bbox = track['bbox']
        
        font_scale = 1.0 if is_target else 0.6
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        cv2.rectangle(ts_canvas, (x, y-label_size[1]-8), 
                     (x+label_size[0]+4, y), color, -1)
        cv2.putText(ts_canvas, label, (x+2, y-4), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (0, 0, 0), 2)
    
    # Draw ID on title bar
    if target_ts_bbox:
        cv2.putText(ts_canvas, f"Following: ID {target_ts_id}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, HIGHLIGHT_COLOR, 2)
    
    # Draw BoT-SORT tracks
    for track in bs_tracks:
        is_target = matching_bs_id and track['id'] == matching_bs_id
        color = HIGHLIGHT_COLOR if is_target else OTHER_COLOR
        thickness = 5 if is_target else 2
        
        x, y, w_box, h_box = track['bbox']
        cv2.rectangle(bs_canvas, (x, y), (x+w_box, y+h_box), color, thickness)
        
        # Label
        label = f"ID:{track['id']}"
        if is_target:
            label = f">>> ID:{track['id']} <<<"
        
        font_scale = 1.0 if is_target else 0.6
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        cv2.rectangle(bs_canvas, (x, y-label_size[1]-8), 
                     (x+label_size[0]+4, y), color, -1)
        cv2.putText(bs_canvas, label, (x+2, y-4), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (0, 0, 0), 2)
    
    # Draw ID on title bar
    if matching_bs_id:
        cv2.putText(bs_canvas, f"Following: ID {matching_bs_id}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, HIGHLIGHT_COLOR, 2)
    else:
        cv2.putText(bs_canvas, f"Following: LOST!", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Combine side-by-side
    combined = np.hstack([ts_canvas, bs_canvas])
    
    # Add info bar at bottom
    bar = np.zeros((120, w*2, 3), dtype=np.uint8)
    
    # Frame counter
    cv2.putText(bar, f"Frame {frame_num}/{total_frames}", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # IDSW warning
    if matching_bs_id:
        idsw_text = f"BoT-SORT ID Switches so far: {idsw_count}"
        color = (0, 255, 255) if idsw_count > 0 else (0, 255, 0)
    else:
        idsw_text = f"BoT-SORT LOST THE TRACK!"
        color = (0, 0, 255)
    
    cv2.putText(bar, idsw_text, (20, 85), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
    
    # Message
    msg = "Same object, watch the ID number change on the right!"
    cv2.putText(bar, msg, (w*2 - 900, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    
    final = np.vstack([combined, bar])
    return final

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, required=True, help='Scene name')
    parser.add_argument('--ts-id', type=int, required=True, help='TrackSSM ID to follow')
    parser.add_argument('--output', type=str, default=None, help='Output video path')
    parser.add_argument('--fps', type=int, default=5, help='Output FPS')
    parser.add_argument('--data-dir', type=str, 
                       default='data/nuscenes_mot_front/val', help='Data directory')
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f'results/MEETING_10_DICEMBRE/02_comparison_videos/idsw_evidence_{args.scene}.mp4'
    
    # Paths
    trackssm_file = f'results/MEETING_10_DICEMBRE/04_tracking_results/trackssm_optimal/data/{args.scene}.txt'
    botsort_file = f'results/MEETING_10_DICEMBRE/04_tracking_results/botsort_bad/data/{args.scene}.txt'
    img_dir = Path(args.data_dir) / args.scene / 'img1'
    
    # Load tracking
    print(f"Loading TrackSSM results from {trackssm_file}...")
    trackssm_tracks = load_tracking(trackssm_file)
    
    print(f"Loading BoT-SORT results from {botsort_file}...")
    botsort_tracks = load_tracking(botsort_file)
    
    # Get image files
    img_files = sorted(img_dir.glob('*.jpg'))
    print(f"Found {len(img_files)} frames in {img_dir}")
    
    if len(img_files) == 0:
        print(f"❌ No images found in {img_dir}")
        return
    
    # Read first frame to get dimensions
    first_img = cv2.imread(str(img_files[0]))
    h, w = first_img.shape[:2]
    
    # Setup video writer
    output_w = w * 2
    output_h = h + 120  # Extra space for info bar
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (output_w, output_h))
    
    print(f"\nGenerating IDSW evidence video: {args.output}")
    print(f"Following TrackSSM ID: {args.ts_id}")
    print(f"Resolution: {output_w}x{output_h}, FPS: {args.fps}")
    
    # Track BoT-SORT ID switches
    prev_bs_id = None
    idsw_count = 0
    
    # Process frames
    for frame_idx, img_file in enumerate(tqdm(img_files, desc="Processing frames"), 1):
        img = cv2.imread(str(img_file))
        
        # Get tracks for this frame
        ts_frame_tracks = trackssm_tracks.get(frame_idx, [])
        bs_frame_tracks = botsort_tracks.get(frame_idx, [])
        
        # Find target object in TrackSSM
        target_ts_track = None
        for track in ts_frame_tracks:
            if track['id'] == args.ts_id:
                target_ts_track = track
                break
        
        # Find matching BoT-SORT track
        matching_bs_id = None
        if target_ts_track:
            matching_bs_id = find_matching_botsort_id(target_ts_track['bbox'], bs_frame_tracks)
            
            # Count IDSW
            if matching_bs_id:
                if prev_bs_id and prev_bs_id != matching_bs_id:
                    idsw_count += 1
                    print(f"  Frame {frame_idx}: IDSW detected! {prev_bs_id} → {matching_bs_id}")
                prev_bs_id = matching_bs_id
        
        # Draw frame
        canvas = draw_frame(img, ts_frame_tracks, bs_frame_tracks, 
                          args.ts_id, matching_bs_id, frame_idx, 
                          len(img_files), idsw_count)
        
        out.write(canvas)
    
    out.release()
    print(f"\n✅ Video saved to: {args.output}")
    print(f"   Total IDSW detected: {idsw_count}")
    print(f"   Duration: {len(img_files)/args.fps:.1f} seconds")

if __name__ == '__main__':
    main()

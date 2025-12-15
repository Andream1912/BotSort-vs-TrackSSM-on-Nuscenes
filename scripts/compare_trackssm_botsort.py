#!/usr/bin/env python3
"""
Compare TrackSSM (optimal) vs BoT-SORT (bad) side-by-side

Usage:
    python scripts/compare_trackssm_botsort.py --scene scene-0102_CAM_FRONT
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Color palette
COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), 
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
]

def get_color(track_id):
    """Get consistent color for track ID"""
    return COLORS[track_id % len(COLORS)]

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

def draw_tracks(img, tracks, title, show_trail=False):
    """Draw tracking results on image"""
    canvas = img.copy()
    
    # Draw title bar
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(canvas, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (255, 255, 255), 2)
    
    # Draw tracks
    for track in tracks:
        tid = track['id']
        x, y, w, h = track['bbox']
        color = get_color(tid)
        
        # Draw bbox
        cv2.rectangle(canvas, (x, y), (x+w, y+h), color, 2)
        
        # Draw track ID
        label = f"ID:{tid}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(canvas, (x, y-label_size[1]-4), 
                     (x+label_size[0], y), color, -1)
        cv2.putText(canvas, label, (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
    
    # Draw stats
    num_tracks = len(tracks)
    stats = f"Active Tracks: {num_tracks}"
    cv2.putText(canvas, stats, (10, canvas.shape[0]-15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return canvas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, required=True, help='Scene name')
    parser.add_argument('--output', type=str, default='comparison_video.mp4', 
                       help='Output video path')
    parser.add_argument('--fps', type=int, default=12, help='Output FPS')
    parser.add_argument('--data-dir', type=str, 
                       default='data/nuscenes_mot_front/val', help='Data directory')
    args = parser.parse_args()
    
    # Paths
    trackssm_file = f'results/TEST_BEST_CONFIG/data/{args.scene}.txt'
    botsort_file = f'results/TEST_BOTSORT_BAD/data/{args.scene}.txt'
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
    
    # Setup video writer for side-by-side comparison
    output_w = w * 2
    output_h = h + 80  # Extra space for titles and stats
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (output_w, output_h))
    
    print(f"\nGenerating comparison video: {args.output}")
    print(f"Resolution: {output_w}x{output_h}, FPS: {args.fps}")
    
    # Count total tracks
    trackssm_total = len(set(t['id'] for tracks in trackssm_tracks.values() for t in tracks))
    botsort_total = len(set(t['id'] for tracks in botsort_tracks.values() for t in tracks))
    
    print(f"\n📊 Stats:")
    print(f"  TrackSSM (optimal): {trackssm_total} unique tracks")
    print(f"  BoT-SORT (bad):     {botsort_total} unique tracks")
    print(f"  Difference:         +{botsort_total - trackssm_total} fragmentation in BoT-SORT\n")
    
    # Process frames
    for frame_idx, img_file in enumerate(tqdm(img_files, desc="Processing frames"), 1):
        img = cv2.imread(str(img_file))
        
        # Get tracks for this frame
        ts_frame_tracks = trackssm_tracks.get(frame_idx, [])
        bs_frame_tracks = botsort_tracks.get(frame_idx, [])
        
        # Draw tracks
        ts_canvas = draw_tracks(img, ts_frame_tracks, 
                               f"TrackSSM (optimal) - {trackssm_total} total tracks")
        bs_canvas = draw_tracks(img, bs_frame_tracks, 
                               f"BoT-SORT (bad params) - {botsort_total} total tracks")
        
        # Combine side-by-side
        combined = np.hstack([ts_canvas, bs_canvas])
        
        # Add comparison bar at bottom
        bar = np.zeros((80, output_w, 3), dtype=np.uint8)
        diff = botsort_total - trackssm_total
        text = f"Frame {frame_idx}/{len(img_files)} | TrackSSM: -{diff} fragmentation ({diff/botsort_total*100:.1f}% better)"
        cv2.putText(bar, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0, 255, 0), 2)
        
        final = np.vstack([combined, bar])
        
        out.write(final)
    
    out.release()
    print(f"\n✅ Video saved to: {args.output}")
    print(f"   Duration: {len(img_files)/args.fps:.1f} seconds")

if __name__ == '__main__':
    main()

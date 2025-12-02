#!/usr/bin/env python3
"""
Script per analizzare in dettaglio i momenti di ID switch in scene-1062.
Genera report visuale con bbox overlay per capire PERCHÉ avvengono gli switch.
"""

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

def load_tracks(filepath):
    """Carica tracking in formato MOT"""
    tracks = defaultdict(list)
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split(',')
            frame = int(parts[0])
            track_id = int(parts[1])
            bbox = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            tracks[frame].append({
                'id': track_id,
                'bbox': bbox,
                'conf': conf
            })
    return tracks

def compute_iou(bbox1, bbox2):
    """Calcola IoU tra due bbox [x, y, w, h]"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    b1 = [x1, y1, x1+w1, y1+h1]
    b2 = [x2, y2, x2+w2, y2+h2]
    
    xi1 = max(b1[0], b2[0])
    yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2])
    yi2 = min(b1[3], b2[3])
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    b1_area = w1 * h1
    b2_area = w2 * h2
    union_area = b1_area + b2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def draw_bbox(img, bbox, label, color, thickness=2):
    """Disegna bbox con label"""
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)
    
    # Label background
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x, y - label_size[1] - 4), (x + label_size[0], y), color, -1)
    cv2.putText(img, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def analyze_critical_frames():
    """Analizza i frame critici dove avvengono gli ID switch"""
    
    # Paths
    kalman_file = Path("results/results_prima_del_meeting/last_minute/kalman/data/scene-1062.txt")
    trackssm_file = Path("results/results_prima_del_meeting/last_minute/trackssm_finetuned/data/scene-1062.txt")
    gt_file = Path("data/nuscenes_mot_front/val/scene-1062_CAM_FRONT/gt/gt.txt")
    img_dir = Path("data/nuscenes_mot_front/val/scene-1062_CAM_FRONT/img1")
    output_dir = Path("results/FINE_TUNING_PARAMETRI/scene-1062_critical_frames")
    output_dir.mkdir(exist_ok=True)
    
    # Carica dati
    kalman = load_tracks(kalman_file)
    trackssm = load_tracks(trackssm_file)
    gt = load_tracks(gt_file)
    
    # Frame critici identificati dall'analisi
    critical_frames = {
        26: "Detection gap - entrambi perdono GT ID=501844",
        27: "Riappare - entrambi riassociano a nuovo ID",
        29: "CASCADE SWITCH START - GT 501844 e 603110",
        30: "CASCADE SWITCH - confusion tra track vicini",
        31: "CASCADE SWITCH - ultimo switch prima stabilizzazione",
        32: "Post-switch - TrackSSM bbox più precise (+21% IoU)",
        33: "Post-switch - TrackSSM bbox molto meglio (+24% IoU)",
    }
    
    print("=" * 80)
    print("GENERAZIONE VISUALIZZAZIONE FRAME CRITICI")
    print("=" * 80)
    
    analysis_data = []
    
    for frame_num in sorted(critical_frames.keys()):
        description = critical_frames[frame_num]
        
        # Carica immagine
        img_path = img_dir / f"{frame_num:06d}.jpg"
        if not img_path.exists():
            print(f"⚠️  Frame {frame_num}: immagine non trovata")
            continue
        
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        # Crea 3 canvas: GT | Kalman | TrackSSM
        canvas = np.zeros((h, w*3, 3), dtype=np.uint8)
        canvas[:, 0:w] = img.copy()
        canvas[:, w:w*2] = img.copy()
        canvas[:, w*2:w*3] = img.copy()
        
        # Titoli
        cv2.putText(canvas, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(canvas, "Kalman Filter", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(canvas, "TrackSSM Finetuned", (w*2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        
        # Frame number e descrizione
        cv2.putText(canvas, f"Frame {frame_num}", (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Descrizione (multi-line se necessario)
        words = description.split()
        line = ""
        y_offset = h - 30
        for word in words:
            test_line = line + word + " "
            if len(test_line) * 10 > w * 3 - 20:
                cv2.putText(canvas, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                line = word + " "
                y_offset += 25
            else:
                line = test_line
        cv2.putText(canvas, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Disegna GT
        frame_analysis = {
            'frame': frame_num,
            'description': description,
            'gt_objects': [],
            'kalman_matches': [],
            'trackssm_matches': []
        }
        
        for gt_obj in gt.get(frame_num, []):
            gt_bbox = gt_obj['bbox']
            gt_id = gt_obj['id']
            
            # Disegna GT bbox
            x, y, bbox_w, bbox_h = [int(v) for v in gt_bbox]
            cv2.rectangle(canvas, (x, y), (x+bbox_w, y+bbox_h), (0, 255, 0), 3)
            label = f"GT={gt_id}"
            cv2.rectangle(canvas, (x, y - 20), (x + 100, y), (0, 255, 0), -1)
            cv2.putText(canvas, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Trova best match in Kalman
            kalman_matches = []
            for k_obj in kalman.get(frame_num, []):
                iou = compute_iou(gt_bbox, k_obj['bbox'])
                if iou > 0.1:
                    kalman_matches.append((k_obj['id'], k_obj['bbox'], iou))
            kalman_matches.sort(key=lambda x: x[2], reverse=True)
            
            # Trova best match in TrackSSM
            trackssm_matches = []
            for t_obj in trackssm.get(frame_num, []):
                iou = compute_iou(gt_bbox, t_obj['bbox'])
                if iou > 0.1:
                    trackssm_matches.append((t_obj['id'], t_obj['bbox'], iou))
            trackssm_matches.sort(key=lambda x: x[2], reverse=True)
            
            # Disegna Kalman match
            if kalman_matches:
                k_id, k_bbox, k_iou = kalman_matches[0]
                x, y, bbox_w, bbox_h = [int(v) for v in k_bbox]
                color = (255, 0, 0) if k_iou > 0.7 else (0, 0, 255)  # Blu se IoU buono, rosso se scarso
                cv2.rectangle(canvas, (w + x, y), (w + x+bbox_w, y+bbox_h), color, 3)
                label = f"K={k_id} IoU={k_iou:.2f}"
                cv2.rectangle(canvas, (w + x, y - 20), (w + x + 150, y), color, -1)
                cv2.putText(canvas, label, (w + x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                frame_analysis['kalman_matches'].append({
                    'gt_id': gt_id,
                    'pred_id': k_id,
                    'iou': float(k_iou)
                })
            else:
                # No match - disegna X sulla GT bbox
                cv2.line(canvas, (w + x, y), (w + x+bbox_w, y+bbox_h), (0, 0, 255), 3)
                cv2.line(canvas, (w + x+bbox_w, y), (w + x, y+bbox_h), (0, 0, 255), 3)
                frame_analysis['kalman_matches'].append({
                    'gt_id': gt_id,
                    'pred_id': None,
                    'iou': 0.0
                })
            
            # Disegna TrackSSM match
            if trackssm_matches:
                t_id, t_bbox, t_iou = trackssm_matches[0]
                x, y, bbox_w, bbox_h = [int(v) for v in t_bbox]
                color = (0, 165, 255) if t_iou > 0.7 else (0, 0, 255)
                cv2.rectangle(canvas, (w*2 + x, y), (w*2 + x+bbox_w, y+bbox_h), color, 3)
                label = f"TS={t_id} IoU={t_iou:.2f}"
                cv2.rectangle(canvas, (w*2 + x, y - 20), (w*2 + x + 150, y), color, -1)
                cv2.putText(canvas, label, (w*2 + x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                frame_analysis['trackssm_matches'].append({
                    'gt_id': gt_id,
                    'pred_id': t_id,
                    'iou': float(t_iou)
                })
            else:
                cv2.line(canvas, (w*2 + x, y), (w*2 + x+bbox_w, y+bbox_h), (0, 0, 255), 3)
                cv2.line(canvas, (w*2 + x+bbox_w, y), (w*2 + x, y+bbox_h), (0, 0, 255), 3)
                frame_analysis['trackssm_matches'].append({
                    'gt_id': gt_id,
                    'pred_id': None,
                    'iou': 0.0
                })
            
            frame_analysis['gt_objects'].append({
                'id': gt_id,
                'bbox': [float(v) for v in gt_bbox]
            })
        
        # Salva immagine
        output_path = output_dir / f"frame_{frame_num:06d}_critical.jpg"
        cv2.imwrite(str(output_path), canvas)
        print(f"✅ Frame {frame_num}: {output_path}")
        
        analysis_data.append(frame_analysis)
    
    # Salva JSON con dati analisi
    json_path = output_dir / "critical_frames_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    print(f"\n✅ Analisi salvata in: {json_path}")
    print(f"✅ {len(critical_frames)} frame critici generati in: {output_dir}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze_critical_frames()

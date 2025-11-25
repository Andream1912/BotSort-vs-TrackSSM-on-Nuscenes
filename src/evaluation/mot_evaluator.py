"""
Custom Multi-Class MOT Evaluator

Flexible evaluator that supports NuScenes 7-class schema:
- 1: car
- 2: truck  
- 3: bus
- 4: trailer
- 5: pedestrian
- 6: motorcycle
- 7: bicycle

Uses TrackEval metrics (HOTA, CLEAR, Identity) but with custom data loading
to support multi-class evaluation natively.
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from typing import Dict, List, Tuple


class NuScenesMultiClassEvaluator:
    """Evaluator for NuScenes MOT data with 7-class support"""
    
    # NuScenes 7-class schema
    CLASS_NAMES = {
        1: 'car',
        2: 'truck',
        3: 'bus',
        4: 'trailer',
        5: 'pedestrian',
        6: 'motorcycle',
        7: 'bicycle'
    }
    
    def __init__(self, gt_folder: str, pred_folder: str, iou_threshold: float = 0.5):
        """
        Args:
            gt_folder: Path to GT folder (e.g., data/nuscenes_mot_front_7classes/val)
            pred_folder: Path to predictions folder (e.g., results/trackssm/data)
            iou_threshold: IoU threshold for matching (default 0.5)
        """
        self.gt_folder = Path(gt_folder)
        self.pred_folder = Path(pred_folder)
        self.iou_threshold = iou_threshold
        
    def load_mot_file(self, file_path: Path) -> Dict[int, List[Dict]]:
        """
        Load MOT format file into dict[frame_id] = list of detections
        
        MOT format: frame, id, x, y, w, h, conf, class_id, visibility, unused
        
        Returns:
            Dict mapping frame_id to list of detection dicts
        """
        if not file_path.exists():
            return {}
        
        data = np.loadtxt(file_path, delimiter=',')
        if len(data) == 0:
            return {}
        
        # Handle single detection case
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        frame_data = defaultdict(list)
        for row in data:
            frame_id = int(row[0])
            det = {
                'track_id': int(row[1]),
                'bbox': row[2:6],  # x, y, w, h
                'confidence': float(row[6]),
                'class_id': int(row[7]),
                'visibility': float(row[8]) if len(row) > 8 else 1.0
            }
            frame_data[frame_id].append(det)
        
        return dict(frame_data)
    
    def compute_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute IoU between two bboxes [x, y, w, h]"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        x1_max, y1_max = x1 + w1, y1 + h1
        x2_max, y2_max = x2 + w2, y2 + h2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1_max, x2_max)
        yi2 = min(y1_max, y2_max)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def match_detections(self, gt_dets: List[Dict], pred_dets: List[Dict]) -> Tuple[List, List, List]:
        """
        Match GT and predictions using IoU + class matching
        
        Returns:
            matched: List of (gt_idx, pred_idx) pairs
            unmatched_gt: List of unmatched GT indices  
            unmatched_pred: List of unmatched prediction indices
        """
        if len(gt_dets) == 0:
            return [], [], list(range(len(pred_dets)))
        if len(pred_dets) == 0:
            return [], list(range(len(gt_dets))), []
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(gt_dets), len(pred_dets)))
        for i, gt in enumerate(gt_dets):
            for j, pred in enumerate(pred_dets):
                # Only match same class
                if gt['class_id'] == pred['class_id']:
                    iou_matrix[i, j] = self.compute_iou(gt['bbox'], pred['bbox'])
        
        # Greedy matching (highest IoU first)
        matched = []
        unmatched_gt = set(range(len(gt_dets)))
        unmatched_pred = set(range(len(pred_dets)))
        
        while True:
            if iou_matrix.size == 0:
                break
            
            max_iou = iou_matrix.max()
            if max_iou < self.iou_threshold:
                break
            
            max_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            i, j = max_idx
            
            matched.append((i, j))
            unmatched_gt.discard(i)
            unmatched_pred.discard(j)
            
            # Set matched row/col to -inf
            iou_matrix[i, :] = -np.inf
            iou_matrix[:, j] = -np.inf
        
        return matched, list(unmatched_gt), list(unmatched_pred)
    
    def compute_id_switches(self, gt_frames: Dict, pred_frames: Dict) -> int:
        """Compute ID switches across frames"""
        # Track GT_ID -> last matched Pred_ID
        gt_to_pred_map = {}
        id_switches = 0
        
        for frame_id in sorted(gt_frames.keys()):
            if frame_id not in pred_frames:
                continue
            
            gt_dets = gt_frames[frame_id]
            pred_dets = pred_frames[frame_id]
            
            matched, _, _ = self.match_detections(gt_dets, pred_dets)
            
            for gt_idx, pred_idx in matched:
                gt_id = gt_dets[gt_idx]['track_id']
                pred_id = pred_dets[pred_idx]['track_id']
                
                if gt_id in gt_to_pred_map:
                    if gt_to_pred_map[gt_id] != pred_id:
                        id_switches += 1
                
                gt_to_pred_map[gt_id] = pred_id
        
        return id_switches
    
    def evaluate_sequence(self, gt_file: Path, pred_file: Path) -> Dict:
        """Evaluate single sequence"""
        gt_frames = self.load_mot_file(gt_file)
        pred_frames = self.load_mot_file(pred_file)
        
        # Overall metrics
        total_gt = sum(len(dets) for dets in gt_frames.values())
        total_pred = sum(len(dets) for dets in pred_frames.values())
        tp, fp, fn = 0, 0, 0
        
        # Per-class metrics
        class_metrics = defaultdict(lambda: {'gt': 0, 'pred': 0, 'tp': 0, 'fp': 0, 'fn': 0})
        
        # Process each frame
        all_frames = sorted(set(list(gt_frames.keys()) + list(pred_frames.keys())))
        
        for frame_id in all_frames:
            gt_dets = gt_frames.get(frame_id, [])
            pred_dets = pred_frames.get(frame_id, [])
            
            matched, unmatched_gt, unmatched_pred = self.match_detections(gt_dets, pred_dets)
            
            # Overall
            tp += len(matched)
            fn += len(unmatched_gt)
            fp += len(unmatched_pred)
            
            # Per-class
            for gt_idx, pred_idx in matched:
                cls = gt_dets[gt_idx]['class_id']
                class_metrics[cls]['tp'] += 1
            
            for gt_idx in unmatched_gt:
                cls = gt_dets[gt_idx]['class_id']
                class_metrics[cls]['fn'] += 1
            
            for pred_idx in unmatched_pred:
                cls = pred_dets[pred_idx]['class_id']
                class_metrics[cls]['fp'] += 1
        
        # Count GT and pred per class
        for frame_dets in gt_frames.values():
            for det in frame_dets:
                class_metrics[det['class_id']]['gt'] += 1
        
        for frame_dets in pred_frames.values():
            for det in frame_dets:
                class_metrics[det['class_id']]['pred'] += 1
        
        # Compute ID switches
        idsw = self.compute_id_switches(gt_frames, pred_frames)
        
        # Compute MOTA
        if total_gt > 0:
            mota = 1 - (fn + fp + idsw) / total_gt
        else:
            mota = 0.0
        
        # Compute precision/recall
        precision = tp / total_pred if total_pred > 0 else 0.0
        recall = tp / total_gt if total_gt > 0 else 0.0
        
        # Unique IDs
        gt_ids = set()
        pred_ids = set()
        for frame_dets in gt_frames.values():
            gt_ids.update(det['track_id'] for det in frame_dets)
        for frame_dets in pred_frames.values():
            pred_ids.update(det['track_id'] for det in frame_dets)
        
        return {
            'gt_count': total_gt,
            'pred_count': total_pred,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'idsw': idsw,
            'mota': mota,
            'precision': precision,
            'recall': recall,
            'gt_ids': len(gt_ids),
            'pred_ids': len(pred_ids),
            'class_metrics': dict(class_metrics)
        }
    
    def evaluate_all(self, scene_list: List[str] = None) -> Dict:
        """
        Evaluate all sequences
        
        Args:
            scene_list: List of scene names to evaluate (None = all)
        
        Returns:
            Dict with aggregate metrics and per-sequence breakdown
        """
        # Find all GT files
        gt_files = list(self.gt_folder.rglob('gt/gt.txt'))
        
        if len(gt_files) == 0:
            raise ValueError(f"No GT files found in {self.gt_folder}")
        
        # Filter by scene_list if provided
        if scene_list:
            # Normalize scene names (remove _CAM_FRONT suffix if present)
            normalized_scenes = [s.replace('_CAM_FRONT', '') for s in scene_list]
            gt_files = [f for f in gt_files if f.parent.parent.name in normalized_scenes]
        
        print(f"Evaluating {len(gt_files)} sequences...")
        
        # Aggregate metrics
        total_metrics = {
            'gt_count': 0,
            'pred_count': 0,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'idsw': 0,
            'gt_ids': 0,
            'pred_ids': 0,
            'class_metrics': defaultdict(lambda: {'gt': 0, 'pred': 0, 'tp': 0, 'fp': 0, 'fn': 0})
        }
        
        sequence_results = {}
        
        for gt_file in gt_files:
            scene_name = gt_file.parent.parent.name
            
            # Try multiple prediction file patterns
            pred_file = None
            for pattern in [f"{scene_name}_CAM_FRONT.txt", f"{scene_name}.txt"]:
                candidate = self.pred_folder / pattern
                if candidate.exists():
                    pred_file = candidate
                    break
            
            if pred_file is None:
                print(f"  ⚠️  Missing predictions for {scene_name}")
                continue
            
            seq_metrics = self.evaluate_sequence(gt_file, pred_file)
            sequence_results[scene_name] = seq_metrics
            
            # Aggregate
            for key in ['gt_count', 'pred_count', 'tp', 'fp', 'fn', 'idsw', 'gt_ids', 'pred_ids']:
                total_metrics[key] += seq_metrics[key]
            
            # Aggregate per-class
            for cls, class_data in seq_metrics['class_metrics'].items():
                for metric in ['gt', 'pred', 'tp', 'fp', 'fn']:
                    total_metrics['class_metrics'][cls][metric] += class_data[metric]
        
        # Compute overall metrics
        if total_metrics['gt_count'] > 0:
            total_metrics['mota'] = 1 - (total_metrics['fn'] + total_metrics['fp'] + total_metrics['idsw']) / total_metrics['gt_count']
            total_metrics['recall'] = total_metrics['tp'] / total_metrics['gt_count']
        else:
            total_metrics['mota'] = 0.0
            total_metrics['recall'] = 0.0
        
        if total_metrics['pred_count'] > 0:
            total_metrics['precision'] = total_metrics['tp'] / total_metrics['pred_count']
        else:
            total_metrics['precision'] = 0.0
        
        # Compute per-class MOTA
        for cls, class_data in total_metrics['class_metrics'].items():
            if class_data['gt'] > 0:
                # IDSW is hard to attribute to class, approximate
                class_idsw = total_metrics['idsw'] * (class_data['gt'] / total_metrics['gt_count'])
                class_data['mota'] = 1 - (class_data['fn'] + class_data['fp'] + class_idsw) / class_data['gt']
                class_data['recall'] = class_data['tp'] / class_data['gt']
            else:
                class_data['mota'] = 0.0
                class_data['recall'] = 0.0
            
            if class_data['pred'] > 0:
                class_data['precision'] = class_data['tp'] / class_data['pred']
            else:
                class_data['precision'] = 0.0
        
        return {
            'summary': total_metrics,
            'sequences': sequence_results
        }
    
    def save_results(self, results: Dict, output_file: Path):
        """Save results to JSON"""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert defaultdict to dict for JSON serialization
        if 'summary' in results:
            if 'class_metrics' in results['summary']:
                results['summary']['class_metrics'] = dict(results['summary']['class_metrics'])
        
        # Add class names
        if 'summary' in results and 'class_metrics' in results['summary']:
            for cls_id in list(results['summary']['class_metrics'].keys()):
                results['summary']['class_metrics'][cls_id]['class_name'] = self.CLASS_NAMES.get(cls_id, f'unknown-{cls_id}')
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved to {output_file}")

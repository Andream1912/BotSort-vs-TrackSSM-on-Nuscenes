#!/usr/bin/env python3
import argparse, os, os.path as osp
from pathlib import Path
import numpy as np, cv2
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits as nusc_splits
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

CAM = "CAM_FRONT"
IMG_W, IMG_H = 1600, 900

def box3d_to_2d(nusc, ann, sd):
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    pose = nusc.get('ego_pose', sd['ego_pose_token'])
    K = np.array(cs['camera_intrinsic'])
    box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
    box.translate(-np.array(pose['translation'])); box.rotate(Quaternion(pose['rotation']).inverse)
    box.translate(-np.array(cs['translation']));   box.rotate(Quaternion(cs['rotation']).inverse)
    c = box.corners()
    if not np.any(c[2, :] > 0): return None
    p = view_points(c, K, normalize=True)[:2, :]
    x1, y1 = p[0].min(), p[1].min()
    x2, y2 = p[0].max(), p[1].max()
    if x2 <= 0 or y2 <= 0 or x1 >= IMG_W or y1 >= IMG_H: return None
    x1 = max(0, min(IMG_W-1, x1)); y1 = max(0, min(IMG_H-1, y1))
    x2 = max(0, min(IMG_W-1, x2)); y2 = max(0, min(IMG_H-1, y2))
    if x2 <= x1 or y2 <= y1: return None
    return [x1, y1, x2-x1, y2-y1]

VALID = {
    "vehicle.car","vehicle.truck","vehicle.bus.bendy","vehicle.bus.rigid",
    "vehicle.trailer","vehicle.motorcycle","vehicle.bicycle",
    "human.pedestrian.adult","human.pedestrian.child","human.pedestrian.construction_worker",
    "human.pedestrian.police_officer","human.pedestrian.personal_mobility",
    "human.pedestrian.stroller","human.pedestrian.wheelchair",
}
CLS_TO_ID = {
    "vehicle.car":1,"vehicle.truck":1,"vehicle.bus.bendy":1,"vehicle.bus.rigid":1,"vehicle.trailer":1,
    "vehicle.motorcycle":1,"vehicle.bicycle":1,
    "human.pedestrian.adult":2,"human.pedestrian.child":2,"human.pedestrian.construction_worker":2,
    "human.pedestrian.police_officer":2,"human.pedestrian.personal_mobility":2,
    "human.pedestrian.stroller":2,"human.pedestrian.wheelchair":2,
}

def write_seqinfo(seq_dir, name, fps, nframes):
    txt = (
        "[Sequence]\n"
        f"name={name}\n"
        "imDir=img1\n"
        f"frameRate={fps:.2f}\n"
        f"seqLength={nframes}\n"
        f"imWidth={IMG_W}\n"
        f"imHeight={IMG_H}\n"
        "imExt=.jpg\n"
    )
    with open(osp.join(seq_dir, "seqinfo.ini"), "w") as f: f.write(txt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nusc_root", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--out_root", required=True)  # es: ./data/nuscenes_mot_front
    ap.add_argument("--split", default="val", choices=["val","train"])
    ap.add_argument("--copy_images", action="store_true")
    args = ap.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.nusc_root, verbose=True)
    val_names = set(nusc_splits.val)
    scenes = [s for s in nusc.scene if (s["name"] in val_names)] if args.split=="val" else [s for s in nusc.scene if (s["name"] not in val_names)]
    out_split = Path(args.out_root) / args.split
    out_split.mkdir(parents=True, exist_ok=True)

    for scene in tqdm(scenes, desc=f"export {args.split}"):
        name = scene["name"]
        seq_dir = out_split / f"{name}_{CAM}"
        img_dir = seq_dir / "img1"
        gt_dir  = seq_dir / "gt"
        img_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        # raccogli frames
        token = scene["first_sample_token"]
        frames = []
        timestamps = []
        while token:
            smp = nusc.get('sample', token)
            if CAM not in smp['data']: token = smp['next']; continue
            sd = nusc.get('sample_data', smp['data'][CAM])
            frames.append(sd["token"]); timestamps.append(sd["timestamp"])
            token = smp['next']
        fps = 12.0 if len(timestamps)<2 else round(1.0/ (np.diff(np.array(timestamps)).mean()/1e6), 2)

        # immagini (symlink) 000001.jpg ...
        for i, sd_tok in enumerate(frames, 1):
            sd = nusc.get('sample_data', sd_tok)
            src = Path(nusc.dataroot) / sd['filename']
            dst = img_dir / f"{i:06d}.jpg"
            try:
                if not dst.exists(): os.symlink(src, dst)
            except OSError:
                import shutil; shutil.copy2(src, dst)

        # GT: MOT format per frame (multi-class binaria 1=vehicle,2=pedestrian)
        lines = []
        frame_id = 1
        for sd_tok in frames:
            sd = nusc.get('sample_data', sd_tok)                 
            smp = nusc.get('sample', sd['sample_token'])  
            for ann_tok in smp['anns']:
                ann = nusc.get('sample_annotation', ann_tok)
                cat = ann['category_name']
                if cat not in VALID: continue
                vis = int(ann['visibility_token']) if ann['visibility_token'] else 0
                if vis < 2: continue
                bb = box3d_to_2d(nusc, ann, sd)
                if bb is None: continue
                x,y,w,h = bb
                cls = CLS_TO_ID[cat]
                # Nota: TrackEval/MOT vuole id numerico; qui NON abbiamo id, potremmo usare instance_token hash,
                # ma per GT format classico serve id costante: usiamo hash stabile:
                tid = abs(hash(ann['instance_token'])) % 1000000 + 1
                lines.append(f"{frame_id},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,{cls},1")
            frame_id += 1

        with open(gt_dir / "gt.txt", "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        write_seqinfo(seq_dir, f"{name}_{CAM}", fps, len(frames))

    print("[OK] MOT export:", out_split)

if __name__ == "__main__":
    main()

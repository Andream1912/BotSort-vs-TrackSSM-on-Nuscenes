#!/usr/bin/env python3
import argparse, os, os.path as osp, glob, cv2, torch
import numpy as np
from tqdm import tqdm

# usa YOLOX locale nel repo
from external.YOLOX.yolox.exp import get_exp
from external.YOLOX.yolox.utils import postprocess
from external.YOLOX.yolox.data.data_augment import ValTransform

COCO_TO_BIN = {
    0: 2,   # person -> 2 (pedestrian)
    1: 1,   # bicycle -> 1 (vehicle)
    2: 1,   # car -> 1
    3: 1,   # motorcycle -> 1
    5: 1,   # bus -> 1
    7: 1,   # truck -> 1
}

def load_model(weights, device):
    exp = get_exp(None, "yolox_x")
    model = exp.get_model()
    ckpt = torch.load(weights, map_location="cpu")
    if "model" in ckpt: ckpt = ckpt["model"]
    model.load_state_dict(ckpt, strict=False)
    model.to(device).eval()
    return model, exp

def run_seq(seq_dir, out_dir, model, exp, device, score_thr=0.3, nms_iou=0.7):
    os.makedirs(out_dir, exist_ok=True)
    img_dir = osp.join(seq_dir, "img1")
    imgs = sorted(glob.glob(osp.join(img_dir, "*.jpg")))
    preproc = ValTransform(legacy=False)
    for i, imp in enumerate(tqdm(imgs, desc=osp.basename(seq_dir)), 1):
        img = cv2.imread(imp)
        h, w = img.shape[:2]

        # preprocess
        inp, _ = preproc(img, None, exp.test_size)
        inp = torch.from_numpy(inp).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(inp)  # <<< PASSA inp QUI
            pred = postprocess(out, exp.num_classes, score_thr, nms_iou, class_agnostic=True)[0]

        lines = []
        if pred is not None:
            b = pred.cpu().numpy().copy()
            # back to original image size
            scale = min(exp.test_size[0] / float(h), exp.test_size[1] / float(w))
            b[:, :4] /= scale
            for x1, y1, x2, y2, conf, cls, _ in b:
                cls = int(cls)
                if cls not in (0,1,2,3,5,7):  # person + bicycle,car,moto,bus,truck
                    continue
                x, y, ww, hh = x1, y1, x2 - x1, y2 - y1
                if ww <= 1 or hh <= 1:
                    continue
                lines.append(f"{i},-1,{x:.2f},{y:.2f},{ww:.2f},{hh:.2f},{float(conf):.4f}")

        with open(osp.join(out_dir, f"{i:06d}.txt"), "w") as f:
            if lines:
                f.write("\n".join(lines) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_root", required=True)     # ./data/nuscenes_mot_front/val
    ap.add_argument("--det_root", required=True)     # ./data/nuscenes_mot_front/detections/val
    ap.add_argument("--weights", required=True)
    ap.add_argument("--score_thr", type=float, default=0.3)
    ap.add_argument("--nms_iou", type=float, default=0.7)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, exp = load_model(args.weights, args.device)
    seqs = sorted([d for d in os.listdir(args.val_root) if osp.isdir(osp.join(args.val_root, d))])
    for seq in seqs:
        run_seq(
            osp.join(args.val_root, seq),
            osp.join(args.det_root, seq),
            model, exp, args.device, args.score_thr, args.nms_iou
        )

if __name__ == "__main__":
    main()

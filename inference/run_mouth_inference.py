#!/usr/bin/env python3
# inference/run_mouth_inference.py
"""
Score frames with a trained mouth model and write probs + index mapping.
"""

from pathlib import Path
import argparse
import csv
import time
import json
import sys

import numpy as np
from PIL import Image
import cv2  # <-- added for CLAHE

import torch
from torch.utils.data import Dataset, DataLoader

# Ensure repo root is on sys.path so imports like `event_training` resolve
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from event_training.training.train_mouth_model import ResNet18EarlyFusion

torch.set_num_threads(1)

# -------------------------
# Dataset for inference
# -------------------------
class InferenceCandDataset(Dataset):
    """
    Load candidate rows (video, frame) and produce model input tensors.
    Reuses the same image loading and transforms used in training.
    """
    def __init__(self, rows, csv_path, frames_root, resize=224):
        from event_training.training.train_mouth_model import MouthDataset
        self._helper = MouthDataset(
            csv_path,
            frames_root=frames_root,
            resize=resize,
            split="train",
            train=False
        )
        self.rows = rows

        # --- CLAHE (must match ROI detector + MouthDataset) ---
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.rows)

    def _load_rgb_clahe(self, p: Path):
        if not p.exists():
            blank = np.zeros((self._helper.resize, self._helper.resize), dtype=np.uint8)
            blank = cv2.cvtColor(blank, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(blank)

        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            gray = np.zeros((self._helper.resize, self._helper.resize), dtype=np.uint8)

        # Apply CLAHE normalization
        gray = self.clahe.apply(gray)

        # Convert back to RGB for transforms
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)

    def __getitem__(self, idx):
        r = self.rows[idx]
        video = r["video"]
        frame = int(r["frame"])

        crop_p, motion_p = self._helper._resolve_paths(video, frame)

        # --- CLAHE-normalized RGB ---
        rgb = self._load_rgb_clahe(crop_p)

        # Motion map (unchanged)
        motion = self._helper._load_motion(motion_p)
        motion_pil = Image.fromarray((np.clip(motion, 0, 1) * 255).astype("uint8"))

        # Deterministic transforms
        rgb_geo = self._helper.geo_transform(rgb)
        motion_geo = self._helper.geo_transform(motion_pil)

        if self._helper.color_transform is not None:
            rgb_geo = self._helper.color_transform(rgb_geo)

        rgb_t = self._helper.to_tensor(rgb_geo)
        rgb_t = self._helper.rgb_norm(rgb_t)
        motion_t = self._helper.to_tensor(motion_geo)

        input_t = torch.cat([rgb_t, motion_t], dim=0)
        return input_t, {"video": video, "frame": frame}

# -------------------------
# Model loader
# -------------------------
def load_model_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ResNet18EarlyFusion(pretrained=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# -------------------------
# Helper: build labels aligned to mapping
# -------------------------
def build_and_save_val_labels(mapping_csv_path, gt_csv_path, out_npy_path, split="val"):
    gt_csv_path = Path(gt_csv_path)
    mapping_csv_path = Path(mapping_csv_path)
    out_npy_path = Path(out_npy_path)

    gt = {}
    with gt_csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            if (r.get("split") or "").strip().lower() != split.lower():
                continue
            v = (r.get("video") or "").strip()
            if not v:
                continue
            f = int(float(r.get("frame") or 0))
            lab = r.get("label")
            gt[(v, f)] = 1 if str(lab).strip().lower() in ("1","1.0","pos","positive","true","t") else 0

    labels = []
    with mapping_csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            v = (r.get("video") or "").strip()
            f = int(float(r.get("frame") or 0))
            labels.append(gt.get((v, f), 0))

    arr = np.array(labels, dtype=np.int8)
    np.save(out_npy_path, arr)
    print(f"Wrote labels npy: {out_npy_path} shape={arr.shape}")
    return arr

# -------------------------
# Inference runner
# -------------------------
def run_inference(ckpt, csv_path, frames_root, out_dir, batch_size, num_workers, device, split):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            if (r.get("split") or "").strip().lower() != split.lower():
                continue
            video = (r.get("video") or "").strip()
            if not video:
                continue
            frame = int(float(r.get("frame") or 0))
            rows.append({"video": video, "frame": frame})

    if len(rows) == 0:
        raise RuntimeError(f"No rows found for split='{split}' in CSV: {csv_path}")

    device = torch.device(device)
    model = load_model_from_ckpt(ckpt, device)

    ds = InferenceCandDataset(rows, csv_path, frames_root)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: (
            torch.stack([item[0] for item in b], dim=0),
            [item[1] for item in b]
        )
    )

    probs = []
    index_rows = []
    total = 0
    t_start = time.time()

    with torch.no_grad():
        for batch_idx, (x, metas) in enumerate(dl, start=1):
            batch_t0 = time.time()
            x = x.to(device)
            logits = model(x)
            p = torch.sigmoid(logits).cpu().numpy().ravel()
            probs.append(p)
            for m in metas:
                index_rows.append({"video": m["video"], "frame": int(m["frame"])})
            total += x.shape[0]
            print(f"[{batch_idx}] scored {x.shape[0]} items time_s={(time.time()-batch_t0):.3f}")

    elapsed = time.time() - t_start
    print(f"Scored total {total} rows in {elapsed:.1f}s")

    probs_arr = np.concatenate(probs, axis=0)
    np.save(out_dir / "probs.npy", probs_arr)

    # Build GT lookup
    gt = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            if (r.get("split") or "").strip().lower() != split.lower():
                continue
            v = (r.get("video") or "").strip()
            if not v:
                continue
            f = int(float(r.get("frame") or 0))
            lab = r.get("label")
            gt[(v, f)] = 1 if str(lab).strip().lower() in ("1","1.0","pos","positive","true","t") else 0

    mapping_path = out_dir / "index_to_row.csv"
    with open(mapping_path, "w", encoding="utf-8", newline="") as fh:
        fieldnames = ["index", "video", "frame", "label"]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(index_rows):
            lbl = gt.get((r["video"], r["frame"]), 0)
            w.writerow({"index": i, "video": r["video"], "frame": r["frame"], "label": lbl})

    val_labels_path = out_dir / "val_labels.npy"
    try:
        arr = build_and_save_val_labels(mapping_path, csv_path, val_labels_path, split=split)
        if probs_arr.shape[0] != arr.shape[0]:
            print("WARNING: probs length and labels length differ:", probs_arr.shape, arr.shape)
        else:
            print("Verified: probs and val_labels lengths match:", probs_arr.shape[0])
    except Exception as e:
        print("Failed to build val_labels.npy:", e)

    meta = {
        "ckpt": str(ckpt),
        "csv": str(csv_path),
        "frames_root": str(frames_root),
        "rows_scored": int(total),
        "elapsed_s": float(elapsed),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "device": str(device),
        "out_dir": str(out_dir)
    }
    with open(out_dir / "inference_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Wrote probs.npy ({probs_arr.shape}) and index_to_row.csv to {out_dir}")

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--frames-root", default="runs/inference")
    p.add_argument("--out-dir", default="runs/train_mouth_2/inference")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--split", default="val")
    return p.parse_args()

def main():
    args = parse_args()
    run_inference(
        ckpt=Path(args.ckpt),
        csv_path=Path(args.csv),
        frames_root=Path(args.frames_root),
        out_dir=Path(args.out_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        split=args.split
    )

if __name__ == "__main__":
    main()

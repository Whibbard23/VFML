#!/usr/bin/env python3
# event_training/datasets/mouth_dataset.py
"""
Mouth dataset that returns a 4-channel tensor (RGB + motion).
Expected layout per video stem:
  <frames_root>/<stem>_roi/crops/crop_000000.jpg
  <frames_root>/<stem>_roi/labels/motion/motion_000000.npy

CSV must contain columns: video,frame,label,split
"""

from pathlib import Path
import csv
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2  # <-- needed for CLAHE

def _default_joint_transforms(resize=224, train=True):
    if train:
        geo = transforms.RandomResizedCrop(resize, scale=(0.9, 1.0))
        color = transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.02, hue=0.01)
        return geo, color
    else:
        geo = transforms.Resize((resize, resize))
        return geo, None

class MouthDataset(Dataset):
    """
    CSV rows: video, frame, label, split
    video: basename or path (stem used to resolve folders)
    frame: integer frame index
    label: 0 or 1
    split: train or val
    """
    def __init__(self, csv_path, frames_root="runs/inference", resize=224, split="train", train=True):
        self.csv_path = Path(csv_path)
        self.frames_root = Path(frames_root)
        self.resize = int(resize)
        self.split = split
        self.train = bool(train)

        # Load rows
        self.rows = []
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
            rdr = csv.DictReader(fh)
            for r in rdr:
                row_split = (r.get("split") or "").strip().lower()
                if row_split != self.split:
                    continue
                video = (r.get("video") or "").strip()
                if not video:
                    continue
                frame = int(float(r.get("frame") or 0))
                label = int(float(r.get("label") or 0))
                self.rows.append({"video": video, "frame": frame, "label": label})

        # Transforms
        self.geo_transform, self.color_transform = _default_joint_transforms(
            resize=self.resize, train=self.train
        )
        self.to_tensor = transforms.ToTensor()
        self.rgb_norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # --- CLAHE (must match ROI detector) ---
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.rows)

    def _resolve_paths(self, video, frame_idx):
        stem = Path(video).stem
        crop_path = self.frames_root / f"{stem}_roi" / "crops" / f"crop_{frame_idx:06d}.jpg"
        motion_path = self.frames_root / f"{stem}_roi" / "labels" / "motion" / f"motion_{frame_idx:06d}.npy"
        return crop_path, motion_path

    def _load_rgb(self, p: Path):
        if not p.exists():
            # fallback: blank image
            blank = np.zeros((self.resize, self.resize), dtype=np.uint8)
            blank = cv2.cvtColor(blank, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(blank)

        # Load grayscale for CLAHE
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            gray = np.zeros((self.resize, self.resize), dtype=np.uint8)

        # Apply CLAHE normalization
        gray = self.clahe.apply(gray)

        # Convert back to RGB for transforms
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)

    def _load_motion(self, p: Path):
        if not p.exists():
            return np.zeros((self.resize, self.resize), dtype=np.float32)
        m = np.load(p)
        if m.ndim == 2:
            return m.astype(np.float32)
        return np.squeeze(m).astype(np.float32)

    def __getitem__(self, idx):
        row = self.rows[idx]
        video = row["video"]
        frame_idx = row["frame"]
        label = float(row["label"])

        crop_path, motion_path = self._resolve_paths(video, frame_idx)

        # Load CLAHE-normalized RGB crop
        rgb_pil = self._load_rgb(crop_path)

        # Load motion map
        motion_np = self._load_motion(motion_path)
        motion_pil = Image.fromarray((np.clip(motion_np, 0, 1) * 255).astype("uint8"))

        # Joint geometric transform
        rgb_geo = self.geo_transform(rgb_pil)
        motion_geo = self.geo_transform(motion_pil)

        # Photometric jitter only on RGB
        if self.color_transform is not None:
            rgb_geo = self.color_transform(rgb_geo)

        # Convert to tensors
        rgb_t = self.to_tensor(rgb_geo)
        rgb_t = self.rgb_norm(rgb_t)

        motion_t = self.to_tensor(motion_geo)  # 1xHxW

        # Combine into 4-channel tensor
        input_t = torch.cat([rgb_t, motion_t], dim=0)
        return input_t, torch.tensor(label, dtype=torch.float32)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    ds = MouthDataset(args.csv, split=args.split, train=True)
    print("Dataset length:", len(ds))
    x, y = ds[0]
    print("Sample shapes:", x.shape, y)

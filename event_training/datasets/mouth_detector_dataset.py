"""
event_training/datasets/mouth_detector_dataset.py

Dataset that:
- reads an events CSV (expects at least columns: 'video', 'frame', optionally 'matched' and 'crop_path')
- derives the detection target frame as one frame after the CSV 'frame' value
- centers returned clips on the BEFORE_ONSET frame (i.e., detection_frame - 1)
- always returns numeric/sanitized meta fields so DataLoader collate will not fail

Notes:
- This implementation reads frames from the source video file using OpenCV.
- `data_root` should be the project root (or the folder that contains the video files referenced by the CSV).
- `clip_len` must be odd to produce a symmetric window around the center frame; if even, the center is chosen as floor(clip_len/2).
- If a requested frame cannot be read, the dataset returns a zero image for that frame (keeps dtypes consistent).
"""

from typing import Optional, Dict, Any, List
import os
import math

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import pandas as pd


def _read_frame_from_video(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """
    Read a single frame (BGR) from a video using OpenCV.
    Returns None if the frame cannot be read.
    """
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    # clamp frame index
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total == 0:
        cap.release()
        return None
    if frame_idx < 0:
        cap.release()
        return None
    if frame_idx >= total:
        # clamp to last frame
        frame_idx = total - 1
    # seek and read
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame  # BGR numpy array


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR numpy array to PIL RGB Image."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


class MouthDetectorDataset(Dataset):
    """
    Dataset that returns (clip_tensor, label_tensor, meta_dict)

    clip_tensor: torch.FloatTensor, shape [T, C, H, W]
    label_tensor: torch.FloatTensor scalar (0.0 or 1.0)
    meta_dict: {
        "video": str,
        "center_idx": int,            # frame index used as center (before_onset)
        "event_frame_index": int,     # detection target frame (before_onset + 1)
        "before_onset": int           # same as center_idx (explicit)
    }
    """

    def __init__(
        self,
        csv_path: str,
        data_root: str = ".",
        clip_len: int = 5,
        transform=None,
        video_exts: Optional[List[str]] = None,
    ):
        assert clip_len >= 1, "clip_len must be >= 1"
        self.clip_len = clip_len
        self.half = clip_len // 2
        self.transform = transform
        self.data_root = data_root
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        # normalize column names to lower-case for robustness
        self.df.columns = [c.strip() for c in self.df.columns]
        if video_exts is None:
            video_exts = [".avi", ".mp4", ".mov", ".mkv"]
        self.video_exts = video_exts

    def __len__(self):
        return len(self.df)

    def _resolve_video_path(self, video_name: str) -> Optional[str]:
        """
        Resolve a video filename from the CSV to an actual file path under data_root.
        If the CSV already contains a path, prefer that. Otherwise search common extensions.
        """
        # if video_name is already a path
        candidate = os.path.join(self.data_root, video_name)
        if os.path.exists(candidate):
            return candidate
        # try with extensions
        base = os.path.splitext(video_name)[0]
        for ext in self.video_exts:
            p = os.path.join(self.data_root, base + ext)
            if os.path.exists(p):
                return p
        # try in a 'videos' subfolder
        for ext in self.video_exts:
            p = os.path.join(self.data_root, "videos", base + ext)
            if os.path.exists(p):
                return p
        return None

    def _read_clip(self, video_path: str, center_frame: int) -> torch.Tensor:
        """
        Read clip_len frames centered at center_frame (which is the BEFORE_ONSET frame).
        If a frame cannot be read, substitute a zero image of the same size as the first successful frame,
        or a default 128x128 black image if none succeed.
        Returns a tensor of shape [T, C, H, W] (float32, 0..1).
        """
        frames = []
        first_size = None
        for i in range(self.clip_len):
            fidx = center_frame - self.half + i
            bgr = _read_frame_from_video(video_path, fidx)
            if bgr is None:
                frames.append(None)
            else:
                pil = _bgr_to_pil(bgr)
                frames.append(pil)
                if first_size is None:
                    first_size = pil.size  # (W, H)

        # If no frames were read successfully, create black frames of 128x128
        if first_size is None:
            first_size = (128, 128)
            frames = [Image.new("RGB", first_size, (0, 0, 0)) for _ in range(self.clip_len)]
        else:
            # replace None entries with black images of first_size
            frames = [f if f is not None else Image.new("RGB", first_size, (0, 0, 0)) for f in frames]

        # apply transform if provided, otherwise default: resize to 128x128 and convert to tensor
        processed = []
        for pil in frames:
            if self.transform is not None:
                out = self.transform(pil)
            else:
                # default transform: resize + to tensor (C,H,W) float32 0..1
                pil2 = pil.resize((128, 128))
                arr = np.array(pil2).astype("float32") / 255.0
                # H,W,3 -> C,H,W
                out = torch.from_numpy(arr).permute(2, 0, 1)
            processed.append(out)

        # stack into [T, C, H, W]
        clip = torch.stack(processed, dim=0)
        return clip

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # CSV fields we expect: 'video' and 'frame'
        video_name = row.get("video") or row.get("Video") or row.get("video_name")
        csv_frame = None
        if "frame" in row and not pd.isna(row["frame"]):
            try:
                csv_frame = int(row["frame"])
            except Exception:
                csv_frame = None

        # Derive detection target: detection_frame = csv_frame + 1 (one later than before_onset)
        # The BEFORE_ONSET frame (center) is csv_frame (we will center the clip on this)
        before_onset = csv_frame if csv_frame is not None else -1
        detection_frame = before_onset + 1 if before_onset >= 0 else -1

        # Resolve video path
        video_path = None
        # if CSV contains a crop_path or full path, prefer that
        if "crop_path" in row and isinstance(row["crop_path"], str) and row["crop_path"].strip():
            candidate = os.path.join(self.data_root, row["crop_path"])
            if os.path.exists(candidate):
                video_path = candidate
        if video_path is None and isinstance(video_name, str):
            video_path = self._resolve_video_path(video_name)

        # Build label: prefer 'matched' column if present, else fallback to event_type heuristics
        matched = row.get("matched", None)
        if pd.notna(matched):
            try:
                label_val = 1.0 if bool(matched) else 0.0
            except Exception:
                label_val = 0.0
        else:
            # fallback: if event_type exists and is not 'none', mark positive
            et = row.get("event_type", "")
            label_val = 1.0 if isinstance(et, str) and et.strip() != "" else 0.0

        # If video_path is missing, we cannot read frames; create a zero clip
        if video_path is None:
            # create zero clip
            clip = torch.zeros((self.clip_len, 3, 128, 128), dtype=torch.float32)
        else:
            # center the clip on the BEFORE_ONSET frame (so the model sees the frames leading up to detection_frame)
            center_idx = before_onset if before_onset >= 0 else 0
            clip = self._read_clip(video_path, center_idx)

        # Build meta and sanitize values so collate never sees None
        meta: Dict[str, Any] = {}
        meta["video"] = video_name if isinstance(video_name, str) else ""
        meta["center_idx"] = int(center_idx) if isinstance(center_idx, (int, np.integer)) and center_idx >= 0 else -1
        meta["event_frame_index"] = int(detection_frame) if detection_frame >= 0 else -1
        meta["before_onset"] = int(before_onset) if before_onset >= 0 else -1

        # Ensure label is a float tensor
        label_tensor = torch.tensor(float(label_val), dtype=torch.float32)

        return clip, label_tensor, meta

#!/usr/bin/env python3
"""
Run dense per-frame inference over a set of videos and write predictions CSV.

Usage example:
  python event_training/scripts/generate_frame_probs.py \
    --model-ckpt event_training/checkpoints/best_mouth_event.pt \
    --split-csv event_csvs/assembly_1_val_events.csv \
    --smoothed-root runs/inference \
    --video-root "W:\ADStudy\VF AD Blinded\Early Tongue Training" \
    --out-csv event_csvs/frame_probs_val.csv

Notes:
  - The script reads the split CSV to determine which videos to process (video column).
  - For each video it reads runs/inference/<stem>_roi/labels/smoothed/<stem>_mouth.txt to get per-frame ROI coords.
  - It opens the video file, iterates every frame index, crops using normalized ROI (last_valid fallback),
    resizes to model input size, runs the model on CPU, and writes rows: video,frame,prob.
  - Designed for CPU; progress printed per video.

  usage:    # activate venv first
$videoRoot = 'W:\ADStudy\VF AD Blinded\Early Tongue Training'
python event_training/inference/mouth_generate_frame_probs.py `
  --model-ckpt event_training/checkpoints/best_mouth_event_YYYYMMDD_HHMMSS.pt `
  --split-csv event_csvs/assembly_1_val_events.csv `
  --smoothed-root runs/inference `
  --video-root "$videoRoot" `
  --out-csv event_csvs/frame_probs_val.csv `
  --input-size 224 --device cpu

"""
from pathlib import Path
import argparse
import csv
import cv2
import torch
import numpy as np
from torchvision import transforms
from datetime import datetime
from event_training.legacy.mouth_dataset import MouthOnsetDataset  # for reference; not used directly

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-ckpt", required=True, help="Path to model checkpoint (.pt) with state_dict")
    p.add_argument("--split-csv", required=True, help="CSV listing videos to process (uses 'video' column)")
    p.add_argument("--smoothed-root", default="runs/inference", help="root where <video>_roi/labels/smoothed/<video>_mouth.txt live")
    p.add_argument("--video-root", default=".", help="root to resolve video paths if split CSV uses basenames")
    p.add_argument("--out-csv", default="event_csvs/frame_probs_val.csv", help="output CSV path")
    p.add_argument("--input-size", type=int, default=224, help="model input size (square)")
    p.add_argument("--device", default="cpu", help="device (cpu only recommended)")
    return p.parse_args()

def read_split_videos(split_csv_path: Path):
    vids = []
    with split_csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            v = (r.get("video") or "").strip().strip('"').strip("'")
            if v and v not in vids:
                vids.append(v)
    return vids

def read_smoothed_mouth(smoothed_path: Path):
    lines = []
    if not smoothed_path.exists():
        return None
    with smoothed_path.open("r", encoding="utf-8-sig") as fh:
        for L in fh:
            s = L.strip()
            if not s:
                lines.append(None)
            else:
                parts = s.split()
                if len(parts) >= 4:
                    try:
                        xc = float(parts[0]); yc = float(parts[1]); w = float(parts[2]); h = float(parts[3])
                        lines.append((xc, yc, w, h))
                    except Exception:
                        lines.append(None)
                else:
                    lines.append(None)
    return lines

def resolve_video_path(video_name: str, video_root: Path):
    p = Path(video_name)
    if p.is_absolute():
        return p
    return (video_root / video_name).resolve()

def video_frame_count_and_size(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None, None, None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count, width, height

def crop_from_norm(frame_bgr, xc, yc, w, h):
    H, W, _ = frame_bgr.shape
    xc_p = xc * W
    yc_p = yc * H
    bw = w * W
    bh = h * H
    x1 = int(round(xc_p - bw / 2)); y1 = int(round(yc_p - bh / 2))
    x2 = int(round(xc_p + bw / 2)); y2 = int(round(yc_p + bh / 2))
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W - 1, x2); y2 = min(H - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return frame_bgr
    return frame_bgr[y1:y2, x1:x2, :]

def build_model_from_ckpt(ckpt_path, device, input_size):
    # assumes checkpoint contains state_dict for a resnet18 with fc -> 1
    from torchvision import models
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    ck = torch.load(ckpt_path, map_location=device)
    # ck may be a dict with 'model_state_dict' or a raw state_dict
    if isinstance(ck, dict) and "model_state_dict" in ck:
        state = ck["model_state_dict"]
    else:
        state = ck
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def main():
    args = parse_args()
    device = torch.device(args.device)
    split_csv = Path(args.split_csv)
    smoothed_root = Path(args.smoothed_root)
    video_root = Path(args.video_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    videos = read_split_videos(split_csv)
    if not videos:
        raise RuntimeError(f"No videos found in split CSV: {split_csv}")

    model = build_model_from_ckpt(args.model_ckpt, device, args.input_size)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
    ])

    total_rows = 0
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["video","frame","prob"])
        for v in videos:
            stem = Path(v).stem
            smoothed_path = smoothed_root / f"{stem}_roi" / "labels" / "smoothed" / f"{stem}_mouth.txt"
            smoothed = read_smoothed_mouth(smoothed_path)
            video_path = resolve_video_path(v, video_root)
            if not video_path.exists():
                print(f"Missing video file: {video_path}  (skipping)")
                continue
            frame_count, vid_w, vid_h = video_frame_count_and_size(video_path)
            if frame_count is None:
                print(f"Cannot open video: {video_path}  (skipping)")
                continue
            if smoothed is None:
                print(f"Missing smoothed mouth file for {v}: {smoothed_path}  (skipping)")
                continue
            if len(smoothed) < frame_count:
                smoothed += [None] * (frame_count - len(smoothed))

            cap = cv2.VideoCapture(str(video_path))
            last_valid = None
            for fi in range(frame_count):
                entry = smoothed[fi]
                if entry is None:
                    if last_valid is None:
                        # no crop available; skip this frame
                        continue
                    xc, yc, w, h = last_valid
                else:
                    xc, yc, w, h = entry
                    last_valid = (xc, yc, w, h)

                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if not ret:
                    continue
                # no contrast standardization (match ROI detector)
                crop = crop_from_norm(frame, xc, yc, w, h)
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                img_t = transform(crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(img_t)
                    prob = float(torch.sigmoid(logits).cpu().numpy().ravel()[0])
                writer.writerow([v, fi, f"{prob:.6f}"])
                total_rows += 1
            cap.release()
            print(f"Processed video {v}: wrote {total_rows} rows so far")

    print(f"Wrote predictions to: {out_csv} (total rows: {total_rows})")

if __name__ == "__main__":
    main()

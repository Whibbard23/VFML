#!/usr/bin/env python3
"""
Precompute Farneback motion magnitude maps inside smoothed mouth ROI.

Usage:
  python tools/mouth_tools/precompute_motion_maps.py \
    --videos-csv event_csvs/assembly_1_video_splits.csv \
    --smoothed-root runs/inference \
    --video-root "W:\ADStudy\VF AD Blinded\Early Tongue Training" \
    --out-root runs/inference \
    --resize 224 \
    --workers 4
"""
from pathlib import Path
import argparse
import numpy as np
import cv2
from multiprocessing import Pool
import functools
import sys

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos-csv", required=True)
    p.add_argument("--smoothed-root", default="runs/inference")
    p.add_argument("--video-root", default=".")
    p.add_argument("--out-root", default="runs/inference")
    p.add_argument("--resize", type=int, default=224, help="resize motion maps to this square size")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=None, help="optional cap per video for quick tests")
    return p.parse_args()

def read_video_list(csv_path: Path):
    import csv
    vids = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rdr = csv.DictReader(fh)
        if rdr.fieldnames:
            rdr.fieldnames = [fn.strip().lstrip("\ufeff") for fn in rdr.fieldnames]
        for r in rdr:
            v = (r.get("video") or "").strip()
            if v:
                vids.append(v)
    return vids

def read_smoothed(smoothed_path: Path):
    if not smoothed_path.exists():
        return None
    lines = []
    with smoothed_path.open("r", encoding="utf-8-sig") as fh:
        for L in fh:
            s = L.strip()
            if not s:
                lines.append(None)
            else:
                parts = s.split()
                if len(parts) >= 4:
                    try:
                        xc, yc, w, h = map(float, parts[:4])
                        lines.append((xc, yc, w, h))
                    except Exception:
                        lines.append(None)
                else:
                    lines.append(None)
    return lines

def process_video(video_name, args):
    video_root = Path(args.video_root)
    smoothed_root = Path(args.smoothed_root)
    out_root = Path(args.out_root)
    resize = args.resize
    max_frames = args.max_frames

    video_path = (Path(video_name) if Path(video_name).is_absolute() else (video_root / video_name)).resolve()
    stem = video_path.stem
    smoothed_path = smoothed_root / f"{stem}_roi" / "labels" / "smoothed" / f"{stem}_mouth.txt"
    motion_dir = out_root / f"{stem}_roi" / "labels" / "motion"
    motion_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        return {"video": str(video_path), "status": "missing_video"}

    smoothed = read_smoothed(smoothed_path)
    if smoothed is None:
        return {"video": str(video_path), "status": "missing_smoothed"}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video": str(video_path), "status": "open_error"}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        frame_count = min(frame_count, max_frames)

    # read first frame
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return {"video": str(video_path), "status": "read_error_first_frame"}
    prev_gray_full = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    for fi in range(1, frame_count):
        ret, cur = cap.read()
        if not ret:
            break
        cur_gray_full = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

        # determine ROI for this frame (use last_valid fallback)
        entry_prev = smoothed[fi-1] if (fi-1) < len(smoothed) else None
        entry_cur = smoothed[fi] if fi < len(smoothed) else None
        entry = entry_cur or entry_prev
        if entry is None:
            # write an empty file to indicate no ROI
            np.save(motion_dir / f"motion_{fi:06d}.npy", np.zeros((resize, resize), dtype=np.float32))
            prev_gray_full = cur_gray_full
            continue

        xc, yc, w, h = entry
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # convert normalized to pixel box
        cx = int(round(xc * W)); cy = int(round(yc * H))
        bw = int(round(w * W)); bh = int(round(h * H))
        x1 = max(0, cx - bw//2); y1 = max(0, cy - bh//2)
        x2 = min(W, cx + bw//2); y2 = min(H, cy + bh//2)
        if x2 <= x1 or y2 <= y1:
            np.save(motion_dir / f"motion_{fi:06d}.npy", np.zeros((resize, resize), dtype=np.float32))
            prev_gray_full = cur_gray_full
            continue

        prev_roi = prev_gray_full[y1:y2, x1:x2]
        cur_roi  = cur_gray_full[y1:y2, x1:x2]

        # Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_roi, cur_roi, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        mag = cv2.GaussianBlur(mag, (5,5), 0)

        # normalize per-frame by 98th percentile to reduce outlier influence
        p98 = np.percentile(mag, 98) if mag.size else 1.0
        if p98 <= 0:
            norm = mag
        else:
            norm = mag / (p98 + 1e-6)
        norm = np.clip(norm, 0.0, 1.0)

        # resize to target square
        norm_resized = cv2.resize(norm, (resize, resize), interpolation=cv2.INTER_LINEAR)
        np.save(motion_dir / f"motion_{fi:06d}.npy", norm_resized.astype(np.float32))

        prev_gray_full = cur_gray_full

    cap.release()
    return {"video": str(video_path), "status": "done", "frames": frame_count}

def main():
    args = parse_args()
    vids = read_video_list(Path(args.videos_csv))
    if not vids:
        print("No videos found in CSV.")
        return

    if args.workers > 1:
        with Pool(args.workers) as p:
            fn = functools.partial(process_video, args=args)
            results = p.map(fn, vids)
    else:
        results = [process_video(v, args) for v in vids]

    for r in results:
        print(r)

if __name__ == "__main__":
    main()

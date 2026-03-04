#!/usr/bin/env python3
"""
Extract mouth crops and precompute Farneback motion magnitude maps.

Assumes per-frame annotated images live at:
  runs/inference/<video_stem>_roi/annotated/<video_stem>/frame000000.jpg

Outputs (per video):
  runs/inference/<video_stem>_roi/crops/crop_000000.jpg
  runs/inference/<video_stem>_roi/labels/motion/motion_000000.npy

Usage example:
  python tools\mouth_tools\extract_crops_and_motion.py \
    --videos-csv event_csvs/assembly_1_video_splits.csv \
    --frames-root runs/inference \
    --smoothed-root runs/inference \
    --out-root runs/inference \
    --resize 224 \
    --workers 1 \
    --max-frames 1000
"""
from pathlib import Path
import argparse
import csv
import numpy as np
import cv2
import time
from multiprocessing import Pool
import functools
import sys
import traceback

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos-csv", required=True, help="CSV with column 'video' listing video basenames or paths")
    p.add_argument("--frames-root", required=True, help="Root that contains <stem>_roi/annotated/<stem>/frame000000.jpg")
    p.add_argument("--smoothed-root", default="runs/inference", help="Root where <stem>_roi/labels/smoothed/<stem>_mouth.txt live")
    p.add_argument("--out-root", default="runs/inference", help="Root where crops and motion will be written")
    p.add_argument("--resize", type=int, default=224, help="Output square size for crops and motion maps")
    p.add_argument("--workers", type=int, default=1, help="Parallel worker processes (1 = no multiprocessing)")
    p.add_argument("--max-frames", type=int, default=None, help="Optional cap per video for quick tests")
    p.add_argument("--start-frame", type=int, default=0, help="Optional start frame index (inclusive)")
    p.add_argument("--end-frame", type=int, default=None, help="Optional end frame index (exclusive)")
    return p.parse_args()

def read_video_list(csv_path: Path):
    vids = []
    with csv_path.open("r", encoding='utf-8-sig', newline='') as fh:
        rdr = csv.DictReader(fh)
        if rdr.fieldnames:
            rdr.fieldnames = [fn.strip().lstrip("\ufeff") for fn in rdr.fieldnames]
        for r in rdr:
            v = (r.get("video") or r.get("video_path") or r.get("filename") or "").strip()
            if v:
                vids.append(v)
    # preserve order, unique
    seen = set()
    out = []
    for v in vids:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out

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
    try:
        frames_root = Path(args.frames_root)
        smoothed_root = Path(args.smoothed_root)
        out_root = Path(args.out_root)
        resize = args.resize
        max_frames = args.max_frames
        start_frame_arg = args.start_frame
        end_frame_arg = args.end_frame

        stem = Path(video_name).stem
        # IMPORTANT: use annotated/<stem>/frame*.jpg layout
        frame_folder = frames_root / f"{stem}_roi" / "annotated" / stem
        if not frame_folder.exists():
            return {"video": video_name, "status": "frames_missing", "path": str(frame_folder)}

        smoothed_path = smoothed_root / f"{stem}_roi" / "labels" / "smoothed" / f"{stem}_mouth.txt"
        smoothed = read_smoothed(smoothed_path)
        if smoothed is None:
            return {"video": video_name, "status": "smoothed_missing", "path": str(smoothed_path)}

        out_video_root = out_root / f"{stem}_roi"
        crops_dir = out_video_root / "crops"
        motion_dir = out_video_root / "labels" / "motion"
        crops_dir.mkdir(parents=True, exist_ok=True)
        motion_dir.mkdir(parents=True, exist_ok=True)

        # list frames sorted
        frame_files = sorted(frame_folder.glob("frame*.jpg"))
        if not frame_files:
            return {"video": video_name, "status": "no_frames_found", "path": str(frame_folder)}
        total_frames = len(frame_files)
        # determine processing range
        start_frame = max(0, start_frame_arg)
        end_frame = total_frames if end_frame_arg is None else min(end_frame_arg, total_frames)
        if max_frames:
            end_frame = min(end_frame, start_frame + max_frames)

        # resume: skip frames already computed
        existing = sorted(motion_dir.glob("motion_*.npy"))
        if existing:
            try:
                last_done = int(existing[-1].stem.split("_")[1])
                if last_done >= start_frame:
                    start_frame = last_done + 1
            except Exception:
                pass

        # read previous frame gray if available
        prev_gray_full = None
        if start_frame > 0:
            pprev = frame_folder / f"frame{start_frame-1:06d}.jpg"
            if pprev.exists():
                img_prev = cv2.imread(str(pprev))
                if img_prev is not None:
                    prev_gray_full = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)

        t0 = time.time()
        for fi in range(start_frame, end_frame):
            if fi % 50 == 0:
                elapsed = time.time() - t0
                print(f"{stem}: frame {fi}/{end_frame} elapsed {elapsed:.1f}s")
            frame_path = frame_folder / f"frame{fi:06d}.jpg"
            if not frame_path.exists():
                # write placeholders to keep indexing consistent
                np.save(motion_dir / f"motion_{fi:06d}.npy", np.zeros((resize, resize), dtype=np.float32))
                continue
            img = cv2.imread(str(frame_path))
            if img is None:
                np.save(motion_dir / f"motion_{fi:06d}.npy", np.zeros((resize, resize), dtype=np.float32))
                continue
            H, W = img.shape[:2]

            # determine ROI entry for this frame (use current or fallback to last valid)
            entry = smoothed[fi] if fi < len(smoothed) else None
            if entry is None:
                # fallback: search backward for last valid
                j = fi - 1
                while j >= 0 and (j >= len(smoothed) or smoothed[j] is None):
                    j -= 1
                entry = smoothed[j] if j >= 0 else None

            if entry is None:
                # no ROI available: save blank crop/motion
                np.save(motion_dir / f"motion_{fi:06d}.npy", np.zeros((resize, resize), dtype=np.float32))
                blank = np.zeros((resize, resize, 3), dtype=np.uint8)
                cv2.imwrite(str(crops_dir / f"crop_{fi:06d}.jpg"), blank, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                continue

            xc, yc, w, h = entry
            cx = int(round(xc * W)); cy = int(round(yc * H))
            bw = int(round(w * W)); bh = int(round(h * H))
            x1 = max(0, cx - bw//2); y1 = max(0, cy - bh//2)
            x2 = min(W, cx + bw//2); y2 = min(H, cy + bh//2)
            if x2 <= x1 or y2 <= y1:
                np.save(motion_dir / f"motion_{fi:06d}.npy", np.zeros((resize, resize), dtype=np.float32))
                blank = np.zeros((resize, resize, 3), dtype=np.uint8)
                cv2.imwrite(str(crops_dir / f"crop_{fi:06d}.jpg"), blank, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                continue

            # extract crop and save JPEG
            crop = img[y1:y2, x1:x2]
            crop_resized = cv2.resize(crop, (resize, resize), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(crops_dir / f"crop_{fi:06d}.jpg"), crop_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            # compute Farneback between prev_gray_full and current
            cur_gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if prev_gray_full is None:
                # first available frame: save zeros for motion
                np.save(motion_dir / f"motion_{fi:06d}.npy", np.zeros((resize, resize), dtype=np.float32))
                prev_gray_full = cur_gray_full
                continue

            prev_roi = prev_gray_full[y1:y2, x1:x2]
            cur_roi  = cur_gray_full[y1:y2, x1:x2]
            if prev_roi is None or prev_roi.size == 0 or cur_roi.size == 0:
                np.save(motion_dir / f"motion_{fi:06d}.npy", np.zeros((resize, resize), dtype=np.float32))
                prev_gray_full = cur_gray_full
                continue

            # Farneback parameters tuned for CPU speed
            flow = cv2.calcOpticalFlowFarneback(prev_roi, cur_roi, None,
                                                pyr_scale=0.5, levels=3, winsize=9,
                                                iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            mag = cv2.GaussianBlur(mag, (5,5), 0)
            p98 = np.percentile(mag, 98) if mag.size else 1.0
            if p98 <= 0:
                norm = mag
            else:
                norm = mag / (p98 + 1e-6)
            norm = np.clip(norm, 0.0, 1.0)
            norm_resized = cv2.resize(norm, (resize, resize), interpolation=cv2.INTER_LINEAR)
            np.save(motion_dir / f"motion_{fi:06d}.npy", norm_resized.astype(np.float32))

            prev_gray_full = cur_gray_full

        return {"video": video_name, "status": "done", "processed_frames": end_frame - start_frame}
    except KeyboardInterrupt:
        return {"video": video_name, "status": "interrupted"}
    except Exception:
        traceback.print_exc()
        return {"video": video_name, "status": "error", "exception": str(sys.exc_info()[1])}

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

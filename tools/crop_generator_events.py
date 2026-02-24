#!/usr/bin/env python3
"""
crop_generator_events.py

Generate normalized ROI crops from event_rois.json and source videos.

Usage:
    python tools/crop_generator_events.py --events event_rois.json --out-dir ./crops

You can also override the built-in video directory with --videos-dir.

Set the built-in video directory below by editing VIDEO_DIR.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os

# -------------------------
# Configuration (edit here)
# -------------------------
# Set this to the folder that contains .avi videos
VIDEO_DIR = Path(r"\\research.drive.wisc.edu\npconnor\ADStudy\VF AD Blinded\Early Tongue Training")

# Output image format and quality
JPEG_QUALITY = 90  # 0-100, higher = better quality/larger files

# Canonical crop size
CANONICAL_W = 128
CANONICAL_H = 128

# CLAHE parameters
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

# How many frames to sample when computing per-video reference median
DEFAULT_SAMPLE_FRAMES = 200

EPS = 1e-6

# Cache file for medians
MEDIAN_CACHE_PATH = Path(".median_cache.json")

# -------------------------
# Helper functions
# -------------------------
def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    return json.loads(path.read_text())

def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))

def _load_median_cache():
    if MEDIAN_CACHE_PATH.exists():
        try:
            return json.loads(MEDIAN_CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}

def _save_median_cache(cache):
    try:
        MEDIAN_CACHE_PATH.write_text(json.dumps(cache, indent=2))
    except Exception:
        pass

def _video_fingerprint(video_path: Path):
    """
    Small fingerprint to detect if the video file changed: use size + mtime.
    """
    try:
        st = video_path.stat()
        return f"{st.st_size}-{int(st.st_mtime)}"
    except Exception:
        return str(video_path)

def compute_video_reference_median(video_path: Path, sample_count=DEFAULT_SAMPLE_FRAMES, thumb_size=(64, 64), force_recompute=False):
    """
    Compute a robust per-video median by sampling up to sample_count frames.
    To speed things up we downsample each sampled frame to thumb_size before
    computing the median. Results are cached in .median_cache.json unless
    force_recompute is True.
    """
    cache = _load_median_cache()
    key = str(video_path)
    fingerprint = _video_fingerprint(video_path)
    cached = cache.get(key)
    if cached and not force_recompute and cached.get("fingerprint") == fingerprint:
        try:
            return float(cached.get("reference_median"))
        except Exception:
            pass

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    indices = np.linspace(0, max(total-1, 0), min(sample_count, total), dtype=int)
    medians = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, f = cap.read()
        if not ret or f is None:
            continue
        # convert to gray and downsample to a small thumbnail for speed
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
        thumb = cv2.resize(gray, thumb_size, interpolation=cv2.INTER_AREA)
        medians.append(float(np.median(thumb)))
    cap.release()
    if not medians:
        ref = 128.0
    else:
        ref = float(np.median(medians))

    # cache the result
    try:
        cache[key] = {"reference_median": ref, "fingerprint": fingerprint, "computed_at": datetime.utcnow().isoformat() + "Z"}
        _save_median_cache(cache)
    except Exception:
        pass

    return ref

def normalize_frame_to_reference(frame_gray: np.ndarray, frame_median: float, ref_median: float):
    if frame_median <= 0:
        return frame_gray
    scale = (ref_median + EPS) / (frame_median + EPS)
    out = frame_gray.astype(np.float32) * scale
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def apply_clahe(img_gray: np.ndarray, clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(img_gray)

def crop_and_process(frame, box, canonical_wh, ref_median=None, apply_video_norm=True):
    x, y, w, h = box
    h_img, w_img = frame.shape[:2]
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(8, min(w, w_img - x))
    h = max(8, min(h, h_img - y))
    crop = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    frame_median = float(np.median(gray))
    if apply_video_norm and ref_median is not None:
        gray = normalize_frame_to_reference(gray, frame_median, ref_median)
    gray = apply_clahe(gray)
    out = cv2.resize(gray, canonical_wh, interpolation=cv2.INTER_LINEAR)
    return out

def ensure_int_box(box):
    return [int(round(v)) for v in box]

# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--events", required=True, help="Path to event_rois.json")
    p.add_argument("--videos-dir", required=False, help="Directory where videos live (optional). If omitted, uses VIDEO_DIR in the script.")
    p.add_argument("--out-dir", default="./crops", help="Output root for crops")
    p.add_argument("--canonical-w", type=int, default=CANONICAL_W)
    p.add_argument("--canonical-h", type=int, default=CANONICAL_H)
    p.add_argument("--sample-frames", type=int, default=DEFAULT_SAMPLE_FRAMES)
    p.add_argument("--only-saved", action="store_true", help="Only generate crops for frames that have saved entries in JSON")
    p.add_argument("--force-recompute", action="store_true", help="Ignore cached medians and recompute for each video")
    args = p.parse_args()

    events_path = Path(args.events)
    # Use command-line videos-dir if provided, otherwise fall back to VIDEO_DIR constant
    videos_dir = Path(args.videos_dir) if args.videos_dir else VIDEO_DIR
    out_root = Path(args.out_dir)
    canonical_wh = (args.canonical_w, args.canonical_h)

    print(f"[INFO] events: {events_path}")
    print(f"[INFO] videos dir: {videos_dir}")
    print(f"[INFO] out dir: {out_root}")

    rois = load_json(events_path)

    video_keys = [k for k in rois.keys() if k != "__event_slot_map__"]
    if not video_keys:
        print("No videos found in JSON.")
        return

    for vname in video_keys:
        print(f"\nProcessing video {vname}")
        vpath = Path(vname)
        if not vpath.exists():
            candidate = videos_dir / vname
            if candidate.exists():
                vpath = candidate
            else:
                found = None
                for ext in (".mp4", ".avi", ".mov", ".mkv"):
                    cand = videos_dir / (vname + ext)
                    if cand.exists():
                        found = cand
                        break
                if found:
                    vpath = found
                else:
                    print(f"[WARN] Video not found for {vname}, skipping.")
                    continue

        print("  computing video reference median ...")
        ref_median = compute_video_reference_median(vpath, sample_count=args.sample_frames, force_recompute=args.force_recompute)
        print(f"  reference median: {ref_median:.2f}")

        video_out = out_root / vname
        mouth_out = video_out / "mouth"
        ues_out = video_out / "ues"
        mouth_out.mkdir(parents=True, exist_ok=True)
        ues_out.mkdir(parents=True, exist_ok=True)

        metadata = {
            "video": str(vpath),
            "reference_median": ref_median,
            "canonical_wh": canonical_wh,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "crops": []
        }

        entries = rois.get(vname, [])
        if not entries:
            print("  no saved entries for this video in JSON.")
            continue

        # Open a single VideoCapture per video and process entries in ascending frame order
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"[WARN] cannot open video {vpath}, skipping.")
            continue

        try:
            # keep original index so metadata preserves entry_index
            indexed_entries = list(enumerate(entries))
            entries_sorted = sorted(indexed_entries, key=lambda t: int(t[1].get("frame_index", 0)))
            last_read_idx = -1

            for orig_i, e in tqdm(entries_sorted, desc=f"  entries ({len(entries)})"):
                try:
                    ev_id = str(e.get("event_id"))
                    frame_idx = int(e.get("frame_index"))
                    mouth_box = e.get("mouth")
                    ues_box = e.get("ues")
                    visibility = str(e.get("visibility", "visible")).strip().lower()
                    if visibility == "visible":
                        weight = 1.0
                    elif visibility == "partial":
                        weight = 0.4
                    else:
                        weight = 0.0

                    # Efficient frame access: advance forward with grab/read when possible
                    if frame_idx < last_read_idx:
                        # backward jump: seek directly
                        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
                        ret, frame = cap.read()
                    else:
                        # try to read forward without expensive seeks
                        cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                        if cur < 0:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
                            ret, frame = cap.read()
                        else:
                            # advance using grab until we reach the desired frame
                            while cur < frame_idx:
                                ok = cap.grab()
                                if not ok:
                                    break
                                cur += 1
                            ret, frame = cap.read()
                    last_read_idx = frame_idx

                    if not ret or frame is None:
                        print(f"    [WARN] cannot read frame {frame_idx} for {vname}")
                        continue

                    outp = None
                    outp2 = None

                    if mouth_box:
                        mb = ensure_int_box(mouth_box)
                        mouth_img = crop_and_process(frame, mb, canonical_wh, ref_median=ref_median, apply_video_norm=True)
                        fname = f"event_{orig_i:03d}_{ev_id}_frame_{frame_idx}_mouth.jpg"
                        outp = mouth_out / fname
                        cv2.imwrite(str(outp), mouth_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

                    if ues_box:
                        ub = ensure_int_box(ues_box)
                        ues_img = crop_and_process(frame, ub, canonical_wh, ref_median=ref_median, apply_video_norm=True)
                        fname2 = f"event_{orig_i:03d}_{ev_id}_frame_{frame_idx}_ues.jpg"
                        outp2 = ues_out / fname2
                        cv2.imwrite(str(outp2), ues_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

                    metadata["crops"].append({
                        "entry_index": orig_i,
                        "event_id": ev_id,
                        "frame_index": frame_idx,
                        "visibility": visibility,
                        "weight": weight,
                        "mouth_path": str(outp.relative_to(video_out)) if outp else None,
                        "ues_path": str(outp2.relative_to(video_out)) if outp2 else None
                    })
                except Exception as ex:
                    print(f"    [ERROR] processing entry {orig_i}: {ex}")
                    continue
        finally:
            cap.release()

        save_json(video_out / "metadata.json", metadata)
        print(f"  saved crops to {video_out}")

    print("\nAll done.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
detector/run_full_pipeline.py

Multi-video end-to-end pipeline:
 - reads video list from CSV (column 'video')
 - runs YOLO inference on every frame
 - links detections into tracks (greedy IoU)
 - filters tracks by mean confidence and logs per-track JSONs
 - selects best track per ROI using event_rois.json anchors (if available)
 - interpolates and Savitzky-Golay smooths the chosen track
 - extracts stabilized crops per frame and saves JPEGs
 - optionally saves ImageNet-normalized .npy tensors (CHW float32)
 - writes manifest.csv with track_id and conf for each crop

Usage:
python detector/run_full_pipeline.py `
  --weights 'detector/models/yolov8_train/weights/best.pt' `
  --video-dir '\\research.drive.wisc.edu\npconnor\ADStudy\VF AD Blinded\Early Tongue Training' `
  --videos-csv 'event_csvs/cleaned_events.csv' `
  --out-dir 'detector/pipeline_output' `
  --conf 0.40 `
  --track-conf-thresh 0.20 `
  --save-normalized True `
  --use-clahe True `
  --clahe-clip 2.0 `
  --clahe-tile '8,8'

"""
import argparse
import csv
import json
import logging
import math
import os
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
from scipy.signal import savgol_filter
from tqdm import tqdm

# -----------------------
# Defaults (adjust if needed)
# -----------------------
DEFAULT_WEIGHTS = "detector/models/yolov8_train/weights/best.pt"
DEFAULT_VIDEO_DIR = Path(r"\\research.drive.wisc.edu\npconnor\ADStudy\VF AD Blinded\Early Tongue Training")
DEFAULT_VIDEOS_CSV = Path("detector/event_csvs/cleaned_events.csv")
DEFAULT_OUT_DIR = Path(r"E:\VF ML Crops")
EVENT_ROIS_PATH = Path("detector/event_rois.json")
CONF_THRES = 0.25
IOU_LINK_THRESH = 0.30
MAX_GAP_FRAMES = 5
SMOOTH_WINDOW = 11
SMOOTH_POLY = 2
PAD_FACTOR = 1.3
OUT_CROP_SIZE = (128, 128)   # user-specified model input size
MIN_BOX_WH = 8.0
LABEL_MAP = {0: "mouth", 1: "ues"}   # adjust to your model's class ids
TRACK_CONF_THRESH = 0.20
JPEG_QUALITY = 95
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# CLAHE defaults
USE_CLAHE_DEFAULT = False
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
# Visualization / drawing defaults
VIS_CONF_MIN = 0.40   # only consider detections >= this confidence for visualization selection
# -----------------------

# -----------------------
# Utilities
# -----------------------
def setup_logging(log_file: Path):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Add file handler for this specific log file (avoid duplicates for same file)
    file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(log_file)]
    if not file_handlers:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), mode="w")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        root.addHandler(fh)

    # Add a single console handler if none exists (prevents duplicate console output)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        root.addHandler(ch)


def normpath_str(p: str) -> str:
    return os.path.normpath(p)

def xyxy_to_cxcywh(xyxy):
    x1,y1,x2,y2 = xyxy
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = x1 + w/2.0
    cy = y1 + h/2.0
    return [float(cx), float(cy), float(w), float(h)]

def cxcywh_to_xyxy(cx,cy,w,h):
    x1 = cx - w/2.0
    y1 = cy - h/2.0
    x2 = cx + w/2.0
    y2 = cy + h/2.0
    return [x1,y1,x2,y2]

def iou_xyxy(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h
    areaA = max(1e-6, (a[2]-a[0])*(a[3]-a[1]))
    areaB = max(1e-6, (b[2]-b[0])*(b[3]-b[1]))
    return inter / (areaA + areaB - inter + 1e-8)

# -----------------------
# Inference (per-video)
# -----------------------
def run_inference_and_save(model, video_path: Path, out_dir: Path, conf_thres: float):
    logging.info(f"[INFER] Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    logging.info(f"[{video_path.name}] frames: {total}")
    raw = []
    frame_idx = 0
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    # We intentionally do not write per-frame vis images to disk (commented out).
    # Use a terminal-only progress bar for frame-level visibility.
    if total <= 0:
        # fallback to while loop if frame count unknown
        pbar = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                res_list = model(frame, conf=conf_thres, verbose=False)
            except Exception as e:
                logging.exception(f"Model inference failed on frame {frame_idx}: {e}")
                boxes = []
                raw.append({"frame": int(frame_idx), "boxes": boxes})
                frame_idx += 1
                continue
            r = res_list[0]
            boxes = []
            if hasattr(r, "boxes") and len(r.boxes) > 0:
                xyxy_arr = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                for (x1,y1,x2,y2), conf, cls in zip(xyxy_arr, confs, clss):
                    boxes.append({
                        "class": int(cls),
                        "conf": float(conf),
                        "xyxy": [float(x1), float(y1), float(x2), float(y2)]
                    })
            raw.append({"frame": int(frame_idx), "boxes": boxes})
            frame_idx += 1
    else:
        # Known total: use tqdm for an ETA and frame progress in terminal only
        with tqdm(total=total, desc=f"Infer {video_path.name}", ncols=80, unit="fr") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    res_list = model(frame, conf=conf_thres, verbose=False)
                except Exception as e:
                    logging.exception(f"Model inference failed on frame {frame_idx}: {e}")
                    boxes = []
                    raw.append({"frame": int(frame_idx), "boxes": boxes})
                    frame_idx += 1
                    pbar.update(1)
                    continue
                r = res_list[0]
                boxes = []
                if hasattr(r, "boxes") and len(r.boxes) > 0:
                    xyxy_arr = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    clss = r.boxes.cls.cpu().numpy().astype(int)
                    for (x1,y1,x2,y2), conf, cls in zip(xyxy_arr, confs, clss):
                        boxes.append({
                            "class": int(cls),
                            "conf": float(conf),
                            "xyxy": [float(x1), float(y1), float(x2), float(y2)]
                        })
                raw.append({"frame": int(frame_idx), "boxes": boxes})
                frame_idx += 1
                pbar.update(1)

    cap.release()
    (out_dir / "raw_detections.json").write_text(json.dumps(raw, indent=2))
    logging.info(f"[{video_path.name}] wrote raw detections ({len(raw)}) to {out_dir/'raw_detections.json'}")
    return raw, total


# -----------------------
# Linking (greedy IoU)
# -----------------------
def link_tracks_greedy(detections, iou_thresh=IOU_LINK_THRESH, max_gap=MAX_GAP_FRAMES):
    tracks_by_label = {}
    for entry in detections:
        t = entry["frame"]
        for b in entry["boxes"]:
            lbl = int(b["class"])
            cx,cy,w,h = xyxy_to_cxcywh(b["xyxy"])
            conf = float(b["conf"])
            tracks_by_label.setdefault(lbl, [])
            matched = False
            for track in tracks_by_label[lbl]:
                last_frame, last_cx, last_cy, last_w, last_h, last_conf = track[-1]
                if t - last_frame > max_gap:
                    continue
                last_xy = cxcywh_to_xyxy(last_cx, last_cy, last_w, last_h)
                cur_xy = cxcywh_to_xyxy(cx, cy, w, h)
                if iou_xyxy(last_xy, cur_xy) >= iou_thresh:
                    track.append((t, cx, cy, w, h, conf))
                    matched = True
                    break
            if not matched:
                tracks_by_label[lbl].append([(t, cx, cy, w, h, conf)])
    serial = {}
    for lbl, tracks in tracks_by_label.items():
        serial[lbl] = []
        for tidx, tr in enumerate(tracks):
            tr_serial = [list(map(float, item)) for item in tr]
            mean_conf = float(np.mean([t[-1] for t in tr])) if len(tr) > 0 else 0.0
            serial[lbl].append({"track_id": tidx, "detections": tr_serial, "mean_conf": mean_conf, "length": len(tr)})
    return tracks_by_label, serial

# -----------------------
# Anchors from event_rois.json
# -----------------------
def load_expected_centers(event_rois_path: Path, video_name: str, label_map: Dict[int,str]) -> Dict[int, Tuple[float,float]]:
    if not event_rois_path.exists():
        logging.warning(f"Event ROIs file not found: {event_rois_path}")
        return {}
    data = json.loads(event_rois_path.read_text())
    if video_name not in data:
        logging.info(f"No annotations for video {video_name} in {event_rois_path}")
        return {}
    ann_list = data[video_name]
    centers_by_label = {lbl: [] for lbl in label_map.keys()}
    for ev in ann_list:
        for lbl, name in label_map.items():
            if name in ev and ev[name]:
                x,y,w,h = ev[name]
                cx = float(x) + float(w)/2.0
                cy = float(y) + float(h)/2.0
                centers_by_label[lbl].append((cx, cy))
    expected = {}
    for lbl, centers in centers_by_label.items():
        if len(centers) == 0:
            continue
        xs = [c[0] for c in centers]; ys = [c[1] for c in centers]
        expected[lbl] = (float(statistics.median(xs)), float(statistics.median(ys)))
    return expected

def mean_track_distance_to_point(track: List[Tuple[int,float,float,float,float,float]], point: Tuple[float,float]) -> float:
    if len(track) == 0:
        return float("inf")
    px, py = point
    dists = [math.hypot(t[1] - px, t[2] - py) for t in track]
    return float(sum(dists) / len(dists))

# -----------------------
# Interpolate & smooth
# -----------------------
def interpolate_and_smooth_track(track, total_frames, window=SMOOTH_WINDOW, poly=SMOOTH_POLY):
    frames = np.array([int(t[0]) for t in track], dtype=int)
    vals = np.array([t[1:5] for t in track], dtype=float)
    timeline = np.arange(total_frames, dtype=int)
    interp = np.full((total_frames, 4), np.nan, dtype=float)
    if len(frames) == 0:
        return interp
    for dim in range(4):
        interp[frames, dim] = vals[:, dim]
        known = ~np.isnan(interp[:, dim])
        if known.sum() == 0:
            continue
        interp[:, dim] = np.interp(timeline, timeline[known], interp[known, dim])
    w = min(window, total_frames if total_frames % 2 == 1 else total_frames-1)
    if w >= 5:
        for dim in range(4):
            interp[:, dim] = savgol_filter(interp[:, dim], w, poly)
    interp[:, 2] = np.maximum(interp[:, 2], MIN_BOX_WH)
    interp[:, 3] = np.maximum(interp[:, 3], MIN_BOX_WH)
    return interp

# -----------------------
# Save per-track JSONs and filtering
# -----------------------
def save_tracks_and_filter(tracks_by_label, out_dir: Path, track_conf_thresh: float):
    tracks_dir = out_dir / "tracks_by_label"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    filtered = {}
    for lbl, tracks in tracks_by_label.items():
        label_name = LABEL_MAP.get(lbl, str(lbl))
        filtered[lbl] = []
        for tidx, tr in enumerate(tracks):
            mean_conf = float(np.mean([t[-1] for t in tr])) if len(tr) > 0 else 0.0
            mean_cx = float(np.mean([t[1] for t in tr])) if len(tr) > 0 else 0.0
            mean_cy = float(np.mean([t[2] for t in tr])) if len(tr) > 0 else 0.0
            tr_obj = {
                "track_id": tidx,
                "label": int(lbl),
                "label_name": label_name,
                "length": len(tr),
                "mean_conf": mean_conf,
                "mean_center": [mean_cx, mean_cy],
                "detections": [[int(t[0]), float(t[1]), float(t[2]), float(t[3]), float(t[4]), float(t[5])] for t in tr]
            }
            (tracks_dir / f"{label_name}_track_{tidx:03d}.json").write_text(json.dumps(tr_obj, indent=2))
            logging.info(f"Saved track file: {label_name}_track_{tidx:03d}.json (len={tr_obj['length']}, mean_conf={mean_conf:.3f})")
            if mean_conf >= track_conf_thresh:
                filtered[lbl].append(tr)
            else:
                logging.info(f"Filtered out track {label_name}_track_{tidx:03d} due to low mean_conf {mean_conf:.3f} < {track_conf_thresh}")
    return filtered

# -----------------------
# Cropping & manifest (JPEG + optional normalized .npy) with optional CLAHE
# -----------------------
def extract_and_save_crops(video_path: Path, smoothed_info: Dict[int, Dict], out_dir: Path,
                           pad_factor=PAD_FACTOR, out_size=OUT_CROP_SIZE,
                           jpeg_quality=JPEG_QUALITY, save_normalized: bool = True,
                           use_clahe: bool = USE_CLAHE_DEFAULT, clahe_clip: float = CLAHE_CLIP, clahe_tile: Tuple[int,int] = CLAHE_TILE):
    """
    smoothed_info: dict[label] -> {"array": np.ndarray (T,4), "track_id": int, "mean_conf": float}
    Saves JPEG crops and optionally normalized numpy arrays (.npy).
    CLAHE is applied to the resized crop (on luminance) if use_clahe is True.
    Crops are generated from the smoothed centers so they reflect stabilized ROI positions.
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    norm_dir = out_dir / "crops_normalized"
    if save_normalized:
        norm_dir.mkdir(parents=True, exist_ok=True)
    for lbl in smoothed_info.keys():
        label_name = LABEL_MAP.get(lbl, str(lbl))
        (crops_dir / label_name).mkdir(parents=True, exist_ok=True)
        if save_normalized:
            (norm_dir / label_name).mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    frame_idx = 0
    # prepare CLAHE object if needed
    clahe_obj = None
    if use_clahe:
        clahe_obj = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=tuple(clahe_tile))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h_img, w_img = frame.shape[:2]
        for lbl, info in smoothed_info.items():
            arr = info["array"]
            track_id = int(info.get("track_id", -1))
            mean_conf = float(info.get("mean_conf", 0.0))
            label_name = LABEL_MAP.get(lbl, str(lbl))
            # Use smoothed center/w/h for cropping (stabilized ROI)
            cx,cy,w,h = arr[frame_idx].tolist()
            size = max(w, h) * pad_factor
            x1 = int(round(cx - size/2.0)); y1 = int(round(cy - size/2.0))
            x2 = int(round(cx + size/2.0)); y2 = int(round(cy + size/2.0))
            left = max(0, -x1); top = max(0, -y1)
            right = max(0, x2 - w_img); bottom = max(0, y2 - h_img)
            if any([left, top, right, bottom]):
                frame_padded = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
                x1 += left; x2 += left; y1 += top; y2 += top
            else:
                frame_padded = frame
            x1c = max(0, min(frame_padded.shape[1]-1, x1))
            y1c = max(0, min(frame_padded.shape[0]-1, y1))
            x2c = max(0, min(frame_padded.shape[1], x2))
            y2c = max(0, min(frame_padded.shape[0], y2))
            if x2c <= x1c or y2c <= y1c:
                cx_i, cy_i = w_img//2, h_img//2
                half = int(max(out_size)/2)
                x1c = max(0, cx_i-half); y1c = max(0, cy_i-half)
                x2c = min(frame_padded.shape[1], cx_i+half); y2c = min(frame_padded.shape[0], cy_i+half)
            crop = frame_padded[y1c:y2c, x1c:x2c]
            crop_resized = cv2.resize(crop, out_size, interpolation=cv2.INTER_LINEAR)

            # optional CLAHE on luminance (apply to resized crop)
            if use_clahe and clahe_obj is not None:
                try:
                    ycrcb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2YCrCb)
                    y, cr, cb = cv2.split(ycrcb)
                    y_clahe = clahe_obj.apply(y)
                    ycrcb_clahe = cv2.merge([y_clahe, cr, cb])
                    crop_resized = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2BGR)
                except Exception:
                    logging.exception("CLAHE application failed on a crop; using unmodified crop.")

            # Save JPEG (visual/archival)
            fname_jpg = f"{label_name}/frame_{frame_idx:06d}.jpg"
            out_path_jpg = crops_dir / fname_jpg
            cv2.imwrite(str(out_path_jpg), crop_resized, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])

            normalized_relpath = ""
            if save_normalized:
                # Convert BGR->RGB, scale to [0,1], normalize with ImageNet stats
                crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                crop_f = crop_rgb.astype(np.float32) / 255.0  # H,W,3 in [0,1]
                crop_norm = (crop_f - IMAGENET_MEAN[None,None,:]) / IMAGENET_STD[None,None,:]
                crop_norm_chw = np.transpose(crop_norm, (2,0,1)).astype(np.float32)
                fname_npy = f"{label_name}/frame_{frame_idx:06d}.npy"
                out_path_npy = norm_dir / fname_npy
                np.save(str(out_path_npy), crop_norm_chw)
                normalized_relpath = str(out_path_npy.relative_to(out_dir))

            manifest_rows.append({
                "frame": frame_idx,
                "roi": label_name,
                "label": int(lbl),
                "track_id": track_id,
                "cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h),
                "conf": float(mean_conf),
                "filename": str(out_path_jpg.relative_to(out_dir)),
                "normalized_filename": normalized_relpath
            })
        frame_idx += 1
    cap.release()
    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame","roi","label","track_id","cx","cy","w","h","conf","filename","normalized_filename"])
        writer.writeheader()
        for r in manifest_rows:
            writer.writerow(r)
    logging.info(f"Wrote crops and manifest to {out_dir}")
    return manifest_path

# -----------------------
# Multi-video orchestration helpers
# -----------------------
def read_video_list_from_csv(csv_path: Path) -> List[str]:
    videos = []
    if not csv_path.exists():
        logging.error(f"Videos CSV not found: {csv_path}")
        return videos
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if "video" not in reader.fieldnames:
            logging.error(f"'video' column not found in CSV: {csv_path}")
            return videos
        for row in reader:
            v = row.get("video")
            if v:
                videos.append(v.strip())
    return videos

def resolve_video_path(video_dir: Path, video_entry: str) -> Path:
    vstr = video_entry.strip().strip('"').strip("'")
    vstr_norm = normpath_str(vstr)
    p = Path(vstr_norm)
    if p.is_absolute():
        return p
    vd = normpath_str(str(video_dir))
    joined = Path(os.path.join(vd, vstr_norm))
    return joined

def process_single_video(model, video_file: Path, per_video_out: Path, conf_thres: float, track_conf_thresh: float, save_normalized: bool, use_clahe: bool):
    per_video_out.mkdir(parents=True, exist_ok=True)
    setup_logging(per_video_out / "pipeline.log")
    logging.info(f"Processing video: {video_file}")
    if not video_file.exists():
        logging.error(f"Video file not found at resolved path: {video_file}")
        return
    raw_detections, total_frames = run_inference_and_save(model, video_file, per_video_out, conf_thres)
    tracks_by_label, serial_tracks = link_tracks_greedy(raw_detections, iou_thresh=IOU_LINK_THRESH, max_gap=MAX_GAP_FRAMES)
    (per_video_out / "tracks.json").write_text(json.dumps(serial_tracks, indent=2))
    logging.info(f"Wrote tracks.json with labels: {list(serial_tracks.keys())}")
    filtered_tracks = save_tracks_and_filter(tracks_by_label, per_video_out, track_conf_thresh)
    video_basename = video_file.name
    expected_centers = load_expected_centers(EVENT_ROIS_PATH, video_basename, LABEL_MAP)
    if expected_centers:
        logging.info(f"Loaded expected centers for labels: {expected_centers}")
    else:
        logging.info("No expected centers found; will fallback to longest/highest-confidence track selection")
    smoothed_info = {}
    for lbl, tracks in filtered_tracks.items():
        label_name = LABEL_MAP.get(lbl, str(lbl))
        if len(tracks) == 0:
            logging.warning(f"No tracks remain for label {label_name} after confidence filtering")
            smoothed_info[lbl] = {"array": np.tile(np.array([0.0,0.0,MIN_BOX_WH,MIN_BOX_WH]), (total_frames,1)), "track_id": -1, "mean_conf": 0.0}
            continue
        if lbl in expected_centers:
            best_score = float("inf")
            best_track = None
            best_tid = None
            for tidx, tr in enumerate(tracks):
                mean_dist = mean_track_distance_to_point(tr, expected_centers[lbl])
                mean_conf = float(np.mean([t[-1] for t in tr])) if len(tr) > 0 else 0.0
                score = mean_dist - 0.5 * mean_conf
                logging.info(f"Label {label_name} track len={len(tr)} mean_conf={mean_conf:.3f} mean_dist={mean_dist:.1f} score={score:.3f}")
                if score < best_score:
                    best_score = score
                    best_track = tr
                    best_tid = tidx
            if best_track is None:
                best_track = max(tracks, key=lambda t: len(t))
                best_tid = 0
                logging.info(f"Fallback to longest track for {label_name}")
        else:
            best_tid = None
            best_score = -float("inf")
            best_track = None
            for tidx, tr in enumerate(tracks):
                mean_conf = float(np.mean([t[-1] for t in tr])) if len(tr) > 0 else 0.0
                score = (mean_conf, len(tr))
                if score > best_score:
                    best_score = score
                    best_track = tr
                    best_tid = tidx
            logging.info(f"No anchor for {label_name}; selected track id={best_tid} with mean_conf={best_score[0]:.3f} len={best_score[1]}")
        interp = interpolate_and_smooth_track(best_track, total_frames, window=SMOOTH_WINDOW, poly=SMOOTH_POLY)
        if np.isnan(interp).any():
            cap = cv2.VideoCapture(str(video_file))
            ret, frame0 = cap.read()
            cap.release()
            if ret:
                h0,w0 = frame0.shape[:2]
                fallback = np.array([w0/2.0, h0/2.0, 40.0, 40.0])
            else:
                fallback = np.array([100.0,100.0,40.0,40.0])
            nan_mask = np.isnan(interp).any(axis=1)
            interp[nan_mask] = fallback
            logging.info(f"Filled NaNs in smoothed array for {label_name} with fallback center")
        mean_conf_selected = float(np.mean([t[-1] for t in best_track])) if len(best_track) > 0 else 0.0
        smoothed_info[lbl] = {"array": interp, "track_id": int(best_tid) if best_tid is not None else -1, "mean_conf": mean_conf_selected}
        np.save(per_video_out / f"smoothed_label_{lbl}.npy", interp)
        logging.info(f"Saved smoothed_label_{lbl}.npy for {label_name} (track_id={smoothed_info[lbl]['track_id']}, mean_conf={smoothed_info[lbl]['mean_conf']:.3f})")
    # Ensure crops are generated from smoothed centers (smoothed_info) and optionally apply CLAHE
    manifest = extract_and_save_crops(video_file, smoothed_info, per_video_out, pad_factor=PAD_FACTOR, out_size=OUT_CROP_SIZE, jpeg_quality=JPEG_QUALITY, save_normalized=save_normalized, use_clahe=use_clahe)
    logging.info(f"Completed processing for {video_file.name}. Manifest: {manifest}")

# -----------------------
# CLI / main
# -----------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    p.add_argument("--videos-csv", default=str(DEFAULT_VIDEOS_CSV))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--conf", type=float, default=CONF_THRES)
    p.add_argument("--track-conf-thresh", type=float, default=TRACK_CONF_THRESH)
    p.add_argument("--save-normalized", type=lambda s: s.lower() in ("true","1","yes"), default=True,
                   help="Save normalized .npy crops alongside JPEGs (default: True)")
    p.add_argument("--use-clahe", type=lambda s: s.lower() in ("true","1","yes"), default=USE_CLAHE_DEFAULT,
                   help="Apply CLAHE to resized ROI crops before normalization (default: False)")
    p.add_argument("--clahe-clip", type=float, default=CLAHE_CLIP, help="CLAHE clipLimit (default 2.0)")
    p.add_argument("--clahe-tile", type=str, default="8,8", help="CLAHE tileGridSize as 'w,h' (default '8,8')")
    args = p.parse_args()

    video_dir = Path(normpath_str(args.video_dir))
    videos_csv = Path(normpath_str(args.videos_csv))
    out_dir = Path(normpath_str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(out_dir / "orchestration.log")
    logging.info("Starting multi-video pipeline")
    logging.info(f"Weights: {args.weights}  Video dir: {video_dir}  Videos CSV: {videos_csv}  Out: {out_dir}")

    model = YOLO(str(args.weights))

    video_list = read_video_list_from_csv(videos_csv)
    seen = set()
    unique_videos = []
    for v in video_list:
        if v not in seen:
            seen.add(v)
            unique_videos.append(v)
    video_list = unique_videos
    if not video_list:
        logging.error("No videos found in CSV; exiting.")
        return

    # parse clahe tile argument
    try:
        tile_parts = tuple(int(x) for x in args.clahe_tile.split(","))
        if len(tile_parts) != 2:
            raise ValueError
    except Exception:
        logging.warning(f"Invalid --clahe-tile '{args.clahe_tile}', falling back to default {CLAHE_TILE}")
        tile_parts = CLAHE_TILE

    # Use tqdm progress bar in the terminal only (tqdm writes to stderr by default).
    for video_name in tqdm(video_list, desc="Processing videos", ncols=80):
        video_path = resolve_video_path(video_dir, video_name)
        logging.info(f"Resolved video entry '{video_name}' -> '{video_path}'")
        if not video_path.exists():
            logging.error(f"Video file not found: {video_path}; skipping.")
            continue
        video_stem = Path(video_name).stem
        per_video_out = out_dir / video_stem
        try:
            process_single_video(model, video_path, per_video_out, args.conf, args.track_conf_thresh, save_normalized=args.save_normalized, use_clahe=args.use_clahe)
        except Exception as e:
            logging.exception(f"Error processing {video_name}: {e}")
            continue

    logging.info("Multi-video pipeline finished.")

if __name__ == "__main__":
    main()

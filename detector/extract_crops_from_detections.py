"""
extract_crops_from_detections.py
Extract normalized CLAHE crops from detector output.

Saves crops to <out_root>/<video_stem>/mouth and <out_root>/<video_stem>/ues.
Prefer smoothed_xyxy if present in the detections JSON.
Run: python detector/extract_crops_from_detections.py --detections <json> --video <video.avi> --out <out_root> --canonical 224 224 --apply-clahe-rgb
"""
import argparse
import json
import os
from pathlib import Path
import cv2
import numpy as np

# -------------------------
# Defaults (can be overridden via CLI)
JPEG_QUALITY = 90
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
EPS = 1e-6
# -------------------------

def compute_video_reference_median(video_path: Path, sample_count=50, thumb_size=(64,64)):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    indices = np.linspace(0, max(total-1,0), min(sample_count, total), dtype=int)
    medians = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, f = cap.read()
        if not ret or f is None:
            continue
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim==3 else f
        thumb = cv2.resize(gray, thumb_size, interpolation=cv2.INTER_AREA)
        medians.append(float(np.median(thumb)))
    cap.release()
    return float(np.median(medians)) if medians else 128.0

def normalize_frame_to_reference(frame_gray, frame_median, ref_median):
    if frame_median <= 0:
        return frame_gray
    scale = (ref_median + EPS) / (frame_median + EPS)
    out = frame_gray.astype(np.float32) * scale
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def apply_clahe(img_gray):
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    return clahe.apply(img_gray)

# CLAHE helper for RGB luminance
_clahe_rgb = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)

def apply_clahe_to_rgb_array(arr_rgb):
    """
    arr_rgb: H,W,3 uint8 in RGB order
    returns: H,W,3 uint8 in RGB order with CLAHE applied on luminance (Y channel)
    """
    if arr_rgb.ndim != 3 or arr_rgb.shape[2] != 3:
        gray = arr_rgb if arr_rgb.ndim == 2 else cv2.cvtColor(arr_rgb, cv2.COLOR_BGR2GRAY)
        out_gray = _clahe_rgb.apply(gray)
        return cv2.cvtColor(out_gray, cv2.COLOR_GRAY2RGB)
    ycrcb = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[:, :, 0]
    y = _clahe_rgb.apply(y)
    ycrcb[:, :, 0] = y
    out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return out

def crop_and_process(frame, xyxy, canonical_wh, ref_median, apply_clahe_rgb=False):
    x1,y1,x2,y2 = [int(round(v)) for v in xyxy]
    h_img, w_img = frame.shape[:2]
    x1 = max(0, min(x1, w_img-1))
    y1 = max(0, min(y1, h_img-1))
    x2 = max(x1+8, min(x2, w_img))
    y2 = max(y1+8, min(y2, h_img))
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim==3 else crop
    frame_median = float(np.median(gray))
    gray = normalize_frame_to_reference(gray, frame_median, ref_median)
    gray = apply_clahe(gray)
    out_gray = cv2.resize(gray, canonical_wh, interpolation=cv2.INTER_LINEAR)
    if apply_clahe_rgb:
        # apply CLAHE on luminance of resized color crop, then convert to grayscale
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop.ndim==3 else cv2.cvtColor(np.stack([crop]*3, axis=-1), cv2.COLOR_BGR2RGB)
        crop_rgb_resized = cv2.resize(crop_rgb, canonical_wh, interpolation=cv2.INTER_LINEAR)
        crop_rgb_clahe = apply_clahe_to_rgb_array(crop_rgb_resized)
        out_gray = cv2.cvtColor(crop_rgb_clahe, cv2.COLOR_RGB2GRAY)
    return out_gray

def select_box_for_crop(boxes):
    """
    boxes: list of dicts with keys 'xyxy', 'conf', optionally 'smoothed_xyxy'
    returns (xyxy, conf) or (None, None)
    """
    if not boxes:
        return None, None
    # choose highest-confidence box
    best = max(boxes, key=lambda x: x.get("conf", 0.0))
    xy = best.get("smoothed_xyxy") or best.get("xyxy")
    return xy, best.get("conf", 0.0)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detections", default=None, help="Path to detections JSON")
    p.add_argument("--video", default=None, help="Path to video file")
    p.add_argument("--out", default=None, help="Output root for crops")
    p.add_argument("--canonical", nargs=2, type=int, default=None, help="Output crop size W H")
    p.add_argument("--jpeg-quality", type=int, default=None, help="JPEG quality for saved crops")
    p.add_argument("--apply-clahe-rgb", action="store_true", help="Apply CLAHE on RGB luminance after resize")
    args = p.parse_args()

    dets_path = Path(args.detections) if args.detections else None
    vpath = Path(args.video) if args.video else None
    out_root = Path(args.out) if args.out else Path("detections")
    canonical_wh = tuple(args.canonical) if args.canonical else (128,128)
    jpeg_quality = int(args.jpeg_quality) if args.jpeg_quality else JPEG_QUALITY
    apply_clahe_rgb = bool(args.apply_clahe_rgb)

    if dets_path is None or not dets_path.exists():
        raise FileNotFoundError(f"Detections JSON not found: {dets_path}")
    if vpath is None or not vpath.exists():
        raise FileNotFoundError(f"Video not found: {vpath}")

    dets = json.loads(dets_path.read_text())
    ref_median = compute_video_reference_median(vpath, sample_count=50)
    print("reference median:", ref_median)

    cap = cv2.VideoCapture(str(vpath))
    video_stem = Path(vpath).stem
    for item in dets:
        frame_idx = int(item["frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        video_out = out_root / video_stem
        mouth_out = video_out / "mouth"
        ues_out = video_out / "ues"
        mouth_out.mkdir(parents=True, exist_ok=True)
        ues_out.mkdir(parents=True, exist_ok=True)

        mouth_boxes = [b for b in item.get("boxes", []) if int(b.get("class", 0)) == 0]
        ues_boxes = [b for b in item.get("boxes", []) if int(b.get("class", 0)) == 1]

        mouth_xy, mouth_conf = select_box_for_crop(mouth_boxes)
        if mouth_xy:
            img = crop_and_process(frame, mouth_xy, canonical_wh, ref_median, apply_clahe_rgb=apply_clahe_rgb)
            fname = f"frame_{frame_idx:06d}_mouth.jpg"
            outp = mouth_out / fname
            cv2.imwrite(str(outp), img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

        ues_xy, ues_conf = select_box_for_crop(ues_boxes)
        if ues_xy:
            img = crop_and_process(frame, ues_xy, canonical_wh, ref_median, apply_clahe_rgb=apply_clahe_rgb)
            fname = f"frame_{frame_idx:06d}_ues.jpg"
            outp = ues_out / fname
            cv2.imwrite(str(outp), img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

    cap.release()
    print("Done extracting crops.")
    
if __name__ == "__main__":
    main()

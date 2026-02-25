"""
extract_crops_from_detections.py
Extract normalized CLAHE crops from detector output.

Edit the CONFIG block below to set DETECTIONS_JSON, VIDEO_PATH, and OUT_ROOT.
Run: python detector/extract_crops_from_detections.py
"""
import argparse, json
from pathlib import Path
import cv2
import numpy as np

# -------------------------
# CONFIGURATION
DETECTIONS_JSON = Path("detector_output_AD128.json")
VIDEO_PATH = Path(r"\\research.drive.wisc.edu\npconnor\ADStudy\VF AD Blinded\Early Tongue Training\AD128.avi")
OUT_ROOT = Path("./crops_from_detector")
CANONICAL = (128, 128)
JPEG_QUALITY = 90
CLAHE_CLIP = 2.0
CLAHE_TILE = (8,8)
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

def crop_and_process(frame, xyxy, canonical_wh, ref_median):
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
    out = cv2.resize(gray, canonical_wh, interpolation=cv2.INTER_LINEAR)
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detections", default=None)
    p.add_argument("--video", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--canonical", nargs=2, type=int, default=None)
    args = p.parse_args()

    dets_path = Path(args.detections) if args.detections else DETECTIONS_JSON
    vpath = Path(args.video) if args.video else VIDEO_PATH
    out_root = Path(args.out) if args.out else OUT_ROOT
    canonical_wh = tuple(args.canonical) if args.canonical else CANONICAL

    dets = json.loads(dets_path.read_text())
    ref_median = compute_video_reference_median(vpath, sample_count=50)
    print("reference median:", ref_median)

    cap = cv2.VideoCapture(str(vpath))
    for item in dets:
        frame_idx = int(item["frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        video_name = vpath.name
        video_out = out_root / video_name
        mouth_out = video_out / "mouth"
        ues_out = video_out / "ues"
        mouth_out.mkdir(parents=True, exist_ok=True)
        ues_out.mkdir(parents=True, exist_ok=True)

        mouth_boxes = [b for b in item["boxes"] if b["class"]==0]
        ues_boxes = [b for b in item["boxes"] if b["class"]==1]
        if mouth_boxes:
            best = max(mouth_boxes, key=lambda x: x["conf"])
            img = crop_and_process(frame, best["xyxy"], canonical_wh, ref_median)
            fname = f"frame_{frame_idx:06d}_mouth.jpg"
            outp = mouth_out / fname
            cv2.imwrite(str(outp), img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ues_boxes:
            best = max(ues_boxes, key=lambda x: x["conf"])
            img = crop_and_process(frame, best["xyxy"], canonical_wh, ref_median)
            fname = f"frame_{frame_idx:06d}_ues.jpg"
            outp = ues_out / fname
            cv2.imwrite(str(outp), img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

    cap.release()
    print("Done extracting crops.")

if __name__ == "__main__":
    main()

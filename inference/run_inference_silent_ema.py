"""
Silent per-frame inference with EMA smoothing.
Place this file in PROJECT_ROOT/inference and run with the venv python.

Outputs:
  runs/inference/AD128_roi/labels/raw/frame000001.txt      # raw detections
  runs/inference/AD128_roi/labels/smoothed/frame000001.txt # EMA-smoothed detections
  runs/inference/AD128_roi/annotated/                      # optional annotated frames
"""
from pathlib import Path
import shutil
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO

# ----------------- User parameters -----------------
# Confidence threshold for saving raw detections
CONF_THRESH = 0.25

# Class IDs
CLASS_NAMES = {0: "mouth", 1: "ues"}
CLASS_IDS = list(CLASS_NAMES.keys())

# EMA smoothing factor (alpha). 0.0 = no update, 1.0 = raw value only.
EMA_ALPHA = 0.35

# Whether to save annotated frames with smoothed boxes
SAVE_ANNOTATED = True

# Output experiment name
OUT_NAME = "AD128_roi"
# ---------------------------------------------------

# Resolve project paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Experiment checkpoint folder (adjust if your run folder differs)
exp = PROJECT_ROOT / "runs" / "detect" / "detector" / "models" / "yolov8_train_cpu"
weights_dir = exp / "weights"
src_ckpt = weights_dir / "best.pt"
if not src_ckpt.exists():
    src_ckpt = weights_dir / "last.pt"
if not src_ckpt.exists():
    raise FileNotFoundError(f"No checkpoint found in {weights_dir}")

# Copy checkpoint to temp to avoid partial-read
tmp_ckpt = Path(tempfile.gettempdir()) / f"ckpt_copy_{src_ckpt.stem}.pt"
shutil.copy2(src_ckpt, tmp_ckpt)

# Video path
unc_folder = r"\\research.drive.wisc.edu\npconnor\ADStudy\VF AD Blinded\Early Tongue Training"
# full video path (include filename)
video_path = Path(unc_folder) / "AD128.avi"

# check existence and raise a clear error if not accessible
if not video_path.exists():
    raise FileNotFoundError(f"Video not found or not accessible: {video_path}")


# Output folders
out_dir = PROJECT_ROOT / "runs" / "inference" / OUT_NAME
labels_raw = out_dir / "labels" / "raw"
labels_smooth = out_dir / "labels" / "smoothed"
annot_dir = out_dir / "annotated"
labels_raw.mkdir(parents=True, exist_ok=True)
labels_smooth.mkdir(parents=True, exist_ok=True)
if SAVE_ANNOTATED:
    annot_dir.mkdir(parents=True, exist_ok=True)

# Load model
model = YOLO(str(tmp_ckpt))

# Helper: write detections in YOLO txt format (class x_center y_center w h conf)
def write_txt(path: Path, dets):
    with open(path, "w") as f:
        for d in dets:
            # d: (cls, xc, yc, w, h, conf)
            f.write(f"{int(d[0])} {d[1]:.6f} {d[2]:.6f} {d[3]:.6f} {d[4]:.6f} {d[5]:.4f}\n")

# Initialize EMA state per class: dict[class] = [xc, yc, w, h] or None
ema_state = {c: None for c in CLASS_IDS}

# Video capture
cap = cv2.VideoCapture(str(video_path))
frame_idx = 0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Loop frames silently
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run model on the frame (returns Results)
    results = model(frame, device="cpu", conf=CONF_THRESH, verbose=False)
    res = results[0]

    # Parse raw detections
    raw_dets = []
    if res.boxes is not None and len(res.boxes) > 0:
        xywhn = res.boxes.xywhn.cpu().numpy()  # normalized x,y,w,h
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        for (x, y, w, h), cls, conf in zip(xywhn, cls_ids, confs):
            if int(cls) in CLASS_IDS and conf >= CONF_THRESH:
                raw_dets.append((int(cls), float(x), float(y), float(w), float(h), float(conf)))

    # Save raw detections
    raw_path = labels_raw / f"frame{frame_idx:06d}.txt"
    write_txt(raw_path, raw_dets)

    # Build best-per-class from raw_dets (highest conf)
    best = {}
    for d in raw_dets:
        cls = d[0]
        if cls not in best or d[5] > best[cls][5]:
            best[cls] = d

    # Apply EMA smoothing per class
    smoothed_dets = []
    for cls in CLASS_IDS:
        if cls in best:
            _, xc, yc, w, h, conf = best[cls]
            if ema_state[cls] is None:
                # initialize EMA with first observed value
                ema_state[cls] = np.array([xc, yc, w, h], dtype=float)
            else:
                # EMA update: s_t = alpha * x_t + (1-alpha) * s_{t-1}
                ema_state[cls] = EMA_ALPHA * np.array([xc, yc, w, h], dtype=float) + (1.0 - EMA_ALPHA) * ema_state[cls]
            s = ema_state[cls]
            smoothed_dets.append((cls, float(s[0]), float(s[1]), float(s[2]), float(s[3]), float(conf)))
        else:
            # No detection this frame: keep previous EMA state if exists (write it), else skip
            if ema_state[cls] is not None:
                s = ema_state[cls]
                smoothed_dets.append((cls, float(s[0]), float(s[1]), float(s[2]), float(s[3]), 0.0))
            # else: nothing to write for this class this frame

    # Save smoothed detections
    smooth_path = labels_smooth / f"frame{frame_idx:06d}.txt"
    write_txt(smooth_path, smoothed_dets)

    # Optionally save annotated frame using smoothed boxes (draw rectangles)
    if SAVE_ANNOTATED:
        vis = frame.copy()
        for d in smoothed_dets:
            cls, xc, yc, w, h, conf = d
            # convert normalized xywh to pixel x1,y1,x2,y2
            x_c = xc * width; y_c = yc * height
            bw = w * width; bh = h * height
            x1 = int(x_c - bw / 2); y1 = int(y_c - bh / 2)
            x2 = int(x_c + bw / 2); y2 = int(y_c + bh / 2)
            color = (0, 255, 0) if conf > 0 else (0, 165, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{CLASS_NAMES.get(cls,cls)} {conf:.2f}"
            cv2.putText(vis, label, (max(0,x1), max(12,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.imwrite(str(annot_dir / f"frame{frame_idx:06d}.jpg"), vis)

    frame_idx += 1

cap.release()
print(f"Silent inference finished. Frames processed: {frame_idx}")
print("Raw labels:", labels_raw)
print("Smoothed labels:", labels_smooth)
if SAVE_ANNOTATED:
    print("Annotated frames:", annot_dir)

#!/usr/bin/env python3
"""
detector/convert_to_yolo.py

Create a YOLOv8-style dataset from event_rois.json (dataset mode) or extract
per-video frames+labels for a single video (per-video mode) so run_all.py can
prepare inputs before inference.

Usage:
  # Dataset mode (process all videos in event_rois.json)
  python detector/convert_to_yolo.py --events event_rois.json --videos-dir "\\\\research.drive.wisc.edu\\pconnor\\ADStudy\\VF AD Blinded\\Early Tongue Training" --out detector/yolo_dataset

  # Per-video mode (prepare frames+labels for a single video; compatible with run_all.py)
  python detector/convert_to_yolo.py --video "videos/AD128.avi" --events event_rois.json --out detector/frames

Behavior:
- Dataset mode writes a YOLOv8-style layout:
    out/
      images/train, images/val
      labels/train, labels/val
      data.yaml
- Per-video mode writes:
    out/<stem>/images/<frame>.jpg
    out/<stem>/labels/<frame>.txt
  and a small marker file out/<stem>/READY to indicate conversion completed.

Edit defaults in the CONFIG block below if you prefer different defaults.
"""
from pathlib import Path
import argparse
import json
import random
import cv2
import sys

# -------------------------
# CONFIGURATION (defaults)
DEFAULT_VIDEO_DIR = Path(r"\\research.drive.wisc.edu\pconnor\ADStudy\VF AD Blinded\Early Tongue Training")
DEFAULT_OUT_DIR = Path("detector/yolo_dataset")
DEFAULT_EVENTS_JSON = Path("event_rois.json")
DEFAULT_VAL_FRAC = 0.15
DEFAULT_SEED = 42
# -------------------------

CLASS_MAP = {"mouth": 0, "ues": 1}

def load_json(p: Path):
    return json.loads(p.read_text())

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def box_to_yolo(box, img_w, img_h):
    # input box: [x, y, w, h] in pixels (x,y top-left)
    x, y, w, h = [float(v) for v in box]
    xc = x + w / 2.0
    yc = y + h / 2.0
    return xc / img_w, yc / img_h, w / img_w, h / img_h

def resolve_video_path(vname: str, videos_dir: Path):
    p = Path(vname)
    if p.exists():
        return p
    if Path(vname).suffix:
        cand = videos_dir / vname
        if cand.exists():
            return cand
    stem = Path(vname).stem
    for ext in (".avi", ".mp4", ".mov", ".mkv"):
        cand = videos_dir / (stem + ext)
        if cand.exists():
            return cand
    vname_lower = stem.lower()
    for f in videos_dir.rglob("*"):
        if not f.is_file():
            continue
        if f.stem.lower().startswith(vname_lower):
            return f
    return None

def write_frame_and_label(vpath: Path, frame_idx: int, out_img: Path, out_lbl: Path, mouth_box, ues_box):
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        return False, "cannot_open_video"
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return False, "frame_read_failed"
    h, w = frame.shape[:2]
    ensure_dir(out_img.parent)
    ensure_dir(out_lbl.parent)
    cv2.imwrite(str(out_img), frame)
    lines = []
    if mouth_box:
        try:
            xc, yc, ww, hh = box_to_yolo(mouth_box, w, h)
            lines.append(f"{CLASS_MAP['mouth']} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
        except Exception:
            pass
    if ues_box:
        try:
            xc, yc, ww, hh = box_to_yolo(ues_box, w, h)
            lines.append(f"{CLASS_MAP['ues']} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
        except Exception:
            pass
    out_lbl.write_text("\n".join(lines))
    return True, None

def dataset_mode(events_path: Path, videos_dir: Path, out: Path, val_frac: float, seed: int):
    events = load_json(events_path)
    entries_by_video = {}
    for vname, items in events.items():
        if vname == "__event_slot_map__":
            continue
        entries_by_video.setdefault(vname, [])
        for e in items:
            entries_by_video[vname].append({
                "frame": int(e.get("frame_index")),
                "mouth": e.get("mouth"),
                "ues": e.get("ues")
            })

    random.seed(seed)
    vids = list(entries_by_video.keys())
    random.shuffle(vids)
    n_val_vids = max(1, int(len(vids) * val_frac))
    val_vids = set(vids[:n_val_vids])

    images_train = out / "images" / "train"
    images_val = out / "images" / "val"
    labels_train = out / "labels" / "train"
    labels_val = out / "labels" / "val"
    ensure_dir(images_train); ensure_dir(images_val); ensure_dir(labels_train); ensure_dir(labels_val)

    written = 0
    for v in vids:
        split = "val" if v in val_vids else "train"
        vpath = resolve_video_path(v, videos_dir)
        if vpath is None:
            print(f"[WARN] video not found: {v}")
            continue
        print(f"[INFO] using video {vpath} for {v} (split={split})")
        for e in entries_by_video[v]:
            frame_idx = e["frame"]
            fname = f"{Path(vpath).stem}_{frame_idx:06d}.jpg"
            if split == "val":
                img_out = images_val / fname
                lbl_out = labels_val / (fname.replace(".jpg", ".txt"))
            else:
                img_out = images_train / fname
                lbl_out = labels_train / (fname.replace(".jpg", ".txt"))
            ok, err = write_frame_and_label(vpath, frame_idx, img_out, lbl_out, e.get("mouth"), e.get("ues"))
            if ok:
                written += 1
            else:
                print(f"[WARN] failed to write frame {frame_idx} for {v}: {err}")

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"train: {str(images_train.resolve())}\nval: {str(images_val.resolve())}\nnc: 2\nnames: ['mouth','ues']\n"
    )
    print(f"Wrote {written} labeled frames to {out}")

def per_video_mode(video_arg: str, events_path: Path, videos_dir: Path, out: Path):
    events = load_json(events_path)
    # events keys may be video stems or filenames; find matching key for this video
    vpath = Path(video_arg)
    if vpath.exists():
        stem = vpath.stem
    else:
        stem = Path(video_arg).stem

    # find the key in events that best matches stem (case-insensitive startswith)
    key = None
    for k in events.keys():
        if k == "__event_slot_map__":
            continue
        if Path(k).stem.lower() == stem.lower() or k.lower().startswith(stem.lower()):
            key = k
            break
    if key is None:
        # fallback: try exact stem match
        if stem in events:
            key = stem
    if key is None:
        print(f"[WARN] no events found for video key matching '{stem}' in {events_path}")
        # still attempt to resolve video path and create empty READY marker
        vpath_resolved = resolve_video_path(video_arg, videos_dir)
        if vpath_resolved is None:
            print(f"[ERROR] cannot resolve video path for '{video_arg}'")
            return
        out_dir = out / Path(vpath_resolved).stem
        ensure_dir(out_dir / "images")
        ensure_dir(out_dir / "labels")
        (out_dir / "READY").write_text("no_events")
        print(f"[INFO] created READY marker at {out_dir / 'READY'} (no events)")
        return

    entries = []
    for e in events.get(key, []):
        entries.append({
            "frame": int(e.get("frame_index")),
            "mouth": e.get("mouth"),
            "ues": e.get("ues")
        })

    vpath_resolved = resolve_video_path(key, videos_dir)
    if vpath_resolved is None:
        # try using provided video_arg as path
        if Path(video_arg).exists():
            vpath_resolved = Path(video_arg)
        else:
            print(f"[ERROR] cannot resolve video path for key '{key}' or '{video_arg}'")
            return

    out_dir = out / Path(vpath_resolved).stem
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    ensure_dir(images_dir); ensure_dir(labels_dir)

    written = 0
    for e in entries:
        frame_idx = e["frame"]
        fname = f"{Path(vpath_resolved).stem}_{frame_idx:06d}.jpg"
        img_out = images_dir / fname
        lbl_out = labels_dir / (fname.replace(".jpg", ".txt"))
        ok, err = write_frame_and_label(vpath_resolved, frame_idx, img_out, lbl_out, e.get("mouth"), e.get("ues"))
        if ok:
            written += 1
        else:
            print(f"[WARN] failed to write frame {frame_idx} for {key}: {err}")

    # marker file to indicate conversion done for this video
    (out_dir / "READY").write_text("ok")
    print(f"Wrote {written} frames+labels to {out_dir} and created READY marker")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--events", default=str(DEFAULT_EVENTS_JSON), help="Path to event_rois.json")
    p.add_argument("--videos-dir", default=None, help="Directory containing videos (overrides CONFIG)")
    p.add_argument("--out", default=None, help="Output directory (overrides CONFIG)")
    p.add_argument("--val-frac", type=float, default=None, help="Validation fraction for dataset mode")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--video", default=None, help="If provided, run per-video mode for this single video (path or key)")
    args = p.parse_args()

    events_path = Path(args.events)
    if not events_path.exists():
        print(f"[ERROR] events JSON not found: {events_path}")
        sys.exit(2)

    videos_dir = Path(args.videos_dir) if args.videos_dir else DEFAULT_VIDEO_DIR
    out = Path(args.out) if args.out else DEFAULT_OUT_DIR
    val_frac = args.val_frac if args.val_frac is not None else DEFAULT_VAL_FRAC
    seed = args.seed if args.seed is not None else DEFAULT_SEED

    ensure_dir(out)

    if args.video:
        per_video_mode(args.video, events_path, videos_dir, out)
    else:
        dataset_mode(events_path, videos_dir, out, val_frac, seed)

if __name__ == "__main__":
    main()

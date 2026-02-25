"""
convert_to_yolo.py
Create a YOLOv8-style dataset from event_rois.json.

Edit the CONFIG block below to set VIDEO_DIR and OUT_DIR.
Run: python detector/convert_to_yolo.py
"""
import json, random, argparse
from pathlib import Path
import cv2

# -------------------------
# CONFIGURATION
VIDEO_DIR = Path(r"\\research.drive.wisc.edu\npconnor\ADStudy\VF AD Blinded\Early Tongue Training")
OUT_DIR = Path("detector/yolo_dataset")
EVENTS_JSON = Path("event_rois.json")
VAL_FRAC = 0.15
SEED = 42
# -------------------------

CLASS_MAP = {"mouth": 0, "ues": 1}

def load_json(p): return json.loads(Path(p).read_text())
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def box_to_yolo(box, img_w, img_h):
    x, y, w, h = [float(v) for v in box]
    xc = x + w/2.0
    yc = y + h/2.0
    return xc/img_w, yc/img_h, w/img_w, h/img_h

def resolve_video_path(vname, videos_dir: Path):
    # If vname is already a path that exists, return it
    p = Path(vname)
    if p.exists():
        return p

    # If vname already contains an extension, try exact filename under videos_dir
    if Path(vname).suffix:
        cand = videos_dir / vname
        if cand.exists():
            return cand

    # Try common extensions appended to the stem (handles keys without extension)
    stem = Path(vname).stem
    for ext in (".avi", ".mp4", ".mov", ".mkv"):
        cand = videos_dir / (stem + ext)
        if cand.exists():
            return cand

    # Fallback: recursive search for files whose stem starts with the key (case-insensitive)
    vname_lower = stem.lower()
    for f in videos_dir.rglob("*"):
        if not f.is_file():
            continue
        if f.stem.lower().startswith(vname_lower):
            return f

    return None


def write_frame_and_label(vpath, frame_idx, out_img, out_lbl, mouth_box, ues_box):
    cap = cv2.VideoCapture(str(vpath))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return False
    h, w = frame.shape[:2]
    cv2.imwrite(str(out_img), frame)
    lines = []
    if mouth_box:
        xc, yc, ww, hh = box_to_yolo(mouth_box, w, h)
        lines.append(f"{CLASS_MAP['mouth']} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
    if ues_box:
        xc, yc, ww, hh = box_to_yolo(ues_box, w, h)
        lines.append(f"{CLASS_MAP['ues']} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
    out_lbl.write_text("\n".join(lines))
    return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--events", default=str(EVENTS_JSON))
    p.add_argument("--videos-dir", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--val-frac", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    events_path = Path(args.events)
    videos_dir = Path(args.videos_dir) if args.videos_dir else VIDEO_DIR
    out = Path(args.out) if args.out else OUT_DIR
    val_frac = args.val_frac if args.val_frac is not None else VAL_FRAC
    seed = args.seed if args.seed is not None else SEED

    images_train = out / "images" / "train"
    images_val = out / "images" / "val"
    labels_train = out / "labels" / "train"
    labels_val = out / "labels" / "val"
    ensure_dir(images_train); ensure_dir(images_val); ensure_dir(labels_train); ensure_dir(labels_val)

    events = load_json(events_path)
    entries_by_video = {}
    for vname, items in events.items():
        if vname == "__event_slot_map__": continue
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
            fname = f"{v}_{frame_idx:06d}.jpg"
            if split == "val":
                img_out = images_val / fname
                lbl_out = labels_val / (fname.replace(".jpg", ".txt"))
            else:
                img_out = images_train / fname
                lbl_out = labels_train / (fname.replace(".jpg", ".txt"))
            ok = write_frame_and_label(vpath, frame_idx, img_out, lbl_out, e.get("mouth"), e.get("ues"))
            if ok:
                written += 1

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"train: {str(images_train.resolve())}\nval: {str(images_val.resolve())}\nnc: 2\nnames: ['mouth','ues']\n"
    )
    print(f"Wrote {written} labeled frames to {out}")

if __name__ == "__main__":
    main()

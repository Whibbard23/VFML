#!/usr/bin/env python3
"""
Generate two consolidated EMA-smoothed .txt files per video (mouth and ues),
saving outputs under runs/inference/<video_stem>_roi with the same subfolder layout.

Usage example:
  python inference/run_inference_smoothed_two_txts_per_video.py \
    --csv event_csvs/assembly_1_train_events.csv \
    --data-root . \
    --out runs/inference \
    --weights runs/detect/detector/models/yolov8_train_cpu/weights/best.pt \
    --conf 0.25 --ema-alpha 0.35 --annotate
"""
from pathlib import Path
import argparse
import shutil
import tempfile
import csv
import cv2
import numpy as np
from ultralytics import YOLO

CLASS_MAP = {0: "mouth", 1: "ues"}  # adjust if your detector uses different ids

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV listing videos (column 'video')")
    p.add_argument("--data-root", default=".", help="root to resolve relative video paths")
    p.add_argument("--out", default="runs/inference", help="root output folder; per-video folders created here")
    p.add_argument("--weights", required=True, help="path to YOLO checkpoint (best.pt or last.pt)")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold for raw detections")
    p.add_argument("--ema-alpha", type=float, default=0.35, help="EMA alpha (0..1)")
    p.add_argument("--annotate", action="store_true", help="save annotated frames with smoothed boxes")
    p.add_argument("--force", action="store_true", help="reprocess videos even if outputs exist")
    p.add_argument("--device", default="cpu", help="device for YOLO inference (cpu or cuda)")
    p.add_argument("--write-raw-per-frame", action="store_true", help="also write per-frame raw txts (optional)")
    p.add_argument("--video-root", default=None, help="optional folder that contains the .avi files; CSV video values are resolved against this root when not absolute")
    return p.parse_args()

def gather_videos_from_csv(csv_path: Path, data_root: Path, video_root: Path | None):
    import csv
    vids = []

    print(f"Reading CSV: {csv_path}  exists: {csv_path.exists()}")
    # use utf-8-sig to handle BOM if present
    with csv_path.open("r", newline="", encoding="utf-8-sig") as fh:
        # peek first few raw lines for debugging
        raw_preview = []
        for _ in range(3):
            pos = fh.tell()
            line = fh.readline()
            if not line:
                break
            raw_preview.append(line.rstrip("\n"))
            # rewind to let DictReader read from start
            fh.seek(pos)
        if raw_preview:
            print("CSV raw preview (first 3 lines):")
            for L in raw_preview:
                print("  ", L)

        reader = csv.DictReader(fh)
        # normalize fieldnames (strip BOM, whitespace, quotes)
        if reader.fieldnames:
            norm_fns = [fn.strip().strip('"').strip("'").lstrip("\ufeff") for fn in reader.fieldnames]
            reader.fieldnames = norm_fns
            print("Parsed CSV fieldnames:", norm_fns)
        else:
            print("Warning: CSV has no header/fieldnames.")
        for r in reader:
            # accept multiple possible column names
            v = (r.get("video") or r.get("video_path") or r.get("filename") or "").strip()
            if not v:
                continue
            # defensive cleanup of quotes and whitespace
            v = v.strip().strip('"').strip("'").strip()
            p = Path(v)
            if not p.is_absolute():
                base = Path(video_root) if video_root is not None else data_root
                p = (base / v).resolve()
            vids.append(str(p))

    # unique preserve order
    seen = set()
    out = []
    for v in vids:
        if v not in seen:
            seen.add(v)
            out.append(Path(v))

    print(f"CSV parsed: {len(vids)} rows, {len(out)} unique videos. First 20 resolved paths:")
    for i, p in enumerate(out[:20]):
        print(f"  {i+1:02d}: {p}  exists: {p.exists()}")
    return out



def write_per_frame_txt(path: Path, dets):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for d in dets:
            f.write(f"{int(d[0])} {d[1]:.6f} {d[2]:.6f} {d[3]:.6f} {d[4]:.6f} {d[5]:.4f}\n")

def process_video_consolidated_per_video(video_path: Path, model: YOLO, out_root: Path, conf_thresh: float,
                                         ema_alpha: float, annotate: bool, force: bool, device: str,
                                         write_raw_per_frame: bool):
    """
    For a single video, create a per-video folder named <video_stem>_roi under out_root,
    then create labels/raw, labels/smoothed, annotated, analysis subfolders and write outputs there.
    """
    video_stem = video_path.stem
    video_out = out_root / f"{video_stem}_roi"
    labels_raw_dir = video_out / "labels" / "raw" / video_stem
    labels_smooth_dir = video_out / "labels" / "smoothed"
    annot_dir = video_out / "annotated" / video_stem if annotate else None
    analysis_dir = video_out / "analysis"
    # create base folders
    labels_raw_dir.mkdir(parents=True, exist_ok=True)
    labels_smooth_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if annotate:
        annot_dir.mkdir(parents=True, exist_ok=True)

    # consolidated output files (one per class per video)
    mouth_file = labels_smooth_dir / f"{video_stem}_mouth.txt"
    ues_file = labels_smooth_dir / f"{video_stem}_ues.txt"

    # skip if both exist and not forcing
    if not force and mouth_file.exists() and ues_file.exists():
        return {"video": str(video_path), "status": "skipped_exists"}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video": str(video_path), "status": "open_error"}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_idx = 0

    # EMA state per class id: dict[class] = np.array([xc,yc,w,h]) or None
    ema_state = {cls: None for cls in CLASS_MAP.keys()}

    # open consolidated files for streaming write
    mouth_fh = mouth_file.open("w")
    ues_fh = ues_file.open("w")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            raw_path = labels_raw_dir / f"frame{frame_idx:06d}.txt"

            # run model
            results = model(frame, device=device, conf=conf_thresh, verbose=False)
            res = results[0]

            raw_dets = []
            if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
                try:
                    xywhn = res.boxes.xywhn.cpu().numpy()  # normalized x,y,w,h
                    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
                    confs = res.boxes.conf.cpu().numpy()
                    for (x, y, w, h), cls, conf in zip(xywhn, cls_ids, confs):
                        if conf >= conf_thresh:
                            raw_dets.append((int(cls), float(x), float(y), float(w), float(h), float(conf)))
                except Exception:
                    # fallback: iterate boxes if attributes differ
                    for b in res.boxes:
                        try:
                            cls = int(b.cls.cpu().numpy())
                            conf = float(b.conf.cpu().numpy())
                            xywh = b.xywhn.cpu().numpy().tolist()[0]
                            x,y,w,h = xywh
                            if conf >= conf_thresh:
                                raw_dets.append((cls, float(x), float(y), float(w), float(h), conf))
                        except Exception:
                            continue

            # optionally write per-frame raw detections
            if write_raw_per_frame:
                write_per_frame_txt(raw_path, raw_dets)

            # build best-per-class by highest conf
            best = {}
            for d in raw_dets:
                cls = d[0]
                if cls not in best or d[5] > best[cls][5]:
                    best[cls] = d

            # update EMA per class and produce smoothed per-frame entries
            smoothed_per_frame = {}
            classes_to_consider = set(list(ema_state.keys()) + list(best.keys()))
            for cls in sorted(classes_to_consider):
                if cls in best:
                    _, xc, yc, w, h, conf = best[cls]
                    if ema_state[cls] is None:
                        ema_state[cls] = np.array([xc, yc, w, h], dtype=float)
                    else:
                        ema_state[cls] = ema_alpha * np.array([xc, yc, w, h], dtype=float) + (1.0 - ema_alpha) * ema_state[cls]
                    s = ema_state[cls]
                    smoothed_per_frame[cls] = (float(s[0]), float(s[1]), float(s[2]), float(s[3]), float(conf))
                else:
                    # no detection this frame: if we have previous EMA state, write it with conf=0
                    if ema_state.get(cls) is not None:
                        s = ema_state[cls]
                        smoothed_per_frame[cls] = (float(s[0]), float(s[1]), float(s[2]), float(s[3]), 0.0)
                    else:
                        smoothed_per_frame[cls] = None

            # write one line per class into consolidated files
            mouth_entry = smoothed_per_frame.get(0)
            if mouth_entry is None:
                mouth_fh.write("\n")
            else:
                mouth_fh.write(f"{mouth_entry[0]:.6f} {mouth_entry[1]:.6f} {mouth_entry[2]:.6f} {mouth_entry[3]:.6f} {mouth_entry[4]:.4f}\n")

            ues_entry = smoothed_per_frame.get(1)
            if ues_entry is None:
                ues_fh.write("\n")
            else:
                ues_fh.write(f"{ues_entry[0]:.6f} {ues_entry[1]:.6f} {ues_entry[2]:.6f} {ues_entry[3]:.6f} {ues_entry[4]:.4f}\n")

            # optionally save annotated frame using smoothed boxes
            if annotate:
                vis = frame.copy()
                for cls, entry in smoothed_per_frame.items():
                    if entry is None:
                        continue
                    xc, yc, w, h, conf = entry
                    x_c = xc * width; y_c = yc * height
                    bw = w * width; bh = h * height
                    x1 = int(round(x_c - bw / 2)); y1 = int(round(y_c - bh / 2))
                    x2 = int(round(x_c + bw / 2)); y2 = int(round(y_c + bh / 2))
                    color = (0, 255, 0) if conf > 0 else (0, 165, 255)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    label = f"{CLASS_MAP.get(cls,cls)} {conf:.2f}"
                    cv2.putText(vis, label, (max(0, x1), max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                annot_path = (video_out / "annotated" / video_stem / f"frame{frame_idx:06d}.jpg")
                annot_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(annot_path), vis)

            frame_idx += 1

    finally:
        mouth_fh.close()
        ues_fh.close()
        cap.release()

    return {"video": str(video_path), "status": f"processed_frames:{frame_idx}", "out_folder": str(video_out)}

def main():
    args = parse_args()
    csv_path = Path(args.csv).resolve()
    data_root = Path(args.data_root).resolve()
    video_root = Path(args.video_root).resolve() if args.video_root else None

    # Validate video_root early so users get a clear error instead of many missing-path messages
    if video_root is not None:
        if not video_root.exists():
            raise FileNotFoundError(f"--video-root does not exist: {video_root}")
        if not video_root.is_dir():
            raise NotADirectoryError(f"--video-root is not a directory: {video_root}")


    videos = gather_videos_from_csv(csv_path, data_root, video_root)

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # resolve weights: copy to temp to avoid partial-read issues
    src_ckpt = Path(args.weights).resolve()
    if not src_ckpt.exists():
        raise FileNotFoundError(f"weights not found: {src_ckpt}")
    tmp_ckpt = Path(tempfile.gettempdir()) / f"ckpt_copy_{src_ckpt.stem}.pt"
    shutil.copy2(src_ckpt, tmp_ckpt)

    if not videos:
        print("No videos found in CSV.")
        return

    print(f"Found {len(videos)} unique videos. Loading model from {tmp_ckpt}")
    model = YOLO(str(tmp_ckpt))

    summary = {"processed":0,"skipped":0,"errors":0}
    for v in videos:
        if not v.exists():
            print(f"Missing video: {v}")
            summary["errors"] += 1
            continue
        print(f"Processing: {v}")
        res = process_video_consolidated_per_video(v, model, out_root, args.conf, args.ema_alpha, args.annotate, args.force, args.device, args.write_raw_per_frame)
        print(res.get("video"), res.get("status"), "->", res.get("out_folder", ""))
        if res["status"].startswith("processed_frames"):
            summary["processed"] += 1
        elif res["status"] == "open_error":
            summary["errors"] += 1
        else:
            summary["skipped"] += 1

    print("Done. Summary:", summary)
    print("Outputs written under:", out_root)

if __name__ == "__main__":
    main()

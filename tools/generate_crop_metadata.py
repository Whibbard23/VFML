#!/usr/bin/env python3
"""
tools/generate_crop_metadata.py

Scan extracted crops under a crops root (default: detections), write a per-video
metadata.json in each video folder, and produce a combined CSV manifest.

Per-video metadata.json format:
{
  "video": "<video_stem>",
  "crops": [
    {
      "frame": 123,
      "roi": "mouth",
      "filename": "frame_000123_mouth.jpg",
      "crop_path": "/abs/path/to/detections/AD128/mouth/frame_000123_mouth.jpg",
      "xyxy_used": [x1,y1,x2,y2],   # optional
      "conf": 0.95,                # optional
      "smoothed": true             # optional
    },
    ...
  ]
}

Combined CSV columns:
video,frame,roi,filename,crop_path,xyxy_used,conf,smoothed

Usage:
  python tools/generate_crop_metadata.py
  python tools/generate_crop_metadata.py --crops-root detections --dets-dir detector_outputs --manifest-csv event_csvs/crops_manifest.csv
"""
import argparse
import json
import csv
from pathlib import Path
from typing import Dict, Optional

def load_detections_map(dets_dir: Path, stem: str) -> Dict[int, Dict[int, Dict[str, object]]]:
    """
    Load smoothed or raw detections JSON for a video stem and return a mapping:
      detections_map[frame_index][class] = {"xyxy": [...], "conf": float, "smoothed": bool}
    If no JSON exists, returns empty dict.
    """
    candidates = [
        dets_dir / f"smoothed_{stem}.json",
        dets_dir / f"{stem}.json"
    ]
    for p in candidates:
        if p.exists():
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            out = {}
            for item in data:
                try:
                    fi = int(item.get("frame"))
                except Exception:
                    continue
                out.setdefault(fi, {})
                for b in item.get("boxes", []):
                    try:
                        cls = int(b.get("class", 0))
                    except Exception:
                        cls = 0
                    xy = b.get("smoothed_xyxy") or b.get("xyxy")
                    conf = b.get("conf", None)
                    out[fi][cls] = {"xyxy": xy, "conf": conf, "smoothed": ("smoothed_xyxy" in b)}
            return out
    return {}

def parse_frame_index_from_name(name: str) -> Optional[int]:
    """
    Expect filename like frame_000123_mouth.jpg or frame_000123_ues.jpg
    Returns integer frame index or None if parsing fails.
    """
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except Exception:
        return None

def gather_crops_for_video(video_dir: Path, dets_map: Dict[int, Dict[int, Dict[str, object]]]) -> list:
    crops = []
    for roi, cls in (("mouth", 0), ("ues", 1)):
        sub = video_dir / roi
        if not sub.exists():
            continue
        for p in sorted(sub.glob(f"frame_*_{roi}.jpg")):
            frame_idx = parse_frame_index_from_name(p.name)
            if frame_idx is None:
                continue
            entry = {
                "frame": frame_idx,
                "roi": roi,
                "filename": p.name,
                "crop_path": str(p.resolve())
            }
            det_for_frame = dets_map.get(frame_idx, {})
            det = det_for_frame.get(cls)
            if det:
                entry["xyxy_used"] = det.get("xyxy")
                if det.get("conf") is not None:
                    try:
                        entry["conf"] = float(det.get("conf"))
                    except Exception:
                        entry["conf"] = det.get("conf")
                entry["smoothed"] = bool(det.get("smoothed", False))
            crops.append(entry)
    crops.sort(key=lambda x: (x["frame"], x["roi"]))
    return crops

def write_metadata(video_dir: Path, metadata: dict, out_name: str = "metadata.json"):
    outp = video_dir / out_name
    outp.write_text(json.dumps(metadata, indent=2))
    return outp

def write_combined_csv(rows: list, out_csv: Path):
    fieldnames = ["video", "frame", "roi", "filename", "crop_path", "xyxy_used", "conf", "smoothed"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            # Ensure single-line cells; xyxy_used as JSON string if present
            row = {
                "video": r.get("video", ""),
                "frame": r.get("frame", ""),
                "roi": r.get("roi", ""),
                "filename": r.get("filename", ""),
                "crop_path": r.get("crop_path", ""),
                "xyxy_used": json.dumps(r.get("xyxy_used")) if r.get("xyxy_used") is not None else "",
                "conf": r.get("conf", ""),
                "smoothed": r.get("smoothed", "")
            }
            writer.writerow(row)
    return out_csv

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crops-root", default="detections", help="Root folder containing per-video crop folders")
    p.add_argument("--dets-dir", default="detector_outputs", help="Directory containing detector JSONs (optional)")
    p.add_argument("--out-name", default="metadata.json", help="Filename to write inside each video folder")
    p.add_argument("--manifest-csv", default="event_csvs/crops_manifest.csv", help="Combined CSV manifest path")
    args = p.parse_args()

    crops_root = Path(args.crops_root)
    dets_dir = Path(args.dets_dir)

    if not crops_root.exists():
        raise FileNotFoundError(f"Crops root not found: {crops_root}")

    combined_rows = []
    for video_dir in sorted(crops_root.iterdir()):
        if not video_dir.is_dir():
            continue
        stem = video_dir.name
        dets_map = {}
        if dets_dir.exists():
            dets_map = load_detections_map(dets_dir, stem)
        crops = gather_crops_for_video(video_dir, dets_map)
        metadata = {"video": stem, "crops": crops}
        outp = write_metadata(video_dir, metadata, out_name=args.out_name)
        print(f"Wrote metadata for {stem}: {outp} ({len(crops)} entries)")
        for c in crops:
            row = {
                "video": stem,
                "frame": c.get("frame"),
                "roi": c.get("roi"),
                "filename": c.get("filename"),
                "crop_path": c.get("crop_path"),
                "xyxy_used": c.get("xyxy_used"),
                "conf": c.get("conf", ""),
                "smoothed": c.get("smoothed", "")
            }
            combined_rows.append(row)

    manifest_path = Path(args.manifest_csv)
    write_combined_csv(combined_rows, manifest_path)
    print(f"Wrote combined CSV manifest: {manifest_path} ({len(combined_rows)} rows)")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_all.py
Iterate detector JSONs and run the extractor for each video, forwarding --apply-clahe-rgb.
Usage example:
  python detector/run_all.py --detections-dir detector_outputs --videos-dir videos --out-root crops_from_detector --canonical 224 224 --apply-clahe-rgb
"""
import argparse
import subprocess
from pathlib import Path
import shlex
import sys

def run_extractor(djson, video, out_root, canonical_w, canonical_h, apply_clahe_rgb, jpeg_quality=None, log_dir=None):
    cmd = [sys.executable, str(Path(__file__).parent / "extract_crops_from_detections.py"),
           "--detections", str(djson),
           "--video", str(video),
           "--out", str(out_root),
           "--canonical", str(canonical_w), str(canonical_h)]
    if apply_clahe_rgb:
        cmd.append("--apply-clahe-rgb")
    if jpeg_quality is not None:
        cmd.extend(["--jpeg-quality", str(jpeg_quality)])
    log_path = None
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / (Path(video).stem + ".log")
        with open(log_path, "wb") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
            proc.wait()
            return proc.returncode, str(log_path)
    else:
        proc = subprocess.Popen(cmd)
        proc.wait()
        return proc.returncode, None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detections-dir", required=True, help="Directory containing detector JSON files (*.json)")
    p.add_argument("--videos-dir", required=True, help="Directory containing .avi videos")
    p.add_argument("--out-root", default="crops_from_detector", help="Output root for crops")
    p.add_argument("--canonical", nargs=2, type=int, default=[128,128], help="Output crop size W H")
    p.add_argument("--apply-clahe-rgb", action="store_true", help="Forward --apply-clahe-rgb to extractor")
    p.add_argument("--jpeg-quality", type=int, default=None, help="Optional JPEG quality forwarded to extractor")
    p.add_argument("--log-dir", default="logs/extract", help="Directory to write per-video extractor logs")
    args = p.parse_args()

    dets_dir = Path(args.detections_dir)
    videos_dir = Path(args.videos_dir)
    out_root = Path(args.out_root)
    canonical_w, canonical_h = args.canonical
    apply_clahe_rgb = args.apply_clahe_rgb
    jpeg_quality = args.jpeg_quality
    log_dir = Path(args.log_dir)

    if not dets_dir.exists():
        print(f"Detections directory not found: {dets_dir}")
        sys.exit(2)
    if not videos_dir.exists():
        print(f"Videos directory not found: {videos_dir}")
        sys.exit(2)

    json_files = sorted(dets_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {dets_dir}")
        sys.exit(0)

    for djson in json_files:
        stem = djson.stem
        video_path = videos_dir / (stem + ".avi")
        if not video_path.exists():
            print(f"SKIP missing video for {djson.name}: expected {video_path}")
            continue
        print(f"Processing {stem} -> video {video_path.name}")
        rc, logp = run_extractor(djson, video_path, out_root, canonical_w, canonical_h, apply_clahe_rgb, jpeg_quality, log_dir)
        if rc != 0:
            print(f"Extractor failed for {stem} (rc={rc}). See log: {logp}")
        else:
            print(f"Done {stem}. Log: {logp}")

if __name__ == "__main__":
    main()

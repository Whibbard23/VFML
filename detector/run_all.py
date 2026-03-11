#!/usr/bin/env python3
"""
run_all.py
Orchestrate: convert -> infer -> smooth -> extract for videos listed in event_csvs/cleaned_events.csv.
Saves CLAHE-normalized crops to detections/<video_stem>/mouth and detections/<video_stem>/ues.
Optionally runs tools/generate_crop_metadata.py at the end.
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd, log_path=None):
    print("RUN:", " ".join(cmd))
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "wb") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
            proc.wait()
            return proc.returncode
    else:
        proc = subprocess.Popen(cmd)
        proc.wait()
        return proc.returncode

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--events-csv", default="event_csvs/cleaned_events.csv", help="CSV listing videos (video column contains .avi filenames)")
    p.add_argument("--videos-dir", default="videos", help="Directory containing .avi files")
    p.add_argument("--dets-dir", default="detector_outputs", help="Directory to read/write detector JSONs")
    p.add_argument("--out-root", default="detections", help="Root directory to write crops (detections/<stem>/mouth, ues)")
    p.add_argument("--canonical", nargs=2, type=int, default=[224,224], help="Output crop size W H")
    p.add_argument("--apply-clahe-rgb", action="store_true", help="Apply CLAHE on RGB luminance during extraction")
    p.add_argument("--smooth-method", choices=["ema","median","kalman"], default="ema", help="Smoothing method")
    p.add_argument("--alpha", type=float, default=0.2, help="EMA alpha (if method=ema)")
    p.add_argument("--log-dir", default="logs/run_all", help="Directory for per-video logs")
    p.add_argument("--generate-metadata", action="store_true", help="Run tools/generate_crop_metadata.py after extraction to write metadata.json and combined CSV")
    p.add_argument("--manifest-csv", default="event_csvs/crops_manifest.csv", help="Path for combined CSV manifest written by metadata generator")
    args = p.parse_args()

    events_csv = Path(args.events_csv)
    videos_dir = Path(args.videos_dir)
    dets_dir = Path(args.dets_dir)
    out_root = Path(args.out_root)
    log_dir = Path(args.log_dir)
    canonical_w, canonical_h = args.canonical

    if not events_csv.exists():
        print("Events CSV not found:", events_csv)
        sys.exit(2)
    if not videos_dir.exists():
        print("Videos directory not found:", videos_dir)
        sys.exit(2)
    dets_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    with open(events_csv, newline='', encoding='utf-8') as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            video_file = row.get("video")
            if not video_file:
                continue
            video_file = video_file.strip()
            if video_file in seen:
                continue
            seen.add(video_file)
            stem = Path(video_file).stem
            video_path = videos_dir / video_file
            det_json = dets_dir / f"{stem}.json"
            smoothed_json = dets_dir / f"smoothed_{stem}.json"
            perlog = log_dir / f"{stem}.log"

            if not video_path.exists():
                print(f"SKIP missing video: {video_path}")
                continue

            # 1) Optional convert step (if present)
            convert_script = Path("detector/convert_to_yolo.py")
            if convert_script.exists():
                print(f"[{stem}] running convert_to_yolo.py")
                rc = run_cmd([sys.executable, str(convert_script), "--video", str(video_path)], perlog)
                if rc != 0:
                    print(f"[{stem}] convert failed (rc={rc}); see {perlog}")
                    continue

            # 2) Inference -> det_json
            infer_script = Path("detector/infer_yolo.py")
            if not infer_script.exists():
                print("Missing inference script detector/infer_yolo.py; cannot run inference.")
                sys.exit(2)
            print(f"[{stem}] running inference -> {det_json}")
            rc = run_cmd([sys.executable, str(infer_script), "--video", str(video_path), "--out-json", str(det_json)], perlog)
            if rc != 0:
                print(f"[{stem}] inference failed (rc={rc}); see {perlog}")
                continue

            # 3) Smoothing -> smoothed_json (smoother will read video to get frame size)
            smooth_script = Path("detector/smooth_boxes.py")
            if not smooth_script.exists():
                print(f"[{stem}] smoothing script not found; skipping smoothing (extractor will use raw boxes).")
                smoothed_json = det_json
            else:
                print(f"[{stem}] smoothing detections -> {smoothed_json}")
                rc = run_cmd([sys.executable, str(smooth_script),
                              "--input", str(det_json),
                              "--video", str(video_path),
                              "--out-dir", str(dets_dir),
                              "--method", args.smooth_method,
                              "--alpha", str(args.alpha)], perlog)
                if rc != 0:
                    print(f"[{stem}] smoothing failed (rc={rc}); see {perlog}")
                    continue

            # 4) Extraction -> detections/<stem>/{mouth,ues}
            extract_script = Path("detector/extract_crops_from_detections.py")
            if not extract_script.exists():
                print("Missing extractor detector/extract_crops_from_detections.py; cannot extract crops.")
                sys.exit(2)
            print(f"[{stem}] extracting crops to {out_root / stem}")
            cmd = [sys.executable, str(extract_script),
                   "--detections", str(smoothed_json),
                   "--video", str(video_path),
                   "--out", str(out_root),
                   "--canonical", str(canonical_w), str(canonical_h)]
            if args.apply_clahe_rgb:
                cmd.append("--apply-clahe-rgb")
            rc = run_cmd(cmd, perlog)
            if rc != 0:
                print(f"[{stem}] extraction failed (rc={rc}); see {perlog}")
                continue

            print(f"[{stem}] pipeline complete. Crops: {out_root / stem}")

    # Optional: generate metadata and combined CSV once after all videos processed
    if args.generate_metadata:
        meta_script = Path("tools/generate_crop_metadata.py")
        if not meta_script.exists():
            print("Metadata generator not found at tools/generate_crop_metadata.py; skipping metadata generation.")
            sys.exit(0)
        print("Generating per-video metadata.json files and combined CSV manifest...")
        rc = run_cmd([sys.executable, str(meta_script),
                      "--crops-root", str(out_root),
                      "--dets-dir", str(dets_dir),
                      "--manifest-csv", str(args.manifest_csv)])
        if rc != 0:
            print(f"Metadata generation failed (rc={rc}); check logs.")
        else:
            print(f"Metadata generation complete. Combined manifest: {args.manifest_csv}")

if __name__ == "__main__":
    main()

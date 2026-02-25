"""
Analyze ROI stability from YOLO label files.

Usage examples (run from project root with venv active):
  & .\.venv\Scripts\python.exe scripts\analyze_roi_stability.py runs/inference/AD128_roi
  & .\.venv\Scripts\python.exe scripts\analyze_roi_stability.py runs/inference/AD128_roi/labels/smoothed
  & .\.venv\Scripts\python.exe scripts\analyze_roi_stability.py runs/inference/AD128_roi --class-names 0:mouth 1:ues --max-frames 200

Features:
- Resolves project root relative to this script.
- Prints progress while reading files.
- Skips files that cannot be read and continues.
- Optional --max-frames to limit processing for quick checks.
- Writes plots and summary to <labels_folder>/../analysis.
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import sys
import time

def parse_args():
    p = argparse.ArgumentParser(description="Analyze ROI stability from YOLO label files")
    p.add_argument("path", help="Path to inference folder or labels folder (e.g., runs/inference/AD128_roi or .../labels/smoothed)")
    p.add_argument("--class-names", nargs="*", help="Optional class names as id:name pairs, e.g. 0:mouth 1:ues", default=[])
    p.add_argument("--no-plots", action="store_true", help="Do not save plots", default=False)
    p.add_argument("--max-frames", type=int, default=None, help="Limit number of frames to analyze (for quick tests)")
    p.add_argument("--verbose", action="store_true", help="Print progress and debug info", default=False)
    return p.parse_args()

def safe_read_lines(path: Path, verbose=False):
    """Read file lines robustly; return list of stripped non-empty lines or None on failure."""
    try:
        # open with explicit encoding and ignore errors to avoid blocking on weird bytes
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            return [ln.strip() for ln in fh if ln.strip()]
    except Exception as e:
        if verbose:
            print(f"Warning: failed to read {path}: {e}", file=sys.stderr)
        return None

def load_labels(labels_dir: Path, max_frames=None, verbose=False):
    files = sorted(labels_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"No label files found in {labels_dir}")
    if max_frames is not None:
        files = files[:max_frames]
    frames = []
    total = len(files)
    start = time.time()
    for i, f in enumerate(files):
        if verbose and (i % 200 == 0 or i < 10):
            print(f"Reading label {i+1}/{total}: {f.name}")
        lines = safe_read_lines(f, verbose=verbose)
        if lines is None:
            # treat as empty frame (no detections) and continue
            frames.append([])
            continue
        dets = []
        for ln in lines:
            parts = ln.split()
            if len(parts) >= 5:
                try:
                    cls = int(float(parts[0])); xc = float(parts[1]); yc = float(parts[2]); w = float(parts[3]); h = float(parts[4])
                    conf = float(parts[5]) if len(parts) > 5 else np.nan
                    dets.append({"cls": cls, "xc": xc, "yc": yc, "w": w, "h": h, "conf": conf})
                except Exception:
                    # skip malformed line
                    continue
        frames.append(dets)
    if verbose:
        elapsed = time.time() - start
        print(f"Finished reading {len(files)} label files in {elapsed:.1f}s")
    return frames, files

def iou_from_rel(xc1,yc1,w1,h1, xc2,yc2,w2,h2):
    if any(np.isnan([xc1,yc1,w1,h1, xc2,yc2,w2,h2])):
        return np.nan
    x1a = xc1 - w1/2; y1a = yc1 - h1/2; x2a = xc1 + w1/2; y2a = yc1 + h1/2
    x1b = xc2 - w2/2; y1b = yc2 - h2/2; x2b = xc2 + w2/2; y2b = yc2 + h2/2
    xi1 = max(x1a, x1b); yi1 = max(y1a, y1b)
    xi2 = min(x2a, x2b); yi2 = min(y2a, y2b)
    iw = max(0.0, xi2 - xi1); ih = max(0.0, yi2 - yi1)
    inter = iw * ih
    area1 = max(0.0, x2a - x1a) * max(0.0, y2a - y1a)
    area2 = max(0.0, x2b - x1b) * max(0.0, y2b - y1b)
    union = area1 + area2 - inter
    return inter/union if union>0 else 0.0

def build_time_series(frames, class_ids):
    T = len(frames)
    per_class = defaultdict(lambda: {"xc": np.full(T, np.nan), "yc": np.full(T, np.nan), "w": np.full(T, np.nan), "h": np.full(T, np.nan), "conf": np.full(T, np.nan)})
    for i, dets in enumerate(frames):
        by_cls = {}
        for d in dets:
            c = d["cls"]
            if c not in by_cls or (not math.isnan(d["conf"]) and d["conf"] > by_cls[c]["conf"]):
                by_cls[c] = d
        for c, d in by_cls.items():
            per_class[c]["xc"][i] = d["xc"]
            per_class[c]["yc"][i] = d["yc"]
            per_class[c]["w"][i] = d["w"]
            per_class[c]["h"][i] = d["h"]
            per_class[c]["conf"][i] = d["conf"]
    for c in class_ids:
        _ = per_class[c]
    return per_class

def analyze_and_plot(per_class, files, out_dir, class_names, save_plots=True):
    T = len(files)
    summary = {}
    for c, data in per_class.items():
        xc = data["xc"]; yc = data["yc"]; w = data["w"]; h = data["h"]
        dx = np.abs(np.diff(xc)); dy = np.abs(np.diff(yc))
        center_dist = np.sqrt(dx**2 + dy**2)
        mean_dx = float(np.nanmean(dx)) if np.any(~np.isnan(dx)) else float('nan')
        mean_dy = float(np.nanmean(dy)) if np.any(~np.isnan(dy)) else float('nan')
        mean_dist = float(np.nanmean(center_dist)) if np.any(~np.isnan(center_dist)) else float('nan')
        max_dist = float(np.nanmax(center_dist)) if np.any(~np.isnan(center_dist)) else float('nan')
        ious = np.array([iou_from_rel(xc[t],yc[t],w[t],h[t], xc[t+1],yc[t+1],w[t+1],h[t+1]) for t in range(T-1)])
        mean_iou = float(np.nanmean(ious)) if np.any(~np.isnan(ious)) else float('nan')
        median_iou = float(np.nanmedian(ious)) if np.any(~np.isnan(ious)) else float('nan')
        min_iou = float(np.nanmin(ious)) if np.any(~np.isnan(ious)) else float('nan')
        missing = int(np.sum(np.isnan(xc)))
        summary[c] = {"mean_dx": mean_dx, "mean_dy": mean_dy, "mean_dist": mean_dist, "max_dist": max_dist, "mean_iou": mean_iou, "median_iou": median_iou, "min_iou": min_iou, "missing_frames": missing, "total_frames": T}
        if save_plots:
            times = np.arange(T)
            plt.figure(figsize=(10,4))
            plt.plot(times, xc, label='xc')
            plt.plot(times, yc, label='yc')
            plt.title(f"Class {c} ({class_names.get(c,c)}) centers over time")
            plt.xlabel("frame")
            plt.ylabel("relative coord")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(out_dir / f"centers_class{c}.png")
            plt.close()

            plt.figure(figsize=(8,4))
            plt.hist(center_dist[~np.isnan(center_dist)], bins=50)
            plt.title(f"Frame-to-frame center distance histogram class {c} ({class_names.get(c,c)})")
            plt.xlabel("relative center distance")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(out_dir / f"jitter_hist_class{c}.png")
            plt.close()

            plt.figure(figsize=(10,3))
            plt.plot(np.arange(T-1), ious, marker='.', linestyle='-')
            plt.title(f"IoU between consecutive frames class {c} ({class_names.get(c,c)})")
            plt.xlabel("frame")
            plt.ylabel("IoU")
            plt.ylim(-0.05,1.05)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(out_dir / f"iou_timeseries_class{c}.png")
            plt.close()
    return summary

def main():
    args = parse_args()

    # Resolve script/project root relative to this file
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    if args.verbose:
        print("Script dir:", SCRIPT_DIR)
        print("Project root:", PROJECT_ROOT)

    p = Path(args.path)
    # If relative path provided, resolve relative to project root
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()

    if not p.exists():
        raise SystemExit(f"Path not found: {p}")

    # Determine labels_dir
    if p.is_dir():
        if (p / "labels").exists():
            sm = p / "labels" / "smoothed"
            raw = p / "labels" / "raw"
            if sm.exists():
                labels_dir = sm
            elif raw.exists():
                labels_dir = raw
            else:
                raise SystemExit(f"No labels found under {p / 'labels'}")
        else:
            # user may have passed labels folder directly
            if p.name == "labels":
                sm = p / "smoothed"
                raw = p / "raw"
                if sm.exists():
                    labels_dir = sm
                elif raw.exists():
                    labels_dir = raw
                else:
                    labels_dir = p
            else:
                labels_dir = p
    else:
        raise SystemExit(f"Path not a directory: {p}")

    if args.verbose:
        print("Using labels folder:", labels_dir)

    frames, files = load_labels(labels_dir, max_frames=args.max_frames, verbose=args.verbose)

    # parse class names
    class_names = {}
    class_ids = set()
    for pair in args.class_names:
        if ":" in pair:
            k,v = pair.split(":",1)
            class_names[int(k)] = v
            class_ids.add(int(k))
    if not class_ids:
        for dets in frames:
            for d in dets:
                class_ids.add(d["cls"])
    class_ids = sorted(list(class_ids))
    if args.verbose:
        print("Detected class ids:", class_ids)

    per_class = build_time_series(frames, class_ids)
    out_dir = labels_dir.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = analyze_and_plot(per_class, files, out_dir, class_names, save_plots=not args.no_plots)

    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as fh:
        fh.write("ROI stability analysis summary\n")
        fh.write(f"Labels folder: {labels_dir}\n")
        fh.write(f"Frames analyzed: {len(files)}\n\n")
        for c, s in summary.items():
            name = class_names.get(c, str(c))
            fh.write(f"Class {c} ({name}): mean_dx={s['mean_dx']:.6f}, mean_dy={s['mean_dy']:.6f}, mean_dist={s['mean_dist']:.6f}, max_dist={s['max_dist']:.6f}, mean_iou={s['mean_iou']:.3f}, median_iou={s['median_iou']:.3f}, min_iou={s['min_iou']:.3f}, missing_frames={s['missing_frames']}/{s['total_frames']}\n")

    print("Analysis complete. Summary:")
    for c, s in summary.items():
        name = class_names.get(c, str(c))
        print(f"Class {c} ({name}): mean_iou={s['mean_iou']:.3f}, median_iou={s['median_iou']:.3f}, mean_center_dist={s['mean_dist']:.6f}, missing={s['missing_frames']}/{s['total_frames']}")
    print("Plots and full summary saved to:", out_dir)
    print("Interpretation thresholds: mean IoU >0.8 excellent; 0.6-0.8 usable; <0.6 unstable")
    return 0

if __name__ == "__main__":
    sys.exit(main())

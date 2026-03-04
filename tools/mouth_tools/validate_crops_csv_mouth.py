#!/usr/bin/env python3
"""
Validate mouth crops CSV produced by crop_and_label_mouth.py.

Checks:
 - per-video counts of labels (0/1)
 - missing pairs: for each onset (label=1) ensure before_onset (frame-1) exists; for each before_onset (label=0)
   that is part of an event ensure onset (frame+1) exists
 - ROI availability: whether smoothed mouth .txt exists and fraction of selected frames that have a valid ROI entry
 - basic sanity: duplicate rows, out-of-range frames

Usage:
  python tools/validate_mouth_crops_csv.py --csv event_csvs/mouth_crops_labels_with_split.csv --smoothed-root runs/inference --video-root "W:\ADStudy\VF AD Blinded\Early Tongue Training"
"""
from pathlib import Path
import argparse
import csv
from collections import defaultdict, Counter

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Generated mouth crops CSV")
    p.add_argument("--smoothed-root", default="runs/inference", help="root where <video_stem>_roi/labels/smoothed/<video>_mouth.txt live")
    p.add_argument("--video-root", default=".", help="root to resolve relative video paths")
    p.add_argument("--max_rows", type=int, default=None, help="optional: limit rows processed for quick checks")
    return p.parse_args()

def read_generated_csv(p: Path, max_rows=None):
    rows = []
    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        rdr = csv.DictReader(fh)
        for i, r in enumerate(rdr):
            if max_rows is not None and i >= max_rows:
                break
            # normalize keys we expect
            rows.append({
                "video": (r.get("video") or "").strip(),
                "frame": int(r.get("frame") or -1),
                "label": int(float(r.get("label") or 0)),
                "xc": r.get("xc"),
                "yc": r.get("yc"),
                "w": r.get("w"),
                "h": r.get("h"),
                "filepath": (r.get("filepath") or "").strip(),
                "split": (r.get("split") or "").strip().lower()
            })
    return rows

def read_smoothed_mouth_lines(smoothed_path: Path):
    if not smoothed_path.exists():
        return None
    lines = []
    with smoothed_path.open("r", encoding="utf-8-sig") as fh:
        for L in fh:
            s = L.strip()
            if not s:
                lines.append(None)
            else:
                parts = s.split()
                if len(parts) >= 4:
                    try:
                        xc = float(parts[0]); yc = float(parts[1]); w = float(parts[2]); h = float(parts[3])
                        lines.append((xc, yc, w, h))
                    except Exception:
                        lines.append(None)
                else:
                    lines.append(None)
    return lines

def validate(rows, smoothed_root: Path, video_root: Path):
    by_video = defaultdict(list)
    for r in rows:
        by_video[r["video"]].append(r)

    summary = []
    total_missing_before = 0
    total_missing_onset = 0
    total_rows = 0
    dup_count = 0

    for v, items in sorted(by_video.items()):
        frames_set = set()
        duplicates = 0
        for it in items:
            key = (it["video"], it["frame"])
            if it["frame"] in frames_set:
                duplicates += 1
            frames_set.add(it["frame"])
        dup_count += duplicates

        counts = Counter([it["label"] for it in items])
        total_rows += len(items)

        # missing pairs
        onset_frames = set([it["frame"] for it in items if it["label"] == 1])
        before_frames = set([it["frame"] for it in items if it["label"] == 0])
        missing_before = sum(1 for f in onset_frames if (f - 1) not in before_frames)
        missing_onset = sum(1 for f in before_frames if (f + 1) not in onset_frames)

        total_missing_before += missing_before
        total_missing_onset += missing_onset

        # ROI availability
        smoothed_path = smoothed_root / f"{Path(v).stem}_roi" / "labels" / "smoothed" / f"{Path(v).stem}_mouth.txt"
        smoothed = read_smoothed_mouth_lines(smoothed_path)
        if smoothed is None:
            roi_exists = False
            valid_roi_count = 0
            roi_fraction = 0.0
            smoothed_len = 0
        else:
            roi_exists = True
            smoothed_len = len(smoothed)
            valid_roi_count = 0
            for it in items:
                fi = it["frame"]
                if 0 <= fi < smoothed_len and smoothed[fi] is not None:
                    valid_roi_count += 1
            roi_fraction = valid_roi_count / max(1, len(items))

        # filepath checks (existence)
        filepaths = set(it["filepath"] for it in items if it.get("filepath"))
        filepath_exists = all(Path(fp).exists() for fp in filepaths) if filepaths else False

        summary.append({
            "video": v,
            "split": items[0].get("split",""),
            "rows": len(items),
            "pos": counts.get(1,0),
            "neg": counts.get(0,0),
            "missing_before": missing_before,
            "missing_onset": missing_onset,
            "roi_exists": roi_exists,
            "roi_valid_count": valid_roi_count,
            "roi_fraction": round(roi_fraction,3),
            "filepath_exists": filepath_exists,
            "duplicates": duplicates
        })

    # print compact table
    header = ["video","split","rows","pos","neg","miss_before","miss_onset","roi_ok_frac","roi_valid","dup","fp_exists"]
    print(",".join(header))
    for s in summary:
        print(f"{s['video']},{s['split']},{s['rows']},{s['pos']},{s['neg']},{s['missing_before']},{s['missing_onset']},{s['roi_fraction']},{s['roi_valid_count']},{s['duplicates']},{int(s['filepath_exists'])}")

    # overall summary
    print("\nOverall summary:")
    print(f"  total_videos: {len(summary)}")
    print(f"  total_rows: {total_rows}")
    print(f"  total_duplicates: {dup_count}")
    print(f"  total_missing_before (onset without before): {total_missing_before}")
    print(f"  total_missing_onset (before without onset): {total_missing_onset}")

def main():
    args = parse_args()
    csv_p = Path(args.csv)
    if not csv_p.exists():
        raise FileNotFoundError(f"CSV not found: {csv_p}")
    rows = read_generated_csv(csv_p, max_rows=args.max_rows)
    if not rows:
        print("No rows found in CSV.")
        return
    validate(rows, Path(args.smoothed_root), Path(args.video_root))

if __name__ == "__main__":
    main()

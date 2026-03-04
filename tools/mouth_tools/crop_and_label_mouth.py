#!/usr/bin/env python3
"""
Create mouth crop labels CSV with train/val split column.

Behavior changes:
- No stem-based lookups or normalization. The script uses the **video string exactly as provided**
  in event/split CSVs and checks for that filename under --video-root (or as an absolute path).
- Accepts multiple event CSVs (pass one or more paths to --events-csv).
- Optional --skip-existing to avoid reprocessing videos already present in the output CSV.
- Skips missing videos only when --skip-missing-videos is set; otherwise raises.

Example:
  python tools/crop_and_label_mouth.py \
    --events-csv event_csvs/assembly_1_train_events.csv event_csvs/assembly_1_val_events.csv \
    --split-csv event_csvs/assembly_1_video_splits.csv \
    --smoothed-root runs/inference \
    --video-root "W:\ADStudy\VF AD Blinded\Early Tongue Training" \
    --out-csv event_csvs/mouth_crops_labels_with_split.csv \
    --skip-existing
"""
from pathlib import Path
import argparse
import csv
import random
import cv2
from collections import defaultdict

RNG_SEED = 42

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--events-csv", required=True, nargs="+", help="One or more CSVs with event annotations (contains 'video','frame','event_type')")
    p.add_argument("--smoothed-root", default="runs/inference", help="root where <video>_roi/labels/smoothed/<video>_mouth.txt live")
    p.add_argument("--video-root", default=".", help="folder to resolve video filenames (expects exact filenames in CSVs)")
    p.add_argument("--split-csv", default="event_csvs/assembly_1_video_splits.csv", help="CSV mapping video -> split (train/val)")
    p.add_argument("--out-csv", default="event_csvs/mouth_crops_labels_with_split.csv", help="output CSV path")
    p.add_argument("--skip-missing-videos", action="store_true", help="skip videos whose file cannot be found instead of erroring")
    p.add_argument("--skip-existing", action="store_true", help="skip videos already present in the output CSV (append new rows for others)")
    p.add_argument("--negatives-per-onset", type=int, default=2, help="random negatives per onset for videos with onsets")
    p.add_argument("--touch-per-onset", type=int, default=1, help="touch_ues negatives per onset")
    p.add_argument("--negatives-per-video", type=int, default=10, help="random negatives for videos with zero onsets")
    p.add_argument("--max-random-per-video", type=int, default=None, help="cap on random negatives per video")
    p.add_argument("--include-videos-without-events", action="store_true", help="include videos listed in split CSV even if they have no events")
    return p.parse_args()

# -----------------------------
# Read event annotations (multiple CSVs)
# -----------------------------
def read_events(events_csv_paths):
    """
    events_csv_paths: iterable of Path or str.
    Returns: before, onset, touch, all_events dicts mapping video -> set(frames)
    Deduplicates identical (video,frame,event_type) rows.
    Uses the video string exactly as provided in CSVs (no stem normalization).
    """
    before = defaultdict(set)
    onset = defaultdict(set)
    touch = defaultdict(set)
    all_events = defaultdict(set)
    seen = set()  # (video, frame, event_type)

    for p in events_csv_paths:
        p = Path(p)
        if not p.exists():
            print(f"Warning: events CSV not found: {p} (skipping)")
            continue
        with p.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames:
                reader.fieldnames = [fn.strip().lstrip("\ufeff") for fn in reader.fieldnames]
            for r in reader:
                v = (r.get("video") or r.get("Video") or r.get("video_name") or "").strip().strip('"').strip("'")
                if not v:
                    continue
                try:
                    f = int(float(r.get("frame") or -1))
                except Exception:
                    continue
                et = (r.get("event_type") or r.get("event") or "").strip()
                key = (v, f, et)
                if key in seen:
                    continue
                seen.add(key)
                if et == "before_onset":
                    before[v].add(f)
                    onset[v].add(f + 1)
                    all_events[v].add(f)
                    all_events[v].add(f + 1)
                elif et == "touch_ues":
                    touch[v].add(f)
                    all_events[v].add(f)
                else:
                    all_events[v].add(f)
    return before, onset, touch, all_events

# -----------------------------
# Read split CSV (video -> split)
# -----------------------------
def read_split_csv(split_csv_path: Path):
    """
    Read mapping of exact video string -> split ('train' or 'val').
    No stem/name normalization; keys are stored exactly as read.
    """
    mapping = {}
    if not split_csv_path or not split_csv_path.exists():
        return mapping
    with split_csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames:
            reader.fieldnames = [fn.strip().lstrip("\ufeff") for fn in reader.fieldnames]
        for r in reader:
            v = (r.get("video") or r.get("video_name") or r.get("Video") or "").strip().strip('"').strip("'")
            if not v:
                keys = reader.fieldnames or []
                if keys:
                    v = (r.get(keys[0]) or "").strip()
            if not v:
                continue
            split = (r.get("split") or r.get("set") or r.get("Split") or "").strip().lower()
            if split == "validation":
                split = "val"
            if split not in ("train", "val"):
                is_train = (r.get("is_train") or "").strip().lower()
                if is_train in ("1", "true", "yes", "y"):
                    split = "train"
                else:
                    split = "train"
            mapping[v] = split
    return mapping

# -----------------------------
# Read smoothed mouth file
# -----------------------------
def read_smoothed_mouth(smoothed_path: Path):
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

# -----------------------------
# Video info (exact filename lookup only)
# -----------------------------
def resolve_video_path_exact(video_key: str, video_root: Path):
    """
    Resolve the video path using the exact video_key string.
    If video_key is absolute, check it directly. Otherwise check video_root / video_key.
    Do NOT try stems or add extensions.
    Returns Path if found, else None.
    """
    p = Path(video_key)
    if p.is_absolute():
        return p.resolve() if p.exists() else None
    cand = (video_root / video_key)
    return cand.resolve() if cand.exists() else None

def video_info(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, None, None
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fc, w, h

# -----------------------------
# Sampling logic (pair-preserving)
# -----------------------------
def sample_frames(
    frame_count,
    smoothed,
    before_set,
    onset_set,
    touch_set,
    all_event_set,
    negatives_per_onset,
    touch_per_onset,
    negatives_per_video,
    max_random=None
):
    selected = {}

    # 1. Include ALL onset frames (positives)
    for f in onset_set:
        if 0 <= f < frame_count:
            selected[f] = 1

    # 2. Include ALL before_onset frames (hard negatives)
    for f in before_set:
        if 0 <= f < frame_count:
            selected[f] = 0

    n_pos = len([f for f in onset_set if 0 <= f < frame_count])

    # 3. Touch UES negatives (optional)
    touch_candidates = [f for f in touch_set if 0 <= f < frame_count and f not in selected]
    random.shuffle(touch_candidates)
    n_touch = touch_per_onset * n_pos
    for f in touch_candidates[:n_touch]:
        selected[f] = 0

    # 4. Random negatives
    if n_pos > 0:
        n_random = negatives_per_onset * n_pos
    else:
        n_random = negatives_per_video

    if max_random is not None:
        n_random = min(n_random, max_random)

    pool = [
        f for f in range(frame_count)
        if f not in selected
        and f < len(smoothed)
        and smoothed[f] is not None
        and f not in all_event_set
    ]
    random.shuffle(pool)
    for f in pool[:n_random]:
        selected[f] = 0

    return selected, n_pos

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    random.seed(RNG_SEED)

    events_csv_paths = [Path(p) for p in args.events_csv]
    smoothed_root = Path(args.smoothed_root)
    video_root = Path(args.video_root)
    out_csv = Path(args.out_csv)
    split_csv = Path(args.split_csv) if args.split_csv else None

    before_map, onset_map, touch_map, all_event_map = read_events(events_csv_paths)
    split_map = read_split_csv(split_csv) if split_csv else {}

    # Build list of videos to process: union of split CSV videos and videos with events
    videos_from_splits = set(split_map.keys())
    videos_from_events = set(onset_map.keys()) | set(before_map.keys()) | set(touch_map.keys()) | set(all_event_map.keys())
    videos = sorted(videos_from_splits | videos_from_events)

    if not videos:
        print("No videos found to process. Exiting.")
        return

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # If skip-existing, read existing out_csv and collect videos already written (exact filenames)
    existing_videos = set()
    existing_rows = []
    append_mode = False
    if out_csv.exists():
        if args.skip_existing:
            with out_csv.open("r", encoding="utf-8-sig", newline="") as fh:
                rdr = csv.DictReader(fh)
                for r in rdr:
                    v = (r.get("video") or "").strip()
                    if v:
                        existing_videos.add(v)  # store exact filename as present in CSV
                        existing_rows.append(r)
            append_mode = True
            print(f"Output CSV {out_csv} exists: will skip {len(existing_videos)} videos already present and append new rows.")
        else:
            append_mode = False
            print(f"Output CSV {out_csv} exists: it will be overwritten.")

    total_rows = 0
    per_split_counts = defaultdict(int)
    written_rows = []

    # open output file (append or write)
    if append_mode:
        outfh = out_csv.open("a", newline="", encoding="utf-8")
        writer = csv.writer(outfh)
    else:
        outfh = out_csv.open("w", newline="", encoding="utf-8")
        writer = csv.writer(outfh)
        writer.writerow(["video","frame","label","xc","yc","w","h","width","height","filepath","split"])

    try:
        for v in videos:
            # v is used exactly as provided in CSVs; do not convert to stem or alter
            if args.skip_existing and v in existing_videos:
                print(f"Skipping video {v} (already present in {out_csv})")
                continue

            # split lookup uses exact key only
            split = split_map.get(v, "train")

            # resolve video path using exact filename only
            video_path = resolve_video_path_exact(v, video_root)
            if video_path is None:
                msg = f"Missing video file for key '{v}' under {video_root}"
                if args.skip_missing_videos:
                    print(msg + "  (skipping)")
                    continue
                else:
                    raise FileNotFoundError(msg)

            stem = Path(v).stem  # used only for smoothed path naming convention
            smoothed_path = smoothed_root / f"{stem}_roi" / "labels" / "smoothed" / f"{stem}_mouth.txt"
            smoothed = read_smoothed_mouth(smoothed_path)

            frame_count, vid_w, vid_h = video_info(video_path)
            if frame_count is None:
                print(f"Unable to open video: {video_path} (skipping)")
                continue

            if smoothed is None:
                print(f"Missing smoothed mouth file for {v}: {smoothed_path} (skipping)")
                continue
            if len(smoothed) < frame_count:
                smoothed += [None] * (frame_count - len(smoothed))

            before_set = before_map.get(v, set())
            onset_set = onset_map.get(v, set())
            touch_set = touch_map.get(v, set())
            all_event_set = all_event_map.get(v, set())

            selected_map, n_pos = sample_frames(
                frame_count, smoothed, before_set, onset_set, touch_set, all_event_set,
                args.negatives_per_onset, args.touch_per_onset, args.negatives_per_video, max_random=args.max_random_per_video
            )

            # If video has no onsets and user requested inclusion, sample negatives_per_video
            if n_pos == 0 and args.include_videos_without_events:
                pool = [i for i in range(frame_count) if i < len(smoothed) and smoothed[i] is not None and i not in all_event_set]
                random.shuffle(pool)
                for f in pool[:args.negatives_per_video]:
                    selected_map[f] = 0

            # write rows for selected frames; use last_valid fallback for crop dims
            last_valid = None
            written = 0
            for fi in sorted(selected_map.keys()):
                entry = smoothed[fi]
                if entry is None:
                    if last_valid is None:
                        # cannot produce crop dims for this frame; skip
                        continue
                    xc, yc, w, h = last_valid
                else:
                    xc, yc, w, h = entry
                    last_valid = (xc, yc, w, h)

                label = selected_map[fi]
                # write the video key exactly as provided in CSVs (expected to include extension)
                writer.writerow([v, fi, label, f"{xc:.6f}", f"{yc:.6f}", f"{w:.6f}", f"{h:.6f}", vid_w, vid_h, str(video_path), split])
                written += 1
                total_rows += 1
                per_split_counts[split] += 1
                written_rows.append({"video": v, "frame": fi, "label": label})

            print(f"Video {v}: split={split} onsets={n_pos} written_rows={written}")

    finally:
        outfh.close()

    print(f"Saved mouth crop labels to: {out_csv} (total new rows: {total_rows})")
    print("Per-split counts (new rows):", dict(per_split_counts))

    # Quick validation: count labels and missing pairs
    counts = defaultdict(int)
    for r in written_rows:
        counts[r["label"]] += 1
    print("Label counts in newly written rows:", dict(counts))

    # Check missing pairs across all written rows (including existing if skip_existing)
    all_written = written_rows + existing_rows
    by_video = defaultdict(set)
    for r in all_written:
        by_video[r["video"]].add(int(r["frame"]))
    missing_before = 0
    missing_onset = 0
    for v in by_video:
        frames = by_video[v]
        for f in list(frames):
            if (f in frames) and ((f - 1) not in frames) and ((f in onset_map.get(v, set()))):
                missing_before += 1
            if (f in frames) and ((f + 1) not in frames) and ((f in before_map.get(v, set()))):
                missing_onset += 1
    print(f"Validation: Missing before_onset pairs: {missing_before}, Missing onset pairs: {missing_onset}")

if __name__ == "__main__":
    main()

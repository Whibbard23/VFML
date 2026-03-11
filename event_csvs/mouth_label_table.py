#!/usr/bin/env python3
"""
event_csvs/mouth_label_table.py

Generate mouth_frame_label_table.csv with columns:
    video,frame,label,split

- video: video id with .avi extension (e.g., AD128.avi)
- frame: integer frame index (parsed from filenames like frame_000123.npy)
- label: 1 for swallow onset frames (before_onset + 1), 0 otherwise
- split: 'train' or 'val' from event_csvs/train_val_split.csv

Usage:
    python event_csvs/mouth_label_table.py `
      --crops-root "E:/VF ML Crops" `
      --cleaned-events "event_csvs/cleaned_events.csv" `
      --split-csv "event_csvs/train_val_split.csv" `
      --out-csv "event_csvs/mouth_frame_label_table.csv" `
      --verbose
"""
from pathlib import Path
import argparse
import csv
import re
from collections import defaultdict

FRAME_RE = re.compile(r"frame[_\-]?0*([0-9]+)\.npy$", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(description="Create mouth_frame_label_table.csv")
    p.add_argument(
        "--crops-root",
        default="E:/VF ML Crops",
        help="Root folder containing per-video crop folders (default: E:/VF ML Crops)",
    )
    p.add_argument(
        "--mouth-subdir",
        default="crops_normalized/mouth",
        help="Relative path under each video folder where mouth .npy files live",
    )
    p.add_argument(
        "--cleaned-events",
        default="event_csvs/cleaned_events.csv",
        help="CSV with cleaned events; must contain columns 'video' and 'before_onset'",
    )
    p.add_argument(
        "--split-csv",
        default="event_csvs/train_val_split.csv",
        help="CSV with columns 'video' and 'split' (train/val)",
    )
    p.add_argument(
        "--out-csv",
        default="event_csvs/mouth_frame_label_table.csv",
        help="Output CSV path",
    )
    p.add_argument("--verbose", action="store_true", help="Print progress and counts")
    return p.parse_args()


def load_train_val_split(path: Path):
    mapping = {}
    if not path.exists():
        return mapping
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video")
            split = row.get("split", "").strip().lower()
            if not vid:
                continue
            # normalize video id to include .avi
            if not vid.lower().endswith(".avi"):
                vid = f"{vid}.avi"
            if split not in ("train", "val"):
                split = "train"
            mapping[vid] = split
    return mapping


def load_onset_frames(cleaned_events_path: Path):
    """
    Returns dict: video -> set(of onset frame ints)
    cleaned_events CSV expected to have columns: video, before_onset
    Onset frame = before_onset + 1
    """
    onset = defaultdict(set)
    if not cleaned_events_path.exists():
        return onset
    with cleaned_events_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video")
            if not vid:
                continue
            # normalize video id to include .avi
            if not vid.lower().endswith(".avi"):
                vid_key = f"{vid}.avi"
            else:
                vid_key = vid
            bo = row.get("before_onset")
            if bo is None or bo == "":
                continue
            try:
                before = int(float(bo))
            except Exception:
                continue
            onset_frame = before + 1
            onset[vid_key].add(onset_frame)
            # also store without .avi key for robustness
            alt = vid_key[:-4] if vid_key.lower().endswith(".avi") else vid_key
            onset[alt].add(onset_frame)
    return onset


def collect_frames_for_video(video_folder: Path, mouth_subdir: str):
    """
    Returns sorted list of frame ints found under video_folder / mouth_subdir
    """
    frames = []
    mouth_dir = video_folder / mouth_subdir
    if not mouth_dir.exists():
        return frames
    for p in mouth_dir.rglob("*.npy"):
        m = FRAME_RE.search(p.name)
        if not m:
            continue
        try:
            frm = int(m.group(1))
        except Exception:
            continue
        frames.append(frm)
    return sorted(set(frames))


def find_video_folders(crops_root: Path):
    """
    Find candidate video folders under crops_root. We expect folders named like 'AD128.avi'.
    Return list of Path objects (direct children that are directories).
    """
    if not crops_root.exists():
        return []
    return [p for p in crops_root.iterdir() if p.is_dir()]


def main():
    args = parse_args()
    crops_root = Path(args.crops_root)
    mouth_subdir = args.mouth_subdir
    cleaned_events = Path(args.cleaned_events)
    split_csv = Path(args.split_csv)
    out_csv = Path(args.out_csv)

    if args.verbose:
        print(f"Using crops root: {crops_root}")
        print(f"Mouth subdir: {mouth_subdir}")
        print(f"Cleaned events CSV: {cleaned_events}")
        print(f"Train/val split CSV: {split_csv}")
        print(f"Output CSV: {out_csv}")

    split_map = load_train_val_split(split_csv)
    onset_map = load_onset_frames(cleaned_events)

    video_folders = find_video_folders(crops_root)
    if args.verbose:
        print(f"Found {len(video_folders)} video folders under crops root")

    rows_written = 0
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "frame", "label", "split"])
        writer.writeheader()

        for vf in sorted(video_folders, key=lambda p: p.name):
            folder_name = vf.name  # expected like 'AD128.avi' or 'AD128.avi '
            video_id = folder_name.strip()
            if not video_id.lower().endswith(".avi"):
                video_id = f"{video_id}.avi"

            frames = collect_frames_for_video(vf, mouth_subdir)
            if args.verbose:
                print(f"{video_id}: found {len(frames)} frames in {vf / mouth_subdir}")

            if not frames:
                continue

            split = split_map.get(video_id)
            if split is None:
                # try without .avi
                alt = video_id[:-4] if video_id.lower().endswith(".avi") else video_id
                split = split_map.get(alt)
            if split is None:
                split = "train"
                if args.verbose:
                    print(f"Warning: split for {video_id} not found; defaulting to 'train'")

            onset_frames = onset_map.get(video_id, set())
            if not onset_frames:
                alt = video_id[:-4] if video_id.lower().endswith(".avi") else video_id
                onset_frames = onset_map.get(alt, set())

            for frm in frames:
                label = 1 if frm in onset_frames else 0
                writer.writerow({"video": video_id, "frame": int(frm), "label": label, "split": split})
                rows_written += 1

    if args.verbose:
        print(f"Wrote {rows_written} rows to {out_csv}")
    else:
        print(f"Wrote {rows_written} rows to {out_csv}")


if __name__ == "__main__":
    main()

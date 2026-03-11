#!/usr/bin/env python3
"""
create_train_val_split.py

Scans YOLO image label folders (train/val) for .txt files named like:
    AD128.avi_000079.txt
and produces a CSV with columns: video,split
where video is e.g. AD128.avi and split is either 'train' or 'val'.

Usage example (PowerShell):
    python event_csvs/create_train_val_split.py `
      --train-dir "C:/Users/Connor Lab/Desktop/VFML/detector/yolo_dataset/labels/train" `
      --val-dir "C:/Users/Connor Lab/Desktop/VFML/detector/yolo_dataset/labels/val" `
      --out-csv "event_csvs/train_val_split.csv"
"""
import argparse
import csv
from pathlib import Path
from typing import Set


def extract_video_id_from_txt(fname: str) -> str:
    """
    Given a filename like 'AD128.avi_000079.txt' or 'AD128_000079.txt',
    return the video id in the form 'AD128.avi' (append .avi if missing).
    """
    stem = Path(fname).stem  # 'AD128.avi_000079' or 'AD128_000079'
    left, _, _ = stem.partition("_")
    video = left
    if not video.lower().endswith(".avi"):
        video = f"{video}.avi"
    return video


def collect_videos_from_dir(dir_path: Path) -> Set[str]:
    videos = set()
    if not dir_path.exists():
        return videos
    for p in dir_path.rglob("*.txt"):
        vid = extract_video_id_from_txt(p.name)
        videos.add(vid)
    return videos


def main():
    p = argparse.ArgumentParser(description="Create train_val_split.csv from YOLO .txt files")
    p.add_argument("--train-dir", required=True, help="Path to yolo_dataset/images/train")
    p.add_argument("--val-dir", required=True, help="Path to yolo_dataset/images/val")
    p.add_argument("--out-csv", required=True, help="Output CSV path (train_val_split.csv)")
    args = p.parse_args()

    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    out_csv = Path(args.out_csv)

    train_videos = collect_videos_from_dir(train_dir)
    val_videos = collect_videos_from_dir(val_dir)

    # Resolve conflicts: if a video appears in both, prefer 'train'
    overlap = train_videos.intersection(val_videos)
    if overlap:
        val_videos = val_videos.difference(overlap)

    all_videos = sorted(train_videos.union(val_videos))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "split"])
        writer.writeheader()
        for v in all_videos:
            split = "train" if v in train_videos else "val"
            writer.writerow({"video": v, "split": split})

    print(f"Wrote {len(all_videos)} rows to {out_csv}")
    if overlap:
        print(f"Note: {len(overlap)} videos found in both train and val; labeled as 'train' by default.")


if __name__ == "__main__":
    main()

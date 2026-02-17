"""
extract_swallow_clips.py

Works with long-format cleaned_events.csv:
    video, event_type, frame

Groups events into swallows by ordering them within each video.
Extracts swallow-level clips using fallback rules.

Output structure:
    swallow_clips/<video_name>/swallow_0001/frame_0000.png
"""

import os
import cv2
import pandas as pd

CLEANED_CSV = r"C:\Users\Connor Lab\Desktop\VFML\event_csvs\cleaned_events.csv"
VIDEO_DIR = r"\\research.drive.wisc.edu\npconnor\ADStudy\VF AD Blinded\Early Tongue Training"
OUT_DIR = r"C:\Users\Connor Lab\Desktop\VFML\swallow_clips"

# Padding rules
PAD_BEFORE = 5
PAD_AFTER = 5
FALLBACK_AFTER_TOUCH = 10
FALLBACK_AFTER_BEFORE = 20
FALLBACK_BEFORE_TOUCH = 20

EVENT_TYPES = ["before_onset", "touch_ues", "leave_ues"]


def compute_swallow_window(events, total_frames):
    """
    events: dict mapping event_type -> frame index (or None)
    Applies fallback rules to compute (start, end).
    """

    b = events.get("before_onset")
    t = events.get("touch_ues")
    l = events.get("leave_ues")

    # Case F: no events at all
    if b is None and t is None and l is None:
        return None

    # Case A: all three present
    if b is not None and t is not None and l is not None:
        start = max(0, b - PAD_BEFORE)
        end = min(total_frames - 1, l + PAD_AFTER)
        return start, end

    # Case B: missing leave_ues
    if b is not None and t is not None and l is None:
        start = max(0, b - PAD_BEFORE)
        end = min(total_frames - 1, t + FALLBACK_AFTER_TOUCH)
        return start, end

    # Case C: only before_onset present
    if b is not None and t is None and l is None:
        start = max(0, b - PAD_BEFORE)
        end = min(total_frames - 1, b + FALLBACK_AFTER_BEFORE)
        return start, end

    # Case D: missing before_onset but have UES events
    if b is None and (t is not None or l is not None):
        anchor = t if t is not None else l
        start = max(0, anchor - FALLBACK_BEFORE_TOUCH)
        end = min(total_frames - 1, (l if l is not None else anchor) + PAD_AFTER)
        return start, end

    # Case E: only one event present
    events_present = [e for e in [b, t, l] if e is not None]
    if len(events_present) == 1:
        e = events_present[0]
        start = max(0, e - FALLBACK_BEFORE_TOUCH)
        end = min(total_frames - 1, e + FALLBACK_AFTER_BEFORE)
        return start, end

    return None


def extract_clip(video_path, start, end, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        return

    for i, f in enumerate(range(start, end + 1)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:04d}.png"), gray)

    cap.release()


def main():
    df = pd.read_csv(CLEANED_CSV)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Sort by video then frame
    df = df.sort_values(["video", "frame"]).reset_index(drop=True)

    os.makedirs(OUT_DIR, exist_ok=True)

    for video_name, group in df.groupby("video"):
        video_path = os.path.join(VIDEO_DIR, video_name)
        if not os.path.exists(video_path):
            print(f"Video not found: {video_name}")
            continue

        # Load video metadata
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Group events into swallows by ordering them
        # Each event row is one event; we treat each row as a separate swallow anchor
        # and gather nearby events of the same swallow.
        # For now: assume each row is a separate swallow event set.
        # (You can refine grouping later if needed.)
        video_out = os.path.join(OUT_DIR, os.path.splitext(video_name)[0])
        os.makedirs(video_out, exist_ok=True)

        # Build swallow index per video
        swallow_id = 0

        # Group by swallow index: each event row is one swallow
        for idx, row in group.iterrows():
            swallow_id += 1

            # Build event dict for this swallow
            events = {etype: None for etype in EVENT_TYPES}
            etype = row["event_type"]
            frame = int(row["frame"])

            if etype in events:
                events[etype] = frame

            # Compute swallow window
            window = compute_swallow_window(events, total_frames)
            if window is None:
                continue

            start, end = window
            out_dir = os.path.join(video_out, f"swallow_{swallow_id:04d}")

            extract_clip(video_path, start, end, out_dir)
            print(f"Extracted swallow_{swallow_id:04d} from {video_name}: {start}–{end}")


if __name__ == "__main__":
    main()

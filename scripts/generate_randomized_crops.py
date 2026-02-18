r"""
generate_randomized_crops.py

Reads:  event_csvs/cleaned_events.csv  (columns: video, before_onset, touch_ues, leave_ues)
Writes:
  - data/base_windows.csv
  - data/crops/<train|val>/crop_<id>/frame_0000.jpg ... (grayscale)
  - data/manifests/crops_manifest.csv  (metadata + labels)
  - data/manifests/train_videos.txt, val_videos.txt

This version derives the list of videos from the CSV (only files referenced there),
is network-safe (reads each video once for all windows), and includes robust NaN
handling and debug output for window computation.

Edit DRY_RUN_ROWS to inspect specific rows before extraction.
"""

import os
import cv2
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# -------------------------
# User configuration
# -------------------------
PROJECT_ROOT = Path(r"C:\Users\Connor Lab\Desktop\VFML")
CLEANED_CSV = PROJECT_ROOT / "event_csvs" / "cleaned_events.csv"
VIDEO_DIR = Path(r"\\research.drive.wisc.edu\npconnor\ADStudy\VF AD Blinded\Early Tongue Training")
OUT_DIR = PROJECT_ROOT / "data"
CROPS_DIR = OUT_DIR / "crops"
MANIFEST_DIR = OUT_DIR / "manifests"
BASE_WINDOWS_CSV = OUT_DIR / "base_windows.csv"

# Crop generation parameters
CLIP_LENGTH = 96
CROPS_PER_SWALLOW = 6
NEGATIVE_RATIO = .3
RANDOM_SEED = 42

# Train/val split
VAL_VIDEO_FRACTION = 0.2

# Fallback window parameters
PAD_BEFORE = 5
PAD_AFTER = 5
FALLBACK_AFTER_TOUCH = 10
FALLBACK_AFTER_BEFORE = 20
FALLBACK_BEFORE_TOUCH = 20

# Dry-run
DRY_RUN = False
DRY_RUN_ROWS = {41, 80, 331, 438, 654, 667}

# Tolerant fallback
TOLERANT_CLAMP = False
MIN_TOLERANT_WINDOW = 10

# -------------------------
# Helpers
# -------------------------
random.seed(RANDOM_SEED)

def video_path_from_name(video_name: str) -> Path:
    return VIDEO_DIR / video_name

def videos_from_csv(csv_path: Path) -> list:
    df_v = pd.read_csv(csv_path, dtype=str, usecols=["video"])
    df_v["video"] = df_v["video"].astype(str).str.strip()
    vids = sorted([v for v in df_v["video"].unique() if v and v.lower().endswith(".avi")])
    return vids

def to_offset(val, crop_start):
    try:
        if not pd.notna(val):
            return -1
    except Exception:
        if val is None:
            return -1
    try:
        return int(val) - int(crop_start)
    except Exception:
        return -1

def compute_base_window(b, t, l, total_frames, debug=False):
    def clamp(s, e):
        s = max(0, int(s))
        e = min(int(total_frames) - 1, int(e))
        if e < s:
            return None
        return (s, e)

    if b is None and t is None and l is None:
        return (None, []) if debug else None

    candidates = []
    present = [e for e in [b, t, l] if e is not None]

    if present:
        earliest = min(present)
        latest = max(present)
        w = clamp(earliest - PAD_BEFORE, latest + PAD_AFTER)
        if w: candidates.append(("earliest_latest", w))

    if b is not None and t is not None and l is not None:
        w = clamp(b - PAD_BEFORE, l + PAD_AFTER)
        if w: candidates.append(("A_all_three", w))

    if b is not None and t is not None and l is None:
        w = clamp(b - PAD_BEFORE, t + FALLBACK_AFTER_TOUCH)
        if w: candidates.append(("B_b_t", w))

    if b is not None and t is None and l is None:
        w = clamp(b - PAD_BEFORE, b + FALLBACK_AFTER_BEFORE)
        if w: candidates.append(("C_only_b", w))

    if b is None and (t is not None or l is not None):
        anchor = t if t is not None else l
        end_anchor = l if l is not None else anchor
        w = clamp(anchor - FALLBACK_BEFORE_TOUCH, end_anchor + PAD_AFTER)
        if w: candidates.append(("D_ues_anchor", w))

    events = [e for e in [b, t, l] if e is not None]
    if len(events) == 1:
        e = events[0]
        w = clamp(e - FALLBACK_BEFORE_TOUCH, e + FALLBACK_AFTER_BEFORE)
        if w: candidates.append(("E_single_event", w))

    if not candidates and present:
        w = clamp(earliest - PAD_BEFORE, latest + PAD_AFTER)
        if w: candidates.append(("safety_earliest_latest", w))

    if not candidates and present and TOLERANT_CLAMP:
        earliest = min(present)
        latest = max(present)
        e = int(total_frames) - 1
        s = max(0, min(int(earliest - PAD_BEFORE), e - MIN_TOLERANT_WINDOW + 1))
        if e - s + 1 >= MIN_TOLERANT_WINDOW:
            candidates.append(("tolerant_clamp", (s, e)))

    if not candidates:
        return (None, candidates) if debug else None

    best = max(candidates, key=lambda kv: kv[1][1] - kv[1][0])
    if debug:
        return best[1], candidates
    return best[1]

def ensure_dirs():
    (CROPS_DIR / "train").mkdir(parents=True, exist_ok=True)
    (CROPS_DIR / "val").mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def split_videos(videos):
    random.shuffle(videos)
    n_val = max(1, int(len(videos) * VAL_VIDEO_FRACTION))
    val = set(videos[:n_val])
    train = set(videos[n_val:])
    return sorted(train), sorted(val)

# -------------------------
# Network-safe extractor
# -------------------------

def extract_windows_sequential(video_path, windows):
    """
    Extract multiple windows from a single video in one sequential pass.
    Includes:
      - corrupt video detection
      - directory existence checks
      - write-failure detection
      - slow-IO detection
      - frame heartbeat
      - per-video start/finish logs
    """

    import time

    video_path = Path(video_path)
    print(f"\n[START] Extracting {len(windows)} crops from {video_path.name}")

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Sort windows by start frame
    windows = sorted(windows, key=lambda w: int(w['start']))
    if not windows:
        print(f"[WARN] No windows to extract for {video_path.name}")
        return

    # -------------------------
    # Open video
    # -------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # -------------------------
    # First-frame sanity check
    # -------------------------
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        cap.release()
        raise IOError(
            f"[CORRUPT] {video_path.name} opens but cannot decode frames "
            f"(corrupt file or unsupported codec)."
        )

    # Reset to frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # -------------------------
    # Clamp windows and ensure directories exist
    # -------------------------
    valid_windows = []
    for w in windows:
        s = max(0, min(int(w['start']), total_frames - 1))
        e = max(0, min(int(w['end']), total_frames - 1))

        if e < s:
            print(f"[WARN] Invalid window ({s},{e}) for crop {w['crop_id']} in {video_path.name}")
            continue

        out_dir = Path(w['out_dir'])
        if not out_dir.exists():
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except Exception as ex:
                raise IOError(f"Failed to create output directory {out_dir}: {ex}")

        valid_windows.append({
            'start': s,
            'end': e,
            'out_dir': out_dir,
            'crop_id': w.get('crop_id'),
            'next_write_index': 0
        })

    if not valid_windows:
        cap.release()
        print(f"[WARN] All windows invalid for {video_path.name}")
        return

    # -------------------------
    # Sequential extraction
    # -------------------------
    win_idx = 0
    active = []

    last_time = time.time()
    heartbeat_interval = 300  # frames
    slow_io_threshold = 5     # seconds per interval

    for f in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] Early termination reading {video_path.name} at frame {f}")
            break

        # Add windows starting at this frame
        while win_idx < len(valid_windows) and valid_windows[win_idx]['start'] == f:
            active.append(valid_windows[win_idx])
            win_idx += 1

        # -------------------------
        # Heartbeat + slow IO detection
        # -------------------------
        if f % heartbeat_interval == 0:
            now = time.time()
            dt = now - last_time
            print(f"[{video_path.name}] frame {f}/{total_frames} | active windows: {len(active)}")
            if dt > slow_io_threshold:
                print(f"[SLOW IO] {video_path.name} took {dt:.1f}s for {heartbeat_interval} frames")
            last_time = now

        # -------------------------
        # Write active windows
        # -------------------------
        if active:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            still_active = []

            for w in active:
                if f <= w['end']:
                    out_path = w['out_dir'] / f"frame_{w['next_write_index']:04d}.jpg"

                    # Ensure directory exists
                    if not w['out_dir'].exists():
                        try:
                            w['out_dir'].mkdir(parents=True, exist_ok=True)
                        except Exception as ex:
                            raise IOError(f"Failed to recreate directory {w['out_dir']}: {ex}")

                    # Write frame safely
                    ok = cv2.imwrite(str(out_path), gray, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if not ok:
                        raise IOError(
                            f"cv2.imwrite failed for {out_path}\n"
                            f"Directory exists: {w['out_dir'].exists()}"
                        )

                    w['next_write_index'] += 1
                    still_active.append(w)

            active = still_active

        # Early exit if all windows done
        if win_idx >= len(valid_windows) and not active:
            break

    cap.release()
    print(f"[DONE] Finished extracting {video_path.name}")



# -------------------------
# Main pipeline
# -------------------------
def main(limit_crops=None):
    ensure_dirs()

    df = pd.read_csv(CLEANED_CSV, dtype=str)
    df['csv_row'] = df.index
    df.columns = [c.strip().lower() for c in df.columns]

    for c in ["before_onset", "touch_ues", "leave_ues"]:
        df[c] = df[c].astype(str).str.strip().replace({"": None, "nan": None, "NaN": None})
        df[c] = pd.to_numeric(df[c], errors="coerce")

    coerced_counts = {c: int(df[c].isna().sum()) for c in ["before_onset", "touch_ues", "leave_ues"]}
    print("Coerced non-numeric counts:", coerced_counts)

    videos = videos_from_csv(CLEANED_CSV)
    train_videos, val_videos = split_videos(videos)

    missing = [v for v in videos if not video_path_from_name(v).exists()]
    if missing:
        print(f"[WARN] {len(missing)} referenced videos missing on network: {missing[:10]}")

    base_rows = []
    df_sorted = df.sort_values(["video", "before_onset", "touch_ues", "leave_ues"], na_position="last").reset_index(drop=True)

    for idx, row in df_sorted.iterrows():
        video = str(row["video"]).strip()
        video_path = video_path_from_name(video)
        if not video_path.exists():
            print(f"[WARN] video not found: {video} (skipping row {idx})")
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[WARN] cannot open video: {video} (skipping row {idx})")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        b = int(row["before_onset"]) if pd.notna(row["before_onset"]) else None
        t = int(row["touch_ues"]) if pd.notna(row["touch_ues"]) else None
        l = int(row["leave_ues"]) if pd.notna(row["leave_ues"]) else None

        window, candidates = compute_base_window(b, t, l, total_frames, debug=True)

        if window is None:
            csv_row = row.get("csv_row", "N/A")
            print(f"[INFO] row {idx} (csv_row={csv_row}) has no usable events; skipping")
            print(f"       video={video}  before_onset={row['before_onset']}  touch_ues={row['touch_ues']}  leave_ues={row['leave_ues']}")
            print(f"       total_frames={total_frames}")
            continue

        start_base, end_base = window
        base_rows.append({
            "row_index": idx,
            "csv_row": int(row.get("csv_row")) if pd.notna(row.get("csv_row")) else None,
            "video": video,
            "before_onset": b,
            "touch_ues": t,
            "leave_ues": l,
            "start_base": start_base,
            "end_base": end_base,
            "total_frames": total_frames
        })

    base_df = pd.DataFrame(base_rows)
    base_df.to_csv(BASE_WINDOWS_CSV, index=False)
    print(f"Wrote base windows for {len(base_df)} swallows to {BASE_WINDOWS_CSV}")

    # ---------------------------------------------------------
    # SUMMARY BLOCK — runs after base windows are generated
    # ---------------------------------------------------------
    print("\n========== DATASET SUMMARY ==========")

    print(f"Videos referenced in CSV: {len(videos)}")

    missing = [v for v in videos if not video_path_from_name(v).exists()]
    print(f"Missing videos on disk: {len(missing)}")
    if missing:
        print("  Missing examples:", missing[:5])

    total_rows = len(df)
    print(f"Total swallow rows in CSV: {total_rows}")

    valid_windows = len(base_df)
    skipped = total_rows - valid_windows
    print(f"Valid swallow windows: {valid_windows}")
    print(f"Skipped swallow rows: {skipped}")

    print("\nEvent completeness:")
    for col in ["before_onset", "touch_ues", "leave_ues"]:
        missing_count = df[col].isna().sum()
        present_count = total_rows - missing_count
        print(f"  {col}: present={present_count}, missing={missing_count}")

    def event_pattern(row):
        return (
            int(pd.notna(row["before_onset"])),
            int(pd.notna(row["touch_ues"])),
            int(pd.notna(row["leave_ues"]))
        )

    pattern_counts = defaultdict(int)
    for _, r in df.iterrows():
        pattern_counts[event_pattern(r)] += 1

    print("\nEvent pattern counts (b, t, l):")
    for pattern, count in sorted(pattern_counts.items()):
        print(f"  {pattern}: {count}")

    print("=====================================\n")

    # ---------------------------------------------------------
    # END SUMMARY BLOCK
    # ---------------------------------------------------------

    if DRY_RUN:
        print("DRY RUN: printing selected rows' computed windows")
        return

    # ---------------------------------------------------------
    # PREPARE CROP TASKS (grouped by video)
    # ---------------------------------------------------------
    crop_tasks_by_video = defaultdict(list)
    manifest_rows = []
    crop_id = 0

    video_list = videos  # only videos referenced in CSV

    for _, br in base_df.iterrows():
        if limit_crops and crop_id >= limit_crops:
            break

        video = br["video"]
        total_frames = int(br["total_frames"])
        b = br["before_onset"]
        t = br["touch_ues"]
        l = br["leave_ues"]

        split = "val" if video in val_videos else "train"

        # -------------------------
        # Safe anchor selection
        # -------------------------
        anchor = None
        for e in (b, t, l):
            if pd.notna(e):
                try:
                    anchor = int(e)
                    break
                except Exception:
                    continue
        if anchor is None:
            continue

        # -------------------------
        # Compute allowable crop_start range
        # -------------------------
        start_base = int(br["start_base"])
        end_base = int(br["end_base"])

        min_start = max(start_base, anchor - (CLIP_LENGTH - 1))
        max_start = min(end_base - CLIP_LENGTH + 1, anchor)

        # If base window shorter than clip length → pad outward
        if end_base - start_base + 1 < CLIP_LENGTH:
            pad_needed = CLIP_LENGTH - (end_base - start_base + 1)
            min_start = max(0, start_base - pad_needed)
            max_start = min(start_base, total_frames - CLIP_LENGTH)

        min_start = max(0, min_start)
        max_start = max(0, max_start)

        # If invalid range → center around anchor
        if max_start < min_start:
            center_start = max(0, min(total_frames - CLIP_LENGTH, anchor - CLIP_LENGTH // 2))
            min_start = max_start = center_start

        # -------------------------
        # Choose crop starts
        # -------------------------
        if split == "val":
            crop_starts = [
                int(max(0, min(total_frames - CLIP_LENGTH, anchor - CLIP_LENGTH // 2)))
            ]
        else:
            crop_starts = []
            for _ in range(CROPS_PER_SWALLOW):
                if min_start == max_start:
                    s = min_start
                else:
                    s = random.randint(int(min_start), int(max_start))
                crop_starts.append(s)

        # -------------------------
        # Register crops
        # -------------------------
        for s in crop_starts:
            if limit_crops and crop_id >= limit_crops:
                break

            crop_start = int(s)
            crop_end = crop_start + CLIP_LENGTH - 1

            crop_start = max(0, min(crop_start, total_frames - CLIP_LENGTH))
            crop_end = crop_start + CLIP_LENGTH - 1

            offset_b = to_offset(b, crop_start)
            offset_t = to_offset(t, crop_start)
            offset_l = to_offset(l, crop_start)

            has_event_in_crop = any(
                0 <= o < CLIP_LENGTH for o in [offset_b, offset_t, offset_l]
            )

            crop_id += 1
            crop_folder = CROPS_DIR / split / f"crop_{crop_id:06d}"

            crop_tasks_by_video[video].append({
                "start": crop_start,
                "end": crop_end,
                "out_dir": str(crop_folder),
                "crop_id": crop_id
            })

            manifest_rows.append({
                "crop_id": crop_id,
                "split": split,
                "video": video,
                "swallow_row_index": int(br["row_index"]),
                "csv_row": int(br.get("csv_row")) if pd.notna(br.get("csv_row")) else None,
                "crop_start": crop_start,
                "crop_end": crop_end,
                "clip_length": CLIP_LENGTH,
                "offset_before_onset": offset_b,
                "offset_touch_ues": offset_t,
                "offset_leave_ues": offset_l,
                "has_event": int(has_event_in_crop)
            })

    # ---------------------------------------------------------
    # NEGATIVE SAMPLING
    # ---------------------------------------------------------
    positives_count = sum(1 for r in manifest_rows if r["has_event"] == 1)
    negatives_target = int(positives_count * NEGATIVE_RATIO)

    neg_count = 0
    attempts = 0
    max_attempts = max(1000, negatives_target * 10)

    # Build lookup of positive event frames per video
    pos_frames = defaultdict(list)
    for _, r in base_df.iterrows():
        v = r["video"]
        for ev in ["before_onset", "touch_ues", "leave_ues"]:
            val = r.get(ev)
            if pd.notna(val):
                try:
                    pos_frames[v].append(int(val))
                except Exception:
                    continue

    # -------------------------
    # Sample negatives
    # -------------------------
    while neg_count < negatives_target and attempts < max_attempts:
        attempts += 1

        video = random.choice(video_list)
        video_path = video_path_from_name(video)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total_frames < CLIP_LENGTH:
            continue

        s = random.randint(0, total_frames - CLIP_LENGTH)
        e = s + CLIP_LENGTH - 1

        # Check overlap with positive event frames
        overlaps = False
        for ev_frame in pos_frames.get(video, []):
            if s <= ev_frame <= e:
                overlaps = True
                break

        if overlaps:
            continue

        # Register negative crop
        neg_count += 1
        crop_id += 1
        split = "train"
        crop_folder = CROPS_DIR / split / f"crop_{crop_id:06d}"

        crop_tasks_by_video[video].append({
            "start": s,
            "end": e,
            "out_dir": str(crop_folder),
            "crop_id": crop_id
        })

        manifest_rows.append({
            "crop_id": crop_id,
            "split": split,
            "video": video,
            "swallow_row_index": -1,
            "csv_row": None,
            "crop_start": s,
            "crop_end": e,
            "clip_length": CLIP_LENGTH,
            "offset_before_onset": -1,
            "offset_touch_ues": -1,
            "offset_leave_ues": -1,
            "has_event": 0
        })
    # ---------------------------------------------------------
    # EXTRACTION (network‑safe sequential read per video)
    # ---------------------------------------------------------
    total_tasks = sum(len(v) for v in crop_tasks_by_video.values())
    print(f"Starting extraction for {len(crop_tasks_by_video)} videos, total crops: {total_tasks}")

    for video, tasks in tqdm(crop_tasks_by_video.items(), desc="Videos"):
        video_path = video_path_from_name(video)
        try:
            extract_windows_sequential(video_path, tasks)
        except Exception as e:
            print(f"[ERROR] extraction failed for {video}: {e}")

    # ---------------------------------------------------------
    # WRITE MANIFEST
    # ---------------------------------------------------------
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = MANIFEST_DIR / "crops_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    print(f"Wrote manifest with {len(manifest_df)} crops to {manifest_path}")
    print(f"Total crops generated: {len(manifest_df)} "
          f"(positives: {positives_count}, negatives: {neg_count})")


# ---------------------------------------------------------
# MAIN GUARD
# ---------------------------------------------------------
if __name__ == "__main__":
    main(limit_crops=None)

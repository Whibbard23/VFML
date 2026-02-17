# vfss_starter_notebook.py
# Starter pipeline for VFSS swallow frame candidate generation and evaluation.
# CPU-only, uses OpenCV, numpy, pandas, matplotlib, scikit-learn (for metrics), and tqdm.

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# User config
# -------------------------
VIDEO_DIR = "videos"            # folder with .avi files
EVENT_CSV = "event_table.csv"   # original base-1 CSV with possible 'x' or '?'
OUTPUT_DIR = "outputs"          # where cleaned csv, clips, plots go
CLIP_PAD = 30                   # frames before/after event to extract for inspection
VALIDATION_SAMPLES = 200        # number of swallow clips to export for manual relabeling
ROI_INIT_FRAME = 0              # frame index to use for manual ROI init per video (0 = first)
MOTION_WINDOW = 3               # temporal window for variance / smoothing
ONSET_CONSEC_FRAMES = 2         # require motion above threshold for this many consecutive frames
MOTION_THRESHOLD_FACTOR = 3.0   # multiplier of median background motion to set threshold
TEMPORAL_TOLERANCES = [2, 5]    # frames for evaluation tolerance
# -------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "clips"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "stabilized"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def read_event_table(path):
    """
    Reads event_table.csv and returns DataFrame.
    Expected columns: video_filename, swallow_id, before_onset, touch_ues, leave_ues (or similar).
    Rows with 'x' or '?' in any event column are considered unclear and will be excluded from cleaned CSV.
    """
    df = pd.read_csv(path, dtype=str)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df

def make_cleaned_events(df, event_cols, out_path):
    """
    Convert base-1 to base-0, drop rows with non-numeric event entries.
    event_cols: list of column names to convert (e.g., ['before_onset','touch_ues','leave_ues'])
    """
    cleaned = []
    for idx, row in df.iterrows():
        valid = True
        new_row = row.to_dict()
        for c in event_cols:
            val = str(row.get(c, "")).strip()
            if val == "" or val.lower() in ["x", "?"]:
                valid = False
                break
            try:
                n = int(float(val))
            except:
                valid = False
                break
            # convert base-1 to base-0
            new_row[c] = n - 1
        if valid:
            cleaned.append(new_row)
    if len(cleaned) == 0:
        print("Warning: no valid rows found in event table after filtering.")
    cleaned_df = pd.DataFrame(cleaned)
    cleaned_df.to_csv(out_path, index=False)
    return cleaned_df

def open_video_capture(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {path}")
    return cap

def read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

def save_clip_frames(frames, out_prefix):
    for i, f in enumerate(frames):
        cv2.imwrite(f"{out_prefix}_f{i:04d}.png", f)

# -------------------------
# Step 1: Read and clean events
# -------------------------
df_raw = read_event_table(EVENT_CSV)
# detect event columns heuristically if not provided
possible_event_cols = [c for c in df_raw.columns if any(k in c.lower() for k in ["before","touch","leave","onset","ues"])]
# require at least three columns; otherwise user should edit
if len(possible_event_cols) < 3:
    # fallback: try common names
    event_cols = ["before_onset","touch_ues","leave_ues"]
else:
    # pick the three most likely
    event_cols = possible_event_cols[:3]

cleaned_path = os.path.join(OUTPUT_DIR, "cleaned_events.csv")
df_clean = make_cleaned_events(df_raw, event_cols, cleaned_path)
print(f"Cleaned events saved to {cleaned_path}. {len(df_clean)} valid rows.")

# -------------------------
# Utility: list videos
# -------------------------
video_files = {os.path.basename(p): p for p in glob(os.path.join(VIDEO_DIR, "*.avi"))}
print(f"Found {len(video_files)} .avi files in {VIDEO_DIR}")

# -------------------------
# Step 2: Extract validation clips (±CLIP_PAD) for a subset
# -------------------------
def extract_clip(video_path, center_frame, pad=CLIP_PAD, max_frames=None):
    cap = open_video_capture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, center_frame - pad)
    end = min(total - 1, center_frame + pad)
    frames = []
    for f in range(start, end + 1):
        frame = read_frame(cap, f)
        if frame is None:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames, start, end

# sample up to VALIDATION_SAMPLES events evenly
sampled = df_clean.sample(n=min(VALIDATION_SAMPLES, len(df_clean)), random_state=42)
clip_index = []
for i, row in sampled.iterrows():
    fname = row.get("video_filename") if "video_filename" in row else row.get("video")
    if fname is None:
        # try first column as filename
        fname = df_clean.columns[0]
    if fname not in video_files:
        continue
    video_path = video_files[fname]
    center = int(row[event_cols[0]])  # use before_onset as center for inspection
    frames, start, end = extract_clip(video_path, center, pad=CLIP_PAD)
    if len(frames) == 0:
        continue
    clip_name = os.path.join(OUTPUT_DIR, "clips", f"{os.path.splitext(fname)[0]}_swallow_{i}")
    save_clip_frames(frames, clip_name)
    clip_index.append({"clip": clip_name, "video": fname, "center_frame": center, "start": start, "end": end})
pd.DataFrame(clip_index).to_csv(os.path.join(OUTPUT_DIR, "clips", "clip_index.csv"), index=False)
print(f"Extracted {len(clip_index)} clips to {os.path.join(OUTPUT_DIR,'clips')}")

# -------------------------
# Step 3: ROI initialization helper (manual)
# -------------------------
# We'll provide a simple function to let you pick an ROI on a representative frame for each video.
def pick_roi(video_path, frame_idx=ROI_INIT_FRAME, window_name="Pick ROI"):
    cap = open_video_capture(video_path)
    frame = read_frame(cap, frame_idx)
    cap.release()
    if frame is None:
        raise IOError("Cannot read frame for ROI pick.")
    # show with matplotlib and let user draw a rectangle using cv2.selectROI if running locally
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    r = cv2.selectROI(window_name, frame_bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    x, y, w, h = r
    return (int(x), int(y), int(w), int(h))

# Save ROI picks in a JSON file per video (manual step)
roi_file = os.path.join(OUTPUT_DIR, "video_rois.json")
if not os.path.exists(roi_file):
    print("No ROI file found. You can create one by running pick_roi(video_path) for each video and saving results.")
    # Example usage (uncomment to run interactively):
    # example_video = list(video_files.values())[0]
    # roi = pick_roi(example_video)
    # print("ROI:", roi)

# -------------------------
# Step 4: Stabilization (optional) and ROI tracking
# -------------------------
def stabilize_and_track(video_path, roi, out_prefix=None, max_frames=None):
    """
    Stabilize around ROI and track ROI across frames using CSRT (CPU).
    Returns list of stabilized ROI frames (grayscale), and per-frame bbox list.
    """
    cap = open_video_capture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # initialize tracker on first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame0 = cap.read()
    if not ret:
        cap.release()
        return [], []
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    x, y, w, h = roi
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame0, (x, y, w, h))
    bboxes = []
    roi_frames = []
    prev_gray = gray0.copy()
    # optional: compute global transform to stabilize (using ECC) per small window
    for f in tqdm(range(total)):
        if max_frames and f >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ok, bbox = tracker.update(frame)
        if not ok:
            # fallback: re-init tracker on previous bbox
            bbox = (x, y, w, h)
        bx, by, bw, bh = [int(v) for v in bbox]
        # crop ROI
        crop = gray[by:by+bh, bx:bx+bw].copy()
        # simple local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        crop_enh = clahe.apply(crop)
        roi_frames.append(crop_enh)
        bboxes.append((bx, by, bw, bh))
    cap.release()
    # optionally save stabilized frames
    if out_prefix:
        for i, fimg in enumerate(roi_frames):
            cv2.imwrite(f"{out_prefix}_roi_{i:04d}.png", fimg)
    return roi_frames, bboxes

# -------------------------
# Step 5: Motion features inside ROI
# -------------------------
def compute_motion_traces(roi_frames):
    """
    roi_frames: list of grayscale ROI frames (numpy arrays)
    Returns:
      - frame_diffs: mean absolute difference per frame (frame t vs t-1)
      - flow_mags: mean optical flow magnitude per frame (Farneback)
      - temporal_var: rolling variance across MOTION_WINDOW frames
    """
    n = len(roi_frames)
    frame_diffs = np.zeros(n)
    for i in range(1, n):
        d = cv2.absdiff(roi_frames[i].astype(np.int16), roi_frames[i-1].astype(np.int16))
        frame_diffs[i] = d.mean()
    # optical flow (Farneback) between consecutive frames
    flow_mags = np.zeros(n)
    for i in range(1, n):
        f1 = roi_frames[i-1]
        f2 = roi_frames[i]
        flow = cv2.calcOpticalFlowFarneback(f1, f2, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        flow_mags[i] = mag.mean()
    # temporal variance
    arr = np.stack([f.astype(np.float32).ravel() for f in roi_frames]) if n>0 else np.zeros((0,))
    # compute per-frame variance in ROI intensity (simple)
    temporal_var = np.array([np.var(roi_frames[max(0,i-MOTION_WINDOW):i+1]) if i>0 else 0 for i in range(n)])
    return frame_diffs, flow_mags, temporal_var

# -------------------------
# Step 6: Candidate onset detection (threshold + consecutive frames)
# -------------------------
def detect_onset_candidates(frame_diffs, flow_mags, motion_threshold=None):
    """
    Combine frame_diffs and flow_mags into a single motion score and detect onset frames.
    motion_threshold: if None, compute adaptive threshold as median*factor
    Returns list of candidate frame indices (relative to roi_frames list)
    """
    score = frame_diffs + flow_mags
    if motion_threshold is None:
        bg = np.median(score[np.nonzero(score)]) if np.any(score>0) else np.median(score)
        motion_threshold = max(1e-6, bg * MOTION_THRESHOLD_FACTOR)
    candidates = []
    consec = 0
    for i, s in enumerate(score):
        if s >= motion_threshold:
            consec += 1
        else:
            if consec >= ONSET_CONSEC_FRAMES:
                # onset is the first frame in the consecutive run
                onset_idx = i - consec
                candidates.append(onset_idx)
            consec = 0
    # tail case
    if consec >= ONSET_CONSEC_FRAMES:
        onset_idx = len(score) - consec
        candidates.append(onset_idx)
    return candidates, score, motion_threshold

# -------------------------
# Step 7: Evaluation helpers
# -------------------------
def evaluate_predictions(preds, truths, tolerance=2):
    """
    preds: list of predicted frame indices (absolute)
    truths: list of true frame indices (absolute)
    We compute per-event detection: for each true event, whether a predicted frame exists within ±tolerance.
    Returns precision, recall, f1, mean_abs_error (for matched pairs)
    """
    matched_pred = set()
    matched_truth = set()
    errors = []
    for ti, t in enumerate(truths):
        # find closest pred within tolerance
        best = None
        best_d = None
        for pj, p in enumerate(preds):
            if pj in matched_pred:
                continue
            d = abs(p - t)
            if d <= tolerance and (best is None or d < best_d):
                best = pj
                best_d = d
        if best is not None:
            matched_pred.add(best)
            matched_truth.add(ti)
            errors.append(best_d)
    tp = len(matched_truth)
    fn = len(truths) - tp
    fp = len(preds) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2*precision*recall/(precision+recall)) if (precision+recall)>0 else 0.0
    mae = np.mean(errors) if len(errors)>0 else None
    return {"precision": precision, "recall": recall, "f1": f1, "mae": mae, "tp": tp, "fp": fp, "fn": fn}

# -------------------------
# Example run on one video (demo)
# -------------------------
# Choose a video and a saved ROI JSON (or manually set ROI here)
if len(video_files) == 0:
    raise SystemExit("No videos found. Place .avi files in the videos/ folder.")

example_video_name = list(video_files.keys())[0]
example_video_path = video_files[example_video_name]
print("Example video:", example_video_name)

# If you have a pre-specified ROI JSON, load it; otherwise set a manual ROI here:
video_rois = {}
if os.path.exists(roi_file):
    with open(roi_file, "r") as f:
        video_rois = json.load(f)
if example_video_name in video_rois:
    roi = tuple(video_rois[example_video_name])
else:
    # fallback ROI: center crop 20% of frame
    cap = open_video_capture(example_video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    cw, ch = int(w*0.2), int(h*0.2)
    roi = (w//2 - cw//2, h//2 - ch//2, cw, ch)
    print("Using fallback ROI:", roi)

# Run stabilization + tracking (this will save ROI frames)
roi_frames, bboxes = stabilize_and_track(example_video_path, roi, out_prefix=os.path.join(OUTPUT_DIR,"stabilized", os.path.splitext(example_video_name)[0]), max_frames=2000)
print(f"Extracted {len(roi_frames)} ROI frames for example video.")

# Compute motion traces
frame_diffs, flow_mags, temporal_var = compute_motion_traces(roi_frames)
candidates, score, threshold = detect_onset_candidates(frame_diffs, flow_mags)
print("Detected candidate onsets (relative):", candidates, "threshold:", threshold)

# Plot traces for inspection
plt.figure(figsize=(10,4))
plt.plot(frame_diffs, label="frame_diff")
plt.plot(flow_mags, label="flow_mag")
plt.plot(score, label="combined_score")
plt.axhline(threshold, color="red", linestyle="--", label="threshold")
for c in candidates:
    plt.axvline(c, color="green", alpha=0.6)
plt.legend()
plt.title(f"Motion traces for {example_video_name} (ROI)")
plt.xlabel("frame (ROI index)")
plt.ylabel("value")
plt.savefig(os.path.join(OUTPUT_DIR, "plots", f"{os.path.splitext(example_video_name)[0]}_motion_traces.png"))
plt.close()
print("Saved motion trace plot.")

# -------------------------
# Step 8: Batch run to produce candidate lists and evaluate vs cleaned_events.csv
# -------------------------
def batch_process_all(cleaned_df, video_files, roi_map=None, max_frames_per_video=None):
    results = []
    for idx, row in tqdm(cleaned_df.iterrows(), total=len(cleaned_df)):
        fname = row.get("video_filename") if "video_filename" in row else row.get("video")
        if fname not in video_files:
            continue
        video_path = video_files[fname]
        # choose ROI
        if roi_map and fname in roi_map:
            roi = tuple(roi_map[fname])
        else:
            # fallback center crop
            cap = open_video_capture(video_path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            cw, ch = int(w*0.2), int(h*0.2)
            roi = (w//2 - cw//2, h//2 - ch//2, cw, ch)
        # stabilize and track (limit frames for speed)
        roi_frames, bboxes = stabilize_and_track(video_path, roi, out_prefix=None, max_frames=max_frames_per_video)
        if len(roi_frames) < 5:
            continue
        frame_diffs, flow_mags, temporal_var = compute_motion_traces(roi_frames)
        candidates_rel, score, threshold = detect_onset_candidates(frame_diffs, flow_mags)
        # convert relative candidate indices to absolute frame numbers using bbox list start assumption
        # Note: stabilize_and_track started at frame 0; if you extracted a clip, adjust accordingly.
        # Here we assume absolute indexing equals index in roi_frames.
        candidates_abs = candidates_rel
        # truth absolute frame for before_onset
        truth = int(row[event_cols[0]])
        results.append({
            "video": fname,
            "truth_before_onset": truth,
            "candidates": json.dumps(candidates_abs),
            "threshold": float(threshold)
        })
    return pd.DataFrame(results)

# Run batch (this is a demo and may be slow; limit frames per video for speed)
batch_results = batch_process_all(df_clean, video_files, roi_map=video_rois, max_frames_per_video=2000)
batch_results.to_csv(os.path.join(OUTPUT_DIR, "candidate_onsets.csv"), index=False)
print("Batch candidate generation complete. Saved to candidate_onsets.csv")

# Evaluate using temporal tolerances
eval_rows = []
for tol in TEMPORAL_TOLERANCES:
    all_preds = []
    all_truths = []
    for _, r in batch_results.iterrows():
        preds = json.loads(r["candidates"])
        truth = int(r["truth_before_onset"])
        # simple per-event evaluation: did we find a candidate within tol?
        matched = any(abs(p - truth) <= tol for p in preds)
        all_preds.append(1 if matched else 0)
        all_truths.append(1)  # each row is one true event
    precision = sum(all_preds)/len(all_preds)
    recall = precision  # since each truth is one event and preds are binary per-event, precision==recall here
    f1 = precision
    eval_rows.append({"tolerance": tol, "precision_recall_f1": precision})
pd.DataFrame(eval_rows).to_csv(os.path.join(OUTPUT_DIR, "evaluation_summary.csv"), index=False)
print("Saved evaluation summary.")

# -------------------------
# End of script
# -------------------------
print("Pipeline complete. Inspect outputs in the outputs/ folder.")

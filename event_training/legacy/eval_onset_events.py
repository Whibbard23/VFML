#!/usr/bin/env python3
"""
Evaluate onset detection at event level with ±N frame tolerance.

Supports two onset selection modes:
  - first_frame (default): take the first frame of each positive segment
  - peak       : take the frame with highest probability inside each positive segment

Inputs:
  --pred-csv    : CSV with columns [video,frame,prob] (frame indices are integers)
  --events-csv  : events CSV that contains 'video','frame','event_type' (uses before_onset -> onset = frame+1)

Options:
  --tolerance   : integer N (frames) for matching tolerance (default 0)
  --prob-thresh : probability threshold to binarize predictions (default 0.5)
  --min-len     : minimum contiguous frames to consider an event (default 1)
  --mode        : 'first_frame' or 'peak' (default 'first_frame')
  --out-prefix  : prefix for output files (default: event_training/eval_onset)
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-csv", required=True, help="Predictions CSV with columns: video,frame,prob")
    p.add_argument("--events-csv", required=True, help="Events CSV containing before_onset rows")
    p.add_argument("--tolerance", type=int, default=0, help="±N frames tolerance for matching (default 0)")
    p.add_argument("--prob-thresh", type=float, default=0.5, help="Probability threshold to binarize (default 0.5)")
    p.add_argument("--min-len", type=int, default=1, help="Minimum contiguous frames to form an event (default 1)")
    p.add_argument("--mode", choices=["first_frame","peak"], default="first_frame", help="Onset selection mode")
    p.add_argument("--out-prefix", default="event_training/eval_onset", help="Prefix for output CSVs")
    return p.parse_args()

def load_predictions(pred_csv):
    df = pd.read_csv(pred_csv)
    required = {"video","frame","prob"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"pred-csv must contain columns: {required}")
    df["frame"] = df["frame"].astype(int)
    return df

def extract_true_onsets(events_csv):
    ev = pd.read_csv(events_csv, encoding="utf-8-sig")
    if "event_type" not in ev.columns or "video" not in ev.columns or "frame" not in ev.columns:
        raise ValueError("events-csv must contain columns: video,frame,event_type")
    before = ev[ev["event_type"] == "before_onset"].dropna(subset=["video","frame"])
    before["frame"] = before["frame"].astype(int)
    before["onset"] = before["frame"] + 1
    grouped = defaultdict(list)
    for _, r in before.iterrows():
        grouped[str(r["video"])].append(int(r["onset"]))
    for k in grouped:
        grouped[k] = sorted(set(grouped[k]))
    return grouped

def preds_to_onsets(df_preds, prob_thresh=0.5, min_len=1, mode="first_frame"):
    """
    For each video, threshold probs, merge contiguous frames into segments,
    return list of predicted onset frames according to mode.
    mode == "first_frame": first frame of segment
    mode == "peak": frame with highest prob inside segment
    """
    out = {}
    for video, g in df_preds.groupby("video"):
        g = g.sort_values("frame")
        frames = g["frame"].to_numpy(dtype=int)
        probs = g["prob"].to_numpy(dtype=float)
        mask = probs >= prob_thresh
        onsets = []
        if mask.size == 0:
            out[video] = []
            continue
        i = 0
        n = len(mask)
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and mask[j+1] and frames[j+1] == frames[j] + 1:
                j += 1
            run_len = j - i + 1
            if run_len >= min_len:
                if mode == "first_frame":
                    onsets.append(int(frames[i]))
                else:  # peak
                    seg_frames = frames[i:j+1]
                    seg_probs = probs[i:j+1]
                    peak_idx = int(np.argmax(seg_probs))
                    onsets.append(int(seg_frames[peak_idx]))
            i = j + 1
        out[video] = onsets
    return out

def match_onsets(true_onsets_map, pred_onsets_map, tol):
    matches = []
    per_video = {}
    all_videos = set(true_onsets_map.keys()) | set(pred_onsets_map.keys())
    for v in sorted(all_videos):
        true_list = sorted(true_onsets_map.get(v, []))
        pred_list = sorted(pred_onsets_map.get(v, []))
        matched_pred_idx = set()
        tp = 0
        local_matches = []
        for t in true_list:
            candidates = [(abs(p - t), idx, p) for idx, p in enumerate(pred_list) if abs(p - t) <= tol and idx not in matched_pred_idx]
            if candidates:
                candidates.sort()
                _, chosen_idx, chosen_p = candidates[0]
                matched_pred_idx.add(chosen_idx)
                err = chosen_p - t
                local_matches.append((v, t, chosen_p, err))
                tp += 1
        fp = len(pred_list) - len(matched_pred_idx)
        fn = len(true_list) - tp
        per_video[v] = {"tp": tp, "fp": fp, "fn": fn, "n_true": len(true_list), "n_pred": len(pred_list)}
        matches.extend(local_matches)
    return matches, per_video

def aggregate_metrics(per_video):
    total_tp = sum(x["tp"] for x in per_video.values())
    total_fp = sum(x["fp"] for x in per_video.values())
    total_fn = sum(x["fn"] for x in per_video.values())
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": total_tp, "fp": total_fp, "fn": total_fn}

def main():
    args = parse_args()
    pred_df = load_predictions(args.pred_csv)
    true_onsets = extract_true_onsets(args.events_csv)

    pred_onsets = preds_to_onsets(pred_df, prob_thresh=args.prob_thresh, min_len=args.min_len, mode=args.mode)

    matches, per_video = match_onsets(true_onsets, pred_onsets, args.tolerance)
    agg = aggregate_metrics(per_video)

    if matches:
        errors = np.array([m[3] for m in matches])
        mean_err = float(np.mean(errors))
        std_err = float(np.std(errors))
    else:
        mean_err = float("nan")
        std_err = float("nan")

    print("Event-level onset evaluation")
    print(f"Mode: {args.mode}  Tolerance (±frames): {args.tolerance}  Prob thresh: {args.prob_thresh}")
    print(f"Total true onsets: {sum(v['n_true'] for v in per_video.values())}")
    print(f"Total predicted onsets: {sum(v['n_pred'] for v in per_video.values())}")
    print(f"TP: {agg['tp']}  FP: {agg['fp']}  FN: {agg['fn']}")
    print(f"Precision: {agg['precision']:.4f}  Recall: {agg['recall']:.4f}  F1: {agg['f1']:.4f}")
    print(f"Onset error (pred - true): mean={mean_err:.3f}  std={std_err:.3f}")

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    per_video_df = pd.DataFrame.from_dict(per_video, orient="index").reset_index().rename(columns={"index":"video"})
    per_video_df.to_csv(f"{out_prefix}_per_video.csv", index=False)
    matches_df = pd.DataFrame(matches, columns=["video","true_onset","pred_onset","error"])
    matches_df.to_csv(f"{out_prefix}_matches.csv", index=False)

    summary = {
        "mode": args.mode,
        "tolerance": args.tolerance,
        "prob_thresh": args.prob_thresh,
        "total_true_onsets": int(sum(v['n_true'] for v in per_video.values())),
        "total_pred_onsets": int(sum(v['n_pred'] for v in per_video.values())),
        "tp": int(agg["tp"]), "fp": int(agg["fp"]), "fn": int(agg["fn"]),
        "precision": float(agg["precision"]), "recall": float(agg["recall"]), "f1": float(agg["f1"]),
        "mean_error": mean_err, "std_error": std_err
    }
    import json
    with open(f"{out_prefix}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote: {out_prefix}_per_video.csv, {out_prefix}_matches.csv, {out_prefix}_summary.json")

if __name__ == "__main__":
    main()

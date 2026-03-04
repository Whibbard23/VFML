#!/usr/bin/env python3
# apply_hysteresis_and_eval.py
"""
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"
python apply_hysteresis_and_eval.py `
  --probs runs/train_mouth_2/inference/probs.npy `
  --mapping runs/train_mouth_2/inference/index_to_row.csv `
  --labels runs/train_mouth_2/inference/val_labels.npy `
  --window 1 `
  --out-dir runs/eval_train_mouth_2

"""

import numpy as np, csv, argparse
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support

def load_mapping(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            rows.append({"index": int(r["index"]), "video": r["video"], "frame": int(r["frame"]), "label": int(r["label"])})
    return rows

def apply_hysteresis(pred, window=2):
    pred = pred.copy()
    n = len(pred)
    for i in range(n):
        if pred[i] == 0:
            start = max(0, i - window)
            end = min(n - 1, i + window)
            if pred[start:end+1].any():
                pred[i] = 1
    return pred

def per_video_group(mapping, probs, labels):
    by_video = {}
    for i, m in enumerate(mapping):
        by_video.setdefault(m["video"], []).append((i, m["frame"], probs[i], labels[i]))
    return by_video

def evaluate(pred, labels):
    prec, rec, f1, _ = precision_recall_fscore_support(labels, pred, average='binary', zero_division=0)
    return prec, rec, f1

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--probs", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--mapping", required=True, help="runs/inspect_val/index_to_row.csv")
    p.add_argument("--out-dir", default="runs/inspect_val/hysteresis")
    p.add_argument("--start", type=float, default=0.0005)
    p.add_argument("--stop", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--window", type=int, default=1, help="hysteresis window in frames (±window)")
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    probs = np.load(args.probs).ravel()
    labels = np.load(args.labels).ravel().astype(int)
    mapping = load_mapping(args.mapping)

    thresholds = np.linspace(args.start, args.stop, args.steps)
    best = None
    for t in thresholds:
        preds = (probs >= t).astype(int)
        preds_h = apply_hysteresis(preds, window=args.window)
        prec, rec, f1 = evaluate(preds_h, labels)
        if best is None or (rec > best[2]) or (rec == best[2] and f1 > best[3]):
            best = (t, prec, rec, f1)
    # write best summary
    t, prec, rec, f1 = best
    print(f"Best after hysteresis ±{args.window}: t={t:.6f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")
    # save predictions for that threshold
    preds = (probs >= t).astype(int)
    preds_h = apply_hysteresis(preds, window=args.window)
    # write detections CSV
    det_path = out / "detections_hysteresis.csv"
    with det_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["index","video","frame","label","prob","pred_h"])
        w.writeheader()
        for i, m in enumerate(mapping):
            w.writerow({"index": i, "video": m["video"], "frame": m["frame"], "label": m["label"], "prob": float(probs[i]), "pred_h": int(preds_h[i])})
    print("Saved detections to", det_path)

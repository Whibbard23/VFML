#!/usr/bin/env python3
# threshold_sweep.py
"""
Load runs/inspect_val/probs.npy and labels.npy, sweep thresholds, print metrics,
and report the lowest threshold that achieves the requested recall target.

Usage:
  python threshold_sweep.py --probs runs/inspect_val/probs.npy --labels runs/inspect_val/labels.npy --target-recall 0.98
"""
import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def sweep(probs, labels, thresholds):
    out = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        out.append((t, float(prec), float(rec), float(f1)))
    return out

def find_lowest_threshold_for_recall(sweep_results, target_recall):
    # sweep_results: list of (t, prec, rec, f1) sorted by t ascending
    candidates = [r for r in sweep_results if r[2] >= target_recall]
    return candidates[0] if candidates else None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probs", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--target-recall", type=float, default=0.98, help="Desired minimum recall (0-1)")
    p.add_argument("--start", type=float, default=0.01)
    p.add_argument("--stop", type=float, default=0.50)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--out", default="runs/inspect_val/threshold_sweep.csv")
    args = p.parse_args()

    probs = np.load(args.probs).ravel()
    labels = np.load(args.labels).ravel().astype(int)
    thresholds = np.linspace(args.start, args.stop, args.steps)

    results = sweep(probs, labels, thresholds)

    # write CSV
    import csv
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["threshold","precision","recall","f1"])
        for t, prec, rec, f1 in results:
            w.writerow([f"{t:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"])

    best = find_lowest_threshold_for_recall(results, args.target_recall)
    print(f"Saved sweep to: {args.out}")
    if best:
        t, prec, rec, f1 = best
        print(f"Lowest threshold achieving recall >= {args.target_recall:.3f}: t={t:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")
    else:
        # report top recalls
        top = sorted(results, key=lambda r: -r[2])[:5]
        print(f"No threshold in [{args.start},{args.stop}] achieved recall >= {args.target_recall:.3f}.")
        print("Top recall candidates:")
        for t, prec, rec, f1 in top:
            print(f"  t={t:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")

if __name__ == "__main__":
    main()

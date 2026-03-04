#!/usr/bin/env python3
import csv
import argparse
from collections import Counter

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV with video,frame,label,split")
    p.add_argument("--split", default="train", help="Which split to count (default: train)")
    p.add_argument("--cap", type=float, default=5.0, help="Max multiplier to report")
    p.add_argument("--safety-factor", type=float, default=0.5, help="Multiply heuristic by this factor")
    args = p.parse_args()

    c = Counter()
    with open(args.csv, "r", encoding="utf-8-sig") as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            if (r.get("split") or "").strip().lower() != args.split.lower():
                continue
            lab = (r.get("label") or "").strip()
            if lab == "":
                continue
            c[lab] += 1

    neg = c.get("0", 0) + c.get("false", 0)
    pos = c.get("1", 0) + c.get("true", 0)
    print(f"Counts (split={args.split}): negatives={neg}, positives={pos}")

    if pos == 0:
        print("No positives found; cannot compute heuristic pos_weight.")
        return

    heuristic = neg / pos
    suggested = min(heuristic * args.safety_factor, args.cap)
    print(f"Heuristic pos_weight = neg/pos = {heuristic:.3f}")
    print(f"Suggested pos-weight-multiplier = {suggested:.3f} (safety_factor={args.safety_factor}, cap={args.cap})")

if __name__ == "__main__":
    main()

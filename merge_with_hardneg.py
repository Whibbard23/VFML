#!/usr/bin/env python3
"""
merge_with_hardneg.py

Merge mined negatives into an existing hardneg CSV, preserving full metadata,
deduplicating by (video,frame), appending mined negatives with full crop metadata,
and enforcing a minimum validation fraction without altering mined negatives
or corrupting existing labels.

Usage example:
python merge_with_hardneg.py `
  --orig event_csvs/mouth_crops_labels_with_hardneg.csv `
  --mined runs/train_mouth_2/mined_negatives_for_train.csv `
  --meta event_csvs/mouth_crops_labels_with_split.csv `
  --out event_csvs/mouth_crops_labels_with_hardneg.csv `
  --min-val-frac 0.08 `
  --seed 42

"""
from pathlib import Path
import csv
import argparse
import random
import sys

def read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rdr = csv.DictReader(fh)
        fieldnames = rdr.fieldnames or []
        for r in rdr:
            rows.append({k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()})
    return rows, fieldnames

def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            # ensure all fieldnames exist in row (write empty string if missing)
            out = {k: (r.get(k, "") if r.get(k, "") is not None else "") for k in fieldnames}
            w.writerow(out)

def normalize_key(video, frame):
    return (video.strip(), str(int(float(frame))))

def main():
    p = argparse.ArgumentParser(description="Merge mined negatives into hardneg CSV preserving metadata")
    p.add_argument("--orig", required=True, help="Existing hardneg CSV with full metadata (video,frame,label,xc,yc,w,h,width,height,filepath,split,...)")
    p.add_argument("--mined", required=True, help="Mined negatives CSV (must contain video,frame; may include prob)")
    p.add_argument("--meta", required=True, help="Full metadata CSV used to build crops (mouth_crops_labels_with_split.csv)")
    p.add_argument("--out", required=True, help="Output augmented CSV")
    p.add_argument("--min-val-frac", type=float, default=0.08, help="Minimum fraction of rows assigned to val")
    p.add_argument("--seed", type=int, default=0, help="Random seed for deterministic sampling")
    args = p.parse_args()

    random.seed(args.seed)

    orig_path = Path(args.orig)
    mined_path = Path(args.mined)
    meta_path = Path(args.meta)
    out_path = Path(args.out)

    if not orig_path.exists():
        print(f"ERROR: orig CSV not found: {orig_path}", file=sys.stderr); sys.exit(2)
    if not mined_path.exists():
        print(f"ERROR: mined CSV not found: {mined_path}", file=sys.stderr); sys.exit(2)
    if not meta_path.exists():
        print(f"ERROR: meta CSV not found: {meta_path}", file=sys.stderr); sys.exit(2)

    orig_rows, orig_fields = read_csv(orig_path)
    mined_rows, mined_fields = read_csv(mined_path)
    meta_rows, meta_fields = read_csv(meta_path)

    # Build metadata lookup from meta CSV (full crop metadata)
    meta_lookup = {}
    for r in meta_rows:
        try:
            key = normalize_key(r["video"], r["frame"])
        except Exception:
            continue
        meta_lookup[key] = r

    # Build merged dict starting from original rows (preserve full metadata and original splits/labels)
    merged = {}
    for r in orig_rows:
        try:
            key = normalize_key(r.get("video", ""), r.get("frame", "0"))
        except Exception:
            continue
        merged[key] = r.copy()

    # Determine final fieldnames: start with orig_fields, then add any meta fields not present,
    # then any mined fields (e.g., prob) not present. Keep order stable.
    final_fields = list(orig_fields)
    for f in meta_fields:
        if f and f not in final_fields:
            final_fields.append(f)
    for f in mined_fields:
        if f and f not in final_fields:
            final_fields.append(f)
    # Ensure core columns exist
    for core in ("video", "frame", "label", "split"):
        if core not in final_fields:
            final_fields.append(core)

    # Append mined negatives: if already present in merged, skip; otherwise, look up full metadata in meta_lookup
    skipped_missing_meta = []
    added = 0
    for r in mined_rows:
        # mined CSV may have header like 'video','frame','prob' or similar
        if not r.get("video") or not r.get("frame"):
            continue
        key = normalize_key(r["video"], r["frame"])
        if key in merged:
            continue  # preserve existing full metadata and original label/split
        if key not in meta_lookup:
            skipped_missing_meta.append(key)
            continue
        full = meta_lookup[key].copy()
        # ensure mined negatives are labeled 0 and assigned to train; never overwrite other fields
        full["label"] = "0"
        full["split"] = "train"
        # if mined CSV contains prob, preserve it in a 'prob' column if present in final_fields
        if "prob" in r and "prob" in final_fields:
            full["prob"] = r.get("prob", "")
        merged[key] = full
        added += 1

    if skipped_missing_meta:
        print(f"Warning: {len(skipped_missing_meta)} mined rows were skipped because metadata lookup failed. Example: {skipped_missing_meta[:5]}")

    # Convert merged dict to list
    rows = list(merged.values())

    # Compute current val fraction
    total = len(rows)
    val_count = sum(1 for r in rows if r.get("split", "").lower() == "val")
    val_frac = val_count / total if total > 0 else 0.0

    # If val fraction is too small, promote some train positives to val (do not promote mined negatives)
    if val_frac < args.min_val_frac:
        need = int(args.min_val_frac * total) - val_count
        if need > 0:
            # Candidates: train positives that are not mined negatives (i.e., present in orig or meta but not newly added mined)
            # We prefer original train positives (orig_rows) to avoid promoting newly appended mined negatives.
            orig_keys = {normalize_key(r.get("video",""), r.get("frame","0")) for r in orig_rows}
            candidates = []
            for r in rows:
                key = normalize_key(r.get("video",""), r.get("frame","0"))
                if r.get("split","").lower() != "train":
                    continue
                lab = str(r.get("label","0")).strip().lower()
                is_pos = lab in ("1","1.0","pos","positive","true","t")
                if not is_pos:
                    continue
                # prefer original rows (not newly added mined negatives)
                if key in orig_keys:
                    candidates.append((0, r["video"], int(float(r["frame"])), r))
                else:
                    # deprioritize rows that came from mined set (shouldn't be positives, but safe)
                    candidates.append((1, r["video"], int(float(r["frame"])), r))
            # If not enough positive candidates, allow train negatives (orig) as fallback
            if len(candidates) < need:
                for r in rows:
                    key = normalize_key(r.get("video",""), r.get("frame","0"))
                    if r.get("split","").lower() != "train":
                        continue
                    lab = str(r.get("label","0")).strip().lower()
                    if lab in ("1","1.0","pos","positive","true","t"):
                        continue  # already included
                    if key in orig_keys:
                        candidates.append((2, r["video"], int(float(r["frame"])), r))
            # Deterministic sort and select
            candidates.sort(key=lambda x: (x[0], x[1], x[2]))
            to_promote = [c[3] for c in candidates[:need]]
            for r in to_promote:
                r["split"] = "val"
            val_count = sum(1 for r in rows if r.get("split", "").lower() == "val")
            val_frac = val_count / total

    # Write output CSV using final_fields (preserve orig order + extras)
    write_csv(out_path, rows, final_fields)

    print(f"Wrote {out_path} rows={len(rows)} added_mined={added} val_count={val_count} val_frac={val_frac:.3f}")
    if skipped_missing_meta:
        print(f"Skipped {len(skipped_missing_meta)} mined rows due to missing metadata. Provide a meta CSV that contains those (video,frame) pairs to include them.")

if __name__ == "__main__":
    main()

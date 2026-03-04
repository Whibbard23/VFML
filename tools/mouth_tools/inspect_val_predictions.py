#!/usr/bin/env python3
# inspect_val_predictions.py
"""
Compute validation predictions and save FP/FN/TP CSVs plus probs.npy and labels.npy.

Usage example:
python inspect_val_predictions.py \
  --csv event_csvs/mouth_crops_labels_with_split.csv \
  --ckpt runs/train_mouth_1/best.pth \
  --frames-root runs/inference \
  --device cpu \
  --batch-size 8 \
  --num-workers 0 \
  --out-dir runs/inspect_val
"""
from pathlib import Path
import csv
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from event_training.training.train_mouth_model import MouthDataset, ResNet18EarlyFusion

def load_model(ckpt_path, device, no_pretrained=True):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ResNet18EarlyFusion(pretrained=not no_pretrained)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model

def run_inspect(csv_path, ckpt_path, frames_root, device_str, batch_size, num_workers, out_dir):
    device = torch.device(device_str)
    ds_val = MouthDataset(csv_path, frames_root=frames_root, split="val", train=False)
    if len(ds_val) == 0:
        raise RuntimeError("Validation dataset is empty for split 'val' in the provided CSV.")
    collate = lambda b: (torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b]))
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)

    model = load_model(ckpt_path, device, no_pretrained=True)

    all_logits = []
    all_probs = []
    all_labels = []
    all_rows = ds_val.rows  # list of dicts in the same order as dataset

    with torch.no_grad():
        for x, y in dl_val:
            x = x.to(device)
            logits = model(x).cpu().numpy().ravel()
            probs = 1.0 / (1.0 + np.exp(-logits))
            all_logits.extend(logits.tolist())
            all_probs.extend(probs.tolist())
            all_labels.extend(y.numpy().tolist())

    probs = np.array(all_probs, dtype=np.float32).ravel()
    labels = np.array(all_labels, dtype=np.int8).ravel()
    preds = (probs >= 0.5).astype(int)

    fp = []
    fn = []
    tp = []
    tn = []
    for row, p, y, prob in zip(all_rows, preds, labels, probs):
        entry = {"video": row["video"], "frame": row["frame"], "label": int(y), "prob": float(prob)}
        if p == 1 and y == 0:
            fp.append(entry)
        elif p == 0 and y == 1:
            fn.append(entry)
        elif p == 1 and y == 1:
            tp.append(entry)
        else:
            tn.append(entry)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_list(lst, name):
        p = out_dir / f"{name}.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["video","frame","label","prob"])
            w.writeheader()
            for r in lst:
                w.writerow(r)
        return p

    p_fp = write_list(fp, "false_positives")
    p_fn = write_list(fn, "false_negatives")
    p_tp = write_list(tp, "true_positives")

    # Save probs and labels for later threshold sweeps
    np.save(out_dir / "probs.npy", probs)
    np.save(out_dir / "labels.npy", labels)

    # Save index -> (video,frame) mapping so rows can be traced back reliably
    mapping_path = out_dir / "index_to_row.csv"
    with mapping_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["index","video","frame","label"])
        w.writeheader()
        for i, r in enumerate(all_rows):
            w.writerow({"index": i, "video": r["video"], "frame": r["frame"], "label": int(r["label"])})

    # summary
    print(f"VAL size: {len(all_rows)}")
    print(f"TP: {len(tp)}  FP: {len(fp)}  FN: {len(fn)}  TN: {len(tn)}")
    print(f"Saved FP -> {p_fp}")
    print(f"Saved FN -> {p_fn}")
    print(f"Saved TP -> {p_tp}")
    print(f"Saved probs.npy and labels.npy to {out_dir}")
    print(f"Saved index mapping to {mapping_path}")

    return probs, labels, preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--ckpt", required=True, help="path to best.pth or last.pth")
    parser.add_argument("--frames-root", default="runs/inference")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-dir", default="runs/inspect_val")
    args = parser.parse_args()

    run_inspect(args.csv, args.ckpt, args.frames_root, args.device, args.batch_size, args.num_workers, args.out_dir)

#!/usr/bin/env python3
# hard_negative_mine.py
"""
Score candidate negative frames with a trained model and write high-confidence
hard negatives to a CSV for use in training.

"""
from pathlib import Path
import csv
import argparse
import time
from collections import defaultdict

import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import DataLoader, Dataset

# reuse model and dataset helpers from training module
from event_training.training.train_mouth_model import MouthDataset, ResNet18EarlyFusion

# limit PyTorch intra-op threads to avoid oversubscription on CPU
torch.set_num_threads(1)

# -------------------------
# Module-level utilities
# -------------------------
def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    return xs, ys

class CandDataset(Dataset):
    """
    Dataset for scoring a list of candidate rows (dicts with keys: video, frame, label).
    Uses MouthDataset helper methods for loading and transforms but does not filter by split.

    This implementation:
    - Reuses helper._load_rgb/_load_motion when available (preferred).
    - If helper lacks CLAHE or _load_rgb, applies CLAHE locally to grayscale crop.
    - Keeps joint geometric transforms and color jitter identical to training.
    """
    def __init__(self, rows, csv_path, frames_root, resize=224):
        # instantiate a helper MouthDataset to reuse transforms and loader helpers
        # set train=False so transforms are deterministic for scoring
        self._helper = MouthDataset(csv_path, frames_root=frames_root, resize=resize, split="train", train=False)
        self.rows = rows

        # If helper provides a CLAHE instance, prefer it; otherwise create one locally
        self._clahe = getattr(self._helper, "clahe", None)
        if self._clahe is None:
            # match ROI detector parameters
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.rows)

    def _load_rgb_clahe_local(self, p: Path):
        # local loader that mirrors MouthDataset._load_rgb behavior (grayscale -> CLAHE -> RGB PIL)
        if not p.exists():
            blank = np.zeros((self._helper.resize, self._helper.resize), dtype=np.uint8)
            blank = cv2.cvtColor(blank, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(blank)

        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            gray = np.zeros((self._helper.resize, self._helper.resize), dtype=np.uint8)

        gray = self._clahe.apply(gray)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)

    def __getitem__(self, idx):
        r = self.rows[idx]
        crop_p, motion_p = self._helper._resolve_paths(r["video"], int(r["frame"]))

        # Prefer helper loader (it should already apply CLAHE). If not present, use local loader.
        try:
            rgb = self._helper._load_rgb(crop_p)
        except Exception:
            rgb = self._load_rgb_clahe_local(crop_p)

        # Motion map (unchanged)
        try:
            motion = self._helper._load_motion(motion_p)
        except Exception:
            # fallback: zeros
            motion = np.zeros((self._helper.resize, self._helper.resize), dtype=np.float32)

        motion_pil = Image.fromarray((np.clip(motion, 0, 1) * 255).astype("uint8"))

        # apply same geometric transforms used by helper
        rgb_geo = self._helper.geo_transform(rgb)
        motion_geo = self._helper.geo_transform(motion_pil)
        if self._helper.color_transform is not None:
            rgb_geo = self._helper.color_transform(rgb_geo)

        rgb_t = self._helper.to_tensor(rgb_geo)
        rgb_t = self._helper.rgb_norm(rgb_t)
        motion_t = self._helper.to_tensor(motion_geo)
        input_t = torch.cat([rgb_t, motion_t], dim=0)
        # return dummy label tensor (not used for scoring)
        return input_t, torch.tensor(0.0, dtype=torch.float32)

# -------------------------
# Model loader
# -------------------------
def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ResNet18EarlyFusion(pretrained=False)
    # support checkpoints that store either 'model_state' or raw state_dict
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# -------------------------
# Incremental scoring helper
# -------------------------
def score_and_write_incremental(model, rows, csv_out_path, csv_in_path, frames_root, device, batch_size, num_workers, top_k_per_video, global_top_n, resume):
    """
    Score candidate negatives grouped by video, write top-K per video immediately,
    and optionally add global top-N after all videos. Supports resume.
    """
    by_video = defaultdict(list)
    for r in rows:
        by_video[r["video"]].append(r)

    outp = Path(csv_out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # resume: load already written keys
    written_keys = set()
    if resume and outp.exists():
        try:
            with outp.open("r", encoding="utf-8", newline="") as fh:
                rdr = csv.DictReader(fh)
                for r in rdr:
                    written_keys.add((r["video"], int(r["frame"])))
        except Exception:
            written_keys = set()

    # open output CSV for append
    fh_out = outp.open("a", encoding="utf-8", newline="")
    writer = csv.writer(fh_out)
    # write header if new file
    try:
        if outp.stat().st_size == 0:
            writer.writerow(["video", "frame", "label", "split", "score"])
    except FileNotFoundError:
        writer.writerow(["video", "frame", "label", "split", "score"])

    model.to(device)
    model.eval()

    total_scored = 0
    try:
        n_videos = len(by_video)
        for vid_idx, (video, group) in enumerate(by_video.items(), start=1):
            # filter out already written frames for resume
            group = [g for g in group if (g["video"], int(g["frame"])) not in written_keys]
            if not group:
                print(f"[{vid_idx}/{n_videos}] video={video} skipped (already processed)")
                continue

            # create CandDataset for this video only
            cand_ds = CandDataset(group, csv_in_path, frames_root)
            dl = DataLoader(cand_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

            # score this video's candidates
            with torch.no_grad():
                batch_idx = 0
                for x, _ in dl:
                    batch_idx += 1
                    batch_start = time.time()
                    x = x.to(device)
                    logits = model(x)
                    p = torch.sigmoid(logits).cpu().numpy().ravel()
                    # attach probabilities back to group entries
                    base_idx = (batch_idx - 1) * dl.batch_size
                    for i, prob in enumerate(p):
                        idx = base_idx + i
                        if idx < len(group):
                            group[idx]["_prob"] = float(prob)
                    total_scored += x.shape[0]
                    print(f"[{vid_idx}/{n_videos}] video={video} batch={batch_idx} size={x.shape[0]} time_s={(time.time()-batch_start):.3f}")
            # select top-K for this video and write immediately
            group_sorted = sorted(group, key=lambda x: -x.get("_prob", 0.0))
            for sel in group_sorted[:top_k_per_video]:
                key = (sel["video"], int(sel["frame"]))
                if key in written_keys:
                    continue
                writer.writerow([sel["video"], int(sel["frame"]), 0, "train", f"{sel['_prob']:.6f}"])
                written_keys.add(key)
            fh_out.flush()
        # optionally add global top-N after all videos
        if global_top_n and global_top_n > 0:
            scored = [r for r in rows if "_prob" in r]
            global_sorted = sorted(scored, key=lambda x: -x["_prob"])
            added = 0
            for sel in global_sorted:
                key = (sel["video"], int(sel["frame"]))
                if key in written_keys:
                    continue
                writer.writerow([sel["video"], int(sel["frame"]), 0, "train", f"{sel['_prob']:.6f}"])
                written_keys.add(key)
                added += 1
                if added >= global_top_n:
                    break
            fh_out.flush()
    except KeyboardInterrupt:
        print("Interrupted by user — flushing progress and exiting cleanly.")
    finally:
        fh_out.close()
    print("Scoring complete. Total scored (approx):", total_scored)

# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV with video,frame,label,split")
    p.add_argument("--ckpt", required=True, help="Path to model checkpoint (best.pth)")
    p.add_argument("--frames-root", default="runs/inference")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out-csv", required=True, help="Output CSV for mined negatives")
    p.add_argument("--top-k-per-video", type=int, default=5, help="Top-K highest-prob negatives per video")
    p.add_argument("--global-top-n", type=int, default=0, help="Also select global top-N negatives")
    p.add_argument("--resume", action="store_true", help="Resume from existing output CSV if present")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # load candidate negatives from train split
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            split = (r.get("split") or "").strip().lower()
            if split != "train":
                continue
            video = (r.get("video") or "").strip()
            if not video:
                continue
            frame = int(float(r.get("frame") or 0))
            label = int(float(r.get("label") or 0))
            if label == 0:
                rows.append({"video": video, "frame": frame, "label": label})

    if len(rows) == 0:
        raise RuntimeError("No candidate negatives found in train split of CSV.")

    device = torch.device(args.device)
    model = load_model(args.ckpt, device)

    # run incremental scoring and write results
    score_and_write_incremental(
        model=model,
        rows=rows,
        csv_out_path=args.out_csv,
        csv_in_path=args.csv,
        frames_root=args.frames_root,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        top_k_per_video=args.top_k_per_video,
        global_top_n=args.global_top_n,
        resume=args.resume
    )

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
event_training/mouth/train_mouth_detector.py

CPU-only framewise baseline training for swallow-onset detection.

Features:
- Reads labels from ../../event_csvs/mouth_frame_label_table.csv by default.
- Loads per-frame JPEGs (preferred) or .npy normalized tensors as fallback.
- Balanced minibatches via BalancedBatchSampler (positive oversampling).
- Moderate class weight in BCEWithLogitsLoss.
- Augmentations: brightness +-20%, rotation +-5 degrees, random scale +-5% (no horizontal flips).
- AdamW optimizer with lr=1e-4.
- Saves per-epoch validation PR curve PNGs.
- Threshold selection routine: picks threshold that achieves at least `--target-recall`
  (default 0.995) on validation while maximizing precision (minimizes false positives).
  If no threshold meets target recall, falls back to threshold maximizing F1.
- Checkpointing and best-model saving.
- CPU-only (no CUDA usage).

USAGE: (Powershell)
  python event_training/mouth/train_mouth_detector.py `
    --label-csv "event_csvs/mouth_frame_label_table.csv" `
    --crops-root "E:/VF ML Crops" `
    --out-dir "C:/Users/Connor Lab/Desktop/VFML/event_training/mouth/models/event_baseline" `
    --epochs 30 `
    --batch-size 64 `
    --img-size 128 `
    --lr 1e-4 `
    --weight-decay 1e-4 `
    --backbone resnet18 `
    --device cpu `
    --num-workers 0 `
    --patience 6 `
    --target-recall 0.995
"""
from pathlib import Path
import argparse
import csv
import json
import os
import random
import math
from typing import Iterator, List
from collections import Counter

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, auc

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models

# Try to import the new Weights enums if available (torchvision >= 0.13)
try:
    # type: ignore
    from torchvision.models import ResNet18_Weights  # noqa: F401
    _HAS_WEIGHTS_ENUM = True
except Exception:
    _HAS_WEIGHTS_ENUM = False

# -------------------------
# Dataset
# -------------------------
FRAME_NAME = "frame_{:06d}"


class MouthFrameDataset(Dataset):
    """
    Loads samples listed in mouth_frame_label_table.csv for a given split (train/val).
    Prefers JPEGs under <crops_root>/<video>/crops/mouth/frame_XXXXXX.jpg
    Falls back to .npy under <crops_root>/<video>/crops_normalized/mouth/frame_XXXXXX.npy
    """

    def __init__(self, label_csv: Path, crops_root: Path, split: str, transform=None, prefer_jpeg=True):
        self.crops_root = Path(crops_root)
        self.transform = transform
        self.prefer_jpeg = prefer_jpeg
        self.samples = []  # tuples: (video, frame_int, label, jpeg_path or None, npy_path or None)

        if not Path(label_csv).exists():
            raise FileNotFoundError(f"Label CSV not found: {label_csv}")

        with open(label_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("split", "").strip().lower() != split:
                    continue
                vid = r["video"].strip()
                if vid.lower().endswith(".avi"):
                    vid = vid[:-4]
                frm = int(r["frame"])
                lbl = int(r["label"])
                jpeg_path = self.crops_root / vid / "crops" / "mouth" / f"{FRAME_NAME.format(frm)}.jpg"
                npy_path = self.crops_root / vid / "crops_normalized" / "mouth" / f"{FRAME_NAME.format(frm)}.npy"
                jpeg_exists = jpeg_path.exists()
                npy_exists = npy_path.exists()
                if not jpeg_exists and not npy_exists:
                    # skip missing files silently (user can inspect logs)
                    continue
                self.samples.append((vid, frm, lbl, jpeg_path if jpeg_exists else None, npy_path if npy_exists else None))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split={split} (checked {label_csv} and {crops_root})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid, frm, lbl, jpeg_path, npy_path = self.samples[idx]
        if jpeg_path is not None and self.prefer_jpeg:
            img = Image.open(jpeg_path).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            else:
                img = T.ToTensor()(img)
            return img, torch.tensor(lbl, dtype=torch.float32)
        elif npy_path is not None:
            arr = np.load(npy_path)  # expected CHW or HWC float32
            t = torch.from_numpy(arr).float()
            # If HWC convert to CHW
            if t.ndim == 3 and t.shape[2] == 3:
                t = t.permute(2, 0, 1)
            # If values are normalized (ImageNet), we still apply simple tensor-space augmentations:
            t = self._tensor_augment(t)
            return t, torch.tensor(lbl, dtype=torch.float32)
        else:
            raise RuntimeError("No image for sample")

    def _tensor_augment(self, t):
        # t: CHW tensor, values may be normalized; apply mild brightness scaling only
        # brightness multiplier in [0.8, 1.2]
        mul = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
        t = t * mul
        return t.clamp(-10.0, 10.0)  # allow wide clamp if normalized


# -------------------------
# Balanced batch sampler
# -------------------------
class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    Yields batches of indices with a fixed number of positives per batch.
    - labels: 1D array-like of 0/1 labels (length == dataset size)
    - batch_size: total batch size
    - pos_per_batch: number of positive samples per batch (int)
    Positives are sampled with replacement (useful when positives << negatives).
    Negatives are sampled without replacement and reshuffled when exhausted.
    """
    def __init__(self, labels: List[int], batch_size: int, pos_per_batch: int):
        self.labels = np.asarray(labels)
        self.batch_size = int(batch_size)
        self.pos_per_batch = int(pos_per_batch)
        if self.pos_per_batch >= self.batch_size:
            raise ValueError("pos_per_batch must be < batch_size")
        self.neg_per_batch = self.batch_size - self.pos_per_batch

        self.pos_idx = list(np.where(self.labels == 1)[0])
        self.neg_idx = list(np.where(self.labels == 0)[0])
        if len(self.pos_idx) == 0 or len(self.neg_idx) == 0:
            raise ValueError("Need at least one positive and one negative sample for BalancedBatchSampler")

        # estimate number of batches per epoch (cover all samples roughly once)
        self.num_samples = len(self.labels)
        self.num_batches = math.ceil(self.num_samples / self.batch_size)

    def __len__(self):
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        # shuffle negatives each epoch
        neg_pool = self.neg_idx.copy()
        random.shuffle(neg_pool)
        neg_ptr = 0

        for _ in range(self.num_batches):
            # sample positives with replacement
            pos_batch = random.choices(self.pos_idx, k=self.pos_per_batch)

            # sample negatives without replacement; reshuffle if not enough left
            if neg_ptr + self.neg_per_batch > len(neg_pool):
                neg_pool = self.neg_idx.copy()
                random.shuffle(neg_pool)
                neg_ptr = 0
            neg_batch = neg_pool[neg_ptr: neg_ptr + self.neg_per_batch]
            neg_ptr += self.neg_per_batch

            batch_indices = pos_batch + neg_batch
            random.shuffle(batch_indices)
            yield batch_indices


# -------------------------
# Model
# -------------------------
def build_model(backbone="resnet18", pretrained=True):
    """
    Build model using new torchvision weights= API when available.
    Falls back to legacy pretrained= argument if Weights enum is not present.
    """
    if backbone == "resnet18":
        if _HAS_WEIGHTS_ENUM:
            # Choose the appropriate weights enum when pretrained requested
            if pretrained:
                # prefer the DEFAULT enum if available, otherwise use IMAGENET1K_V1
                try:
                    weights = ResNet18_Weights.DEFAULT
                except Exception:
                    weights = ResNet18_Weights.IMAGENET1K_V1
            else:
                weights = None
            model = models.resnet18(weights=weights)
        else:
            # older torchvision: use legacy pretrained flag
            model = models.resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
        return model
    else:
        raise ValueError("Unsupported backbone")


# -------------------------
# Transforms and utils
# -------------------------
def make_transforms(img_size=128, train=True):
    # brightness +-20%, rotation +-5deg, scale 0.95-1.05, no horizontal flip
    if train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomAffine(degrees=5, translate=(0.0, 0.0), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.2),
            T.ToTensor(),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])


def compute_pos_weight(labels):
    # labels: 1D numpy array or list of 0/1
    labels = np.asarray(labels)
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    if pos == 0:
        return 1.0
    ratio = neg / max(1, pos)
    return float(min(ratio, 5.0))


def save_pr_curve(y_true, y_scores, out_path: Path, epoch: int):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"PR AUC={pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Validation PR Curve - Epoch {epoch}")
    plt.grid(True)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return precision, recall, thresholds


def select_threshold_for_target_recall(y_true, y_scores, target_recall=0.995):
    """
    Choose threshold that achieves recall >= target_recall and maximizes precision.
    If none meet target_recall, choose threshold that maximizes F1.
    Returns: chosen_threshold, info_dict
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    best_thresh = None
    best_prec = -1.0
    best_rec = 0.0
    # thresholds length = n-1 where n = len(precision)
    for i, thr in enumerate(np.append(thresholds, 1.0)):
        prec = precision[i + 1] if i + 1 < len(precision) else precision[-1]
        rec = recall[i + 1] if i + 1 < len(recall) else recall[-1]
        if rec >= target_recall and prec > best_prec:
            best_prec = prec
            best_thresh = thresholds[i] if i < len(thresholds) else 1.0
            best_rec = rec

    if best_thresh is not None:
        return float(best_thresh), {"mode": "target_recall", "precision": float(best_prec), "recall": float(best_rec)}

    # fallback: maximize F1
    f1_scores = []
    thr_list = list(thresholds) + [1.0]
    for i, thr in enumerate(thr_list):
        prec = precision[i + 1] if i + 1 < len(precision) else precision[-1]
        rec = recall[i + 1] if i + 1 < len(recall) else recall[-1]
        if prec + rec == 0:
            f1 = 0.0
        else:
            f1 = 2 * prec * rec / (prec + rec)
        f1_scores.append(f1)
    best_idx = int(np.argmax(f1_scores))
    chosen_thr = thr_list[best_idx]
    chosen_prec = precision[best_idx + 1] if best_idx + 1 < len(precision) else precision[-1]
    chosen_rec = recall[best_idx + 1] if best_idx + 1 < len(recall) else recall[-1]
    return float(chosen_thr), {"mode": "max_f1", "precision": float(chosen_prec), "recall": float(chosen_rec), "f1": float(f1_scores[best_idx])}


# -------------------------
# Training loop
# -------------------------
def train(args):
    # Force CPU-only device
    device = torch.device("cpu")
    print("Device: cpu (CPU-only mode)")

    label_csv = Path(args.label_csv)
    crops_root = Path(args.crops_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_transform = make_transforms(img_size=args.img_size, train=True)
    val_transform = make_transforms(img_size=args.img_size, train=False)

    train_ds = MouthFrameDataset(label_csv, crops_root, split="train", transform=train_transform)
    val_ds = MouthFrameDataset(label_csv, crops_root, split="val", transform=val_transform)

    # Build sampler to oversample positives
    train_labels = [s[2] for s in train_ds.samples]
    train_labels_arr = np.array(train_labels)
    class_counts = Counter(train_labels)
    print("Train class counts:", class_counts)
    pos = int(class_counts.get(1, 0))
    neg = int(class_counts.get(0, 0))
    ratio_pos_to_neg = pos / neg if neg > 0 else None
    if ratio_pos_to_neg is not None:
        print(f"Positive samples: {pos}  Negative samples: {neg}  pos/neg={ratio_pos_to_neg:.6f}  (≈1 positive per {neg/pos:.1f} negatives)")
    else:
        print(f"Positive samples: {pos}  Negative samples: {neg}")

    # choose pos_per_batch: at least 1, but not more than batch_size-1
    # using the increased heuristic: batch_size // 4
    pos_per_batch = max(1, min(args.batch_size - 1, max(1, args.batch_size // 4)))
    print(f"Using BalancedBatchSampler with pos_per_batch={pos_per_batch} (batch_size={args.batch_size})")

    # BalancedBatchSampler will oversample positives (with replacement) and sample negatives without replacement
    balanced_sampler = BalancedBatchSampler(train_labels_arr, batch_size=args.batch_size, pos_per_batch=pos_per_batch)

    # Use batch_sampler argument (do not pass sampler or shuffle)
    train_loader = DataLoader(train_ds, batch_sampler=balanced_sampler,
                              num_workers=args.num_workers, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_batch)

    print("DataLoader created. train_len=", len(train_ds), "val_len=", len(val_ds))

    # prepare per-100-batch average train loss CSV (open once, close after training)
    train_loss_avg_csv = out_dir / "train_loss_avg_per_100.csv"
    train_loss_avg_fh = train_loss_avg_csv.open("w", newline="")
    train_loss_avg_writer = csv.writer(train_loss_avg_fh)
    train_loss_avg_writer.writerow(["epoch", "window_start_batch", "window_end_batch", "avg_loss", "samples_in_window"])
    train_loss_avg_fh.flush()

    model = build_model(backbone=args.backbone, pretrained=True).to(device)

    # Moderate pos_weight (suggested baseline). This is intentionally a moderate fixed value
    # to avoid extreme loss amplification when positives are oversampled.
    pos_weight = 3.0
    # If you prefer to use the computed heuristic but cap it, you could do:
    # pos_weight = min(compute_pos_weight(train_labels_arr), 10.0)
    print(f"Using pos_weight={pos_weight:.4f} for BCEWithLogitsLoss")
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, verbose=True
        )
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )

    best_epoch = -1
    best_val_f1 = -1.0
    metrics_log = []

    for epoch in range(1, args.epochs + 1):
        print(f"Starting epoch {epoch}/{args.epochs}")
        model.train()
        running_loss = 0.0

        # initialize 100-batch window accumulators
        window_sum = 0.0
        window_count = 0
        window_start = 0

        # training loop
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()

            # gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

            # accumulate for 100-batch window
            window_sum += float(loss.item())
            window_count += 1

            # when window reaches 100 batches, write average
            if window_count >= 100:
                window_end = batch_idx
                avg_loss = window_sum / window_count
                train_loss_avg_writer.writerow([epoch, window_start, window_end, f"{avg_loss:.6f}", window_count])
                train_loss_avg_fh.flush()
                # reset window
                window_sum = 0.0
                window_count = 0
                window_start = batch_idx + 1

            # periodic progress print
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch} batch {batch_idx}/{len(train_loader)} imgs={imgs.shape} loss={loss.item():.4f}")

        # flush any remaining partial window at epoch end
        if window_count > 0:
            window_end = batch_idx
            avg_loss = window_sum / window_count
            train_loss_avg_writer.writerow([epoch, window_start, window_end, f"{avg_loss:.6f}", window_count])
            train_loss_avg_fh.flush()
            window_sum = 0.0
            window_count = 0
            window_start = 0

        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = model(imgs)
                all_logits.append(logits.cpu().numpy().ravel())
                all_labels.append(labels.cpu().numpy().ravel())
        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels).astype(int)
        probs = 1.0 / (1.0 + np.exp(-all_logits))

        # PR curve and threshold selection
        pr_png = out_dir / f"pr_epoch_{epoch:03d}.png"
        precision, recall, thresholds = save_pr_curve(all_labels, probs, pr_png, epoch)
        chosen_thr, info = select_threshold_for_target_recall(all_labels, probs, target_recall=args.target_recall)

        # compute metrics at chosen threshold
        preds = (probs >= chosen_thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(all_labels, preds, average="binary", zero_division=0)

        # scheduler step on val F1
        scheduler.step(f1)

        print(f"Epoch {epoch:03d}  train_loss={train_loss:.4f}  val_f1={f1:.4f}  val_prec={p:.4f}  val_rec={r:.4f}  thr={chosen_thr:.4f}  mode={info.get('mode')}")

        # save epoch metrics
        metrics_log.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_f1": float(f1),
            "val_precision": float(p),
            "val_recall": float(r),
            "chosen_threshold": float(chosen_thr),
            "threshold_mode": info.get("mode"),
            "threshold_info": info,
        })
        # write metrics CSV each epoch
        metrics_csv = out_dir / "val_metrics.csv"
        write_metrics_csv(metrics_csv, metrics_log)

        # save threshold info JSON
        thr_json = out_dir / f"threshold_epoch_{epoch:03d}.json"
        with thr_json.open("w") as fh:
            json.dump({"threshold": chosen_thr, "info": info}, fh, indent=2)

        # checkpoint every epoch
        ckpt = out_dir / f"ckpt_epoch_{epoch:03d}.pth"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_f1": f1,
            "chosen_threshold": chosen_thr,
            "threshold_info": info,
        }, ckpt)

        # update best
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_epoch = epoch
            best_ckpt = out_dir / "best.pth"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_f1": f1,
                "chosen_threshold": chosen_thr,
                "threshold_info": info,
            }, best_ckpt)
            print(f"Saved best checkpoint to {best_ckpt} (val_f1={f1:.4f})")

        # early stopping
        if epoch - best_epoch >= args.patience:
            print(f"No improvement for {args.patience} epochs (best epoch {best_epoch}), stopping.")
            break

    # final summary
    print("Training complete. Best epoch:", best_epoch, "best_val_f1:", best_val_f1)
    final_json = out_dir / "final_summary.json"
    with final_json.open("w") as fh:
        json.dump({"best_epoch": best_epoch, "best_val_f1": best_val_f1, "metrics": metrics_log}, fh, indent=2)

    # close averaged-loss CSV handle
    train_loss_avg_fh.close()

    return out_dir



def collate_batch(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    labels = torch.stack(labels).unsqueeze(1)
    return imgs, labels


def write_metrics_csv(path: Path, metrics_list):
    fieldnames = ["epoch", "train_loss", "val_f1", "val_precision", "val_recall", "chosen_threshold", "threshold_mode"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics_list:
            writer.writerow({
                "epoch": m["epoch"],
                "train_loss": m["train_loss"],
                "val_f1": m["val_f1"],
                "val_precision": m["val_precision"],
                "val_recall": m["val_recall"],
                "chosen_threshold": m["chosen_threshold"],
                "threshold_mode": m["threshold_mode"],
            })


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train framewise mouth event detector (baseline) - CPU only")
    p.add_argument("--label-csv", default="../../event_csvs/mouth_frame_label_table.csv", help="Path to mouth_frame_label_table.csv")
    p.add_argument("--crops-root", default="E:/VF ML Crops", help="Root folder containing per-video crop folders")
    p.add_argument("--out-dir", default="../../models/event_baseline", help="Output directory for models and metrics")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--device", default="cpu", help="Ignored; script runs CPU-only")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers (0 recommended for CPU-only)")
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--target-recall", type=float, default=0.995, help="Target recall for threshold selection (e.g., 0.995 for near-zero FN)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        out = train(args)
        print("Outputs saved to:", out)
    except Exception as e:
        print("Error during training:", str(e))
        raise

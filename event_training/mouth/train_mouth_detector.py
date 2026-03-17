#!/usr/bin/env python3
"""
event_training/mouth/train_mouth_detector.py

CPU-only framewise baseline training for swallow-onset detection.

Features:
- Reads labels from ../../event_csvs/mouth_frame_label_table.csv by default.
- Loads per-frame JPEGs (preferred) or .npy normalized tensors as fallback.
- Caches decoded JPEGs as .npy (HWC uint8) to avoid repeated JPEG decode (disabled by default in this variant).
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

"""
from pathlib import Path
import argparse
import csv
import json
import os
import random
import math
import sys
from typing import Iterator, List
from collections import Counter

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, auc

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
# Small helpers for safe saving
# -------------------------
def _safe_torch_save(obj, path: Path):
    """
    Atomically save a torch object to `path` by writing to a temporary file and renaming.
    Prints success or error so failures are visible in logs.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(obj, tmp)
        tmp.replace(path)
        print(f"Saved: {path}")
    except Exception as e:
        print(f"ERROR saving {path}: {type(e).__name__}: {e}")
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


# -------------------------
# Safe exists helper (used in __getitem__ to avoid a single bad stat from crashing)
# -------------------------
def _safe_exists(p: Path):
    try:
        return p.exists()
    except Exception as e:
        print(f"Warning: cannot stat path {p!s}: {type(e).__name__}: {e}", file=sys.stderr)
        return False


# -------------------------
# Dataset
# -------------------------
FRAME_NAME = "frame_{:06d}"


class MouthFrameDataset(Dataset):
    """
    Lazy-check variant:
    - During __init__, we only parse the CSV and store candidate paths (no expensive stat calls).
    - Existence checks are performed in __getitem__ using a safe wrapper to avoid startup stalls.
    - This reduces startup time for very large CSVs or slow filesystems.
    """
    def __init__(self, label_csv: Path, crops_root: Path, split: str, transform=None, prefer_jpeg=True, cache_enabled=False):
        self.crops_root = Path(crops_root)
        self.transform = transform
        self.prefer_jpeg = prefer_jpeg
        self.cache_enabled = bool(cache_enabled)
        self.split = split
        self.samples = []  # tuples: (video, frame_int, label, jpeg_path, npy_path)

        if not Path(label_csv).exists():
            raise FileNotFoundError(f"Label CSV not found: {label_csv}")

        # Lazy: store candidate paths without calling .exists() here to avoid long startup
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
                # store both candidate paths; existence will be checked lazily in __getitem__
                self.samples.append((vid, frm, lbl, jpeg_path, npy_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split={split} (checked {label_csv} and {crops_root})")

    def __len__(self):
        return len(self.samples)

    def _cache_path_for(self, vid: str, frm: int) -> Path:
        # cache under crops_root/.cache_tensors/<split>/<video>/mouth/frame_XXXXXX.npy
        cache_root = self.crops_root / ".cache_tensors" / self.split / vid / "mouth"
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root / f"{FRAME_NAME.format(frm)}.npy"

    def __getitem__(self, idx):
        vid, frm, lbl, jpeg_path, npy_path = self.samples[idx]

        # First, prefer precomputed normalized .npy if it actually exists
        if npy_path is not None and _safe_exists(npy_path):
            try:
                arr = np.load(npy_path)  # expected CHW or HWC float32
                t = torch.from_numpy(arr).float()
                # If HWC convert to CHW
                if t.ndim == 3 and t.shape[2] == 3:
                    t = t.permute(2, 0, 1)
                t = self._tensor_augment(t)
                return t, torch.tensor(lbl, dtype=torch.float32)
            except Exception as e:
                # If loading fails, warn and fall through to JPEG path
                print(f"Warning: failed to load npy {npy_path}: {type(e).__name__}: {e}", file=sys.stderr)

        # Otherwise handle JPEG path with optional caching of decoded RGB HWC uint8
        if jpeg_path is not None and _safe_exists(jpeg_path) and self.prefer_jpeg:
            cache_path = self._cache_path_for(vid, frm) if self.cache_enabled else None
            if cache_path is not None and _safe_exists(cache_path):
                try:
                    arr = np.load(cache_path)
                    img = Image.fromarray(arr)
                except Exception as e:
                    print(f"Warning: failed to load cache {cache_path}: {type(e).__name__}: {e}", file=sys.stderr)
                    img = None
            else:
                img = None

            if img is None:
                try:
                    img = Image.open(jpeg_path).convert("RGB")
                    if cache_path is not None:
                        try:
                            arr = np.array(img)
                            np.save(cache_path, arr, allow_pickle=False)
                        except Exception:
                            # caching is best-effort; ignore failures
                            pass
                except Exception as e:
                    # If JPEG decode fails, raise a controlled error so the training loop can handle it
                    raise RuntimeError(f"Failed to open JPEG {jpeg_path}: {type(e).__name__}: {e}")

            if self.transform is not None:
                img = self.transform(img)
            else:
                img = T.ToTensor()(img)
            return img, torch.tensor(lbl, dtype=torch.float32)

        # If neither file exists or both failed to load, raise a clear error for this sample
        raise RuntimeError(f"No image available for sample vid={vid} frame={frm} (checked jpeg={jpeg_path}, npy={npy_path})")

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


def select_threshold_for_target_recall_safe(y_true, y_scores, target_recall=0.995, min_threshold=1e-2):
    """
    Prefer thresholds that meet target_recall but avoid returning 0.0.
    If the best threshold < min_threshold or no threshold meets target_recall,
    fall back to threshold that maximizes F1 (with thr >= min_threshold if possible).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    best_thresh = None
    best_prec = -1.0
    best_rec = 0.0
    # thresholds correspond to precision[1:], recall[1:]
    thr_array = np.append(thresholds, 1.0)
    for i, thr_val in enumerate(thr_array):
        prec = precision[i + 1] if i + 1 < len(precision) else precision[-1]
        rec = recall[i + 1] if i + 1 < len(recall) else recall[-1]
        if rec >= target_recall and prec > best_prec and thr_val >= min_threshold:
            best_prec = prec
            best_thresh = float(thr_val)
            best_rec = rec

    if best_thresh is not None:
        return best_thresh, {"mode": "target_recall", "precision": float(best_prec), "recall": float(best_rec)}

    # fallback: maximize F1 but prefer thresholds >= min_threshold
    thr_list = list(thresholds) + [1.0]
    f1_scores = []
    for i, thr in enumerate(thr_list):
        prec = precision[i + 1] if i + 1 < len(precision) else precision[-1]
        rec = recall[i + 1] if i + 1 < len(recall) else recall[-1]
        if prec + rec == 0:
            f1 = 0.0
        else:
            f1 = 2 * prec * rec / (prec + rec)
        f1_scores.append((f1, thr))

    # prefer best F1 with thr >= min_threshold
    f1_scores_sorted = sorted(f1_scores, key=lambda x: x[0], reverse=True)
    for f1, thr in f1_scores_sorted:
        if thr >= min_threshold:
            preds = (y_scores >= thr).astype(int)
            p, r, f1_val, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
            return float(thr), {"mode": "max_f1", "precision": float(p), "recall": float(r), "f1": float(f1_val)}

    # last resort: return the absolute best F1 even if thr < min_threshold
    best_f1, best_thr = max(f1_scores, key=lambda x: x[0])
    preds = (y_scores >= best_thr).astype(int)
    p, r, f1_val, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    return float(best_thr), {"mode": "max_f1_unconstrained", "precision": float(p), "recall": float(r), "f1": float(f1_val)}


def evaluate_on_balanced_subset(model, val_loader, device, max_neg_per_pos=5):
    """
    Build a balanced validation set by sampling up to max_neg_per_pos negatives per positive.
    Returns: all_labels, all_probs (numpy arrays)
    """
    model.eval()
    pos_samples = []
    neg_samples = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            labs = labels.cpu().numpy().ravel().astype(int)
            for p, l in zip(probs, labs):
                if l == 1:
                    pos_samples.append((p, l))
                else:
                    neg_samples.append((p, l))

    if len(pos_samples) == 0:
        # fallback to full validation if no positives
        combined = pos_samples + neg_samples
        all_probs = np.array([x[0] for x in combined])
        all_labels = np.array([x[1] for x in combined])
        return all_labels, all_probs

    # sample negatives to match positives up to max_neg_per_pos
    n_pos = len(pos_samples)
    n_neg_target = min(len(neg_samples), n_pos * max_neg_per_pos)
    neg_sampled = random.sample(neg_samples, n_neg_target) if n_neg_target < len(neg_samples) else neg_samples

    combined = pos_samples + neg_sampled
    random.shuffle(combined)
    all_probs = np.array([x[0] for x in combined])
    all_labels = np.array([x[1] for x in combined])
    return all_labels, all_probs


# -------------------------
# Training loop
# -------------------------
def train(args):
    # Force CPU-only device
    device = torch.device("cpu")
    print("Device: cpu (CPU-only mode)")

    # Limit PyTorch thread usage to avoid BLAS contention on multi-core CPUs
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    label_csv = Path(args.label_csv)
    crops_root = Path(args.crops_root)
    out_dir = Path(args.out_dir)
    out_dir = out_dir.expanduser().resolve()
    print("Outputs will be written to:", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_transform = make_transforms(img_size=args.img_size, train=True)
    val_transform = make_transforms(img_size=args.img_size, train=False)

    # Enable caching in dataset is disabled here for faster startup; set cache_enabled=True if you want caching.
    train_ds = MouthFrameDataset(label_csv, crops_root, split="train", transform=train_transform, prefer_jpeg=True, cache_enabled=False)
    val_ds = MouthFrameDataset(label_csv, crops_root, split="val", transform=val_transform, prefer_jpeg=True, cache_enabled=False)

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

    # choose pos_per_batch: conservative default to reduce overfitting to positives
    # cap at 2 positives per batch for batch_size >= 16 (conservative)
    pos_per_batch = max(1, min(2, max(1, args.batch_size // 32)))
    print(f"Using BalancedBatchSampler with pos_per_batch={pos_per_batch} (batch_size={args.batch_size})")

    # BalancedBatchSampler will oversample positives (with replacement) and sample negatives without replacement
    balanced_sampler = BalancedBatchSampler(train_labels_arr, batch_size=args.batch_size, pos_per_batch=pos_per_batch)

    # DataLoader parallelism: if user passed num_workers <= 0, pick a modest default
    if args.num_workers and args.num_workers > 0:
        num_workers = args.num_workers
    else:
        # choose up to half of CPU cores but at least 1, cap at 8
        num_workers = max(1, min(8, max(1, (os.cpu_count() or 2) // 2)))

    # Use batch_sampler argument (do not pass sampler or shuffle)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=balanced_sampler,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_batch,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_batch,
    )

    print("DataLoader created. train_len=", len(train_ds), "val_len=", len(val_ds), "num_workers=", num_workers)

    # prepare per-100-batch average train loss CSV (open once, close after training)
    train_loss_avg_csv = out_dir / "train_loss_avg_per_100.csv"
    train_loss_avg_fh = train_loss_avg_csv.open("w", newline="")
    train_loss_avg_writer = csv.writer(train_loss_avg_fh)
    train_loss_avg_writer.writerow(["epoch", "window_start_batch", "window_end_batch", "avg_loss", "samples_in_window"])
    train_loss_avg_fh.flush()

    model = build_model(backbone=args.backbone, pretrained=True).to(device)

    # Conservative pos_weight to avoid extreme amplification early
    pos_weight = 1.0
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

        # Validation: evaluate on balanced subset for threshold selection and diagnostics
        all_labels_bal, probs_bal = evaluate_on_balanced_subset(model, val_loader, device, max_neg_per_pos=5)

        # Save validation logits/probs for later inspection
        np.save(out_dir / f"val_logits_epoch_{epoch:03d}.npy", {"labels": all_labels_bal, "probs": probs_bal}, allow_pickle=True)

        # PR curve and safer threshold selection on balanced subset
        pr_png = out_dir / f"pr_epoch_{epoch:03d}.png"
        precision, recall, thresholds = save_pr_curve(all_labels_bal, probs_bal, pr_png, epoch)
        chosen_thr, info = select_threshold_for_target_recall_safe(all_labels_bal, probs_bal, target_recall=args.target_recall, min_threshold=1e-2)

        # compute metrics at chosen threshold (balanced)
        preds_bal = (probs_bal >= chosen_thr).astype(int)
        p_bal, r_bal, f1_bal, _ = precision_recall_fscore_support(all_labels_bal, preds_bal, average="binary", zero_division=0)

        # Also compute metrics at thr=0.0 and thr=0.5 for diagnostics (on balanced subset)
        preds_thr0 = (probs_bal >= 0.0).astype(int)
        p0, r0, f10, _ = precision_recall_fscore_support(all_labels_bal, preds_thr0, average="binary", zero_division=0)
        preds_thr05 = (probs_bal >= 0.5).astype(int)
        p05, r05, f105, _ = precision_recall_fscore_support(all_labels_bal, preds_thr05, average="binary", zero_division=0)

        # For backward compatibility with existing logs, also compute metrics on full validation (unbalanced)
        all_logits_full = []
        all_labels_full = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = model(imgs)
                all_logits_full.append(logits.cpu().numpy().ravel())
                all_labels_full.append(labels.cpu().numpy().ravel())
        if len(all_logits_full) > 0:
            all_logits_full = np.concatenate(all_logits_full)
            all_labels_full = np.concatenate(all_labels_full).astype(int)
            probs_full = 1.0 / (1.0 + np.exp(-all_logits_full))
            preds_full = (probs_full >= chosen_thr).astype(int)
            p_full, r_full, f1_full, _ = precision_recall_fscore_support(all_labels_full, preds_full, average="binary", zero_division=0)
        else:
            p_full = r_full = f1_full = 0.0

        # scheduler step on balanced F1 (prefer balanced signal for LR scheduling)
        scheduler.step(f1_bal)

        print(
            f"Epoch {epoch:03d}  train_loss={train_loss:.6f}  "
            f"bal_f1={f1_bal:.4f} bal_prec={p_bal:.4f} bal_rec={r_bal:.4f} "
            f"thr={chosen_thr:.6f} mode={info.get('mode')}"
        )
        print(
            f"  diagnostics: thr0 -> f1={f10:.4f} prec={p0:.4f} rec={r0:.4f}; "
            f"thr0.5 -> f1={f105:.4f} prec={p05:.4f} rec={r05:.4f}; "
            f"full_unbalanced -> f1={f1_full:.4f} prec={p_full:.4f} rec={r_full:.4f}"
        )

        # save epoch metrics (include balanced and full metrics)
        metrics_log.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "bal_f1": float(f1_bal),
            "bal_precision": float(p_bal),
            "bal_recall": float(r_bal),
            "chosen_threshold": float(chosen_thr),
            "threshold_mode": info.get("mode"),
            "diag_thr0_f1": float(f10),
            "diag_thr05_f1": float(f105),
            "full_val_f1": float(f1_full),
            "full_val_precision": float(p_full),
            "full_val_recall": float(r_full),
        })
        # write metrics CSV each epoch (extended schema)
        metrics_csv = out_dir / "val_metrics.csv"
        fieldnames = [
            "epoch", "train_loss", "bal_f1", "bal_precision", "bal_recall",
            "chosen_threshold", "threshold_mode", "diag_thr0_f1", "diag_thr05_f1",
            "full_val_f1", "full_val_precision", "full_val_recall"
        ]
        with metrics_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in metrics_log:
                writer.writerow({
                    "epoch": m["epoch"],
                    "train_loss": m["train_loss"],
                    "bal_f1": m["bal_f1"],
                    "bal_precision": m["bal_precision"],
                    "bal_recall": m["bal_recall"],
                    "chosen_threshold": m["chosen_threshold"],
                    "threshold_mode": m["threshold_mode"],
                    "diag_thr0_f1": m["diag_thr0_f1"],
                    "diag_thr05_f1": m["diag_thr05_f1"],
                    "full_val_f1": m["full_val_f1"],
                    "full_val_precision": m["full_val_precision"],
                    "full_val_recall": m["full_val_recall"],
                })

        # save threshold info JSON
        thr_json = out_dir / f"threshold_epoch_{epoch:03d}.json"
        with thr_json.open("w") as fh:
            json.dump({"threshold": chosen_thr, "info": info}, fh, indent=2)

        # checkpoint every epoch (use atomic safe save)
        ckpt = out_dir / f"ckpt_epoch_{epoch:03d}.pth"
        _safe_torch_save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "bal_f1": f1_bal,
            "chosen_threshold": chosen_thr,
            "threshold_info": info,
        }, ckpt)

        # update best (use balanced F1 as primary)
        if f1_bal > best_val_f1:
            best_val_f1 = f1_bal
            best_epoch = epoch
            best_ckpt = out_dir / "best.pth"
            _safe_torch_save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "bal_f1": f1_bal,
                "chosen_threshold": chosen_thr,
                "threshold_info": info,
            }, best_ckpt)
            print(f"Saved best checkpoint to {best_ckpt} (bal_f1={f1_bal:.4f})")

        # early stopping (based on balanced F1)
        if epoch - best_epoch >= args.patience:
            print(f"No improvement for {args.patience} epochs (best epoch {best_epoch}), stopping.")
            break

    # final summary
    print("Training complete. Best epoch:", best_epoch, "best_bal_f1:", best_val_f1)
    final_json = out_dir / "final_summary.json"
    with final_json.open("w") as fh:
        json.dump({"best_epoch": best_epoch, "best_bal_f1": best_val_f1, "metrics": metrics_log}, fh, indent=2)

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
    p.add_argument("--label-csv", default="event_csvs/mouth_frame_label_table.csv", help="Path to mouth_frame_label_table.csv")
    p.add_argument("--crops-root", default="E:/VF ML Crops", help="Root folder containing per-video crop folders")
    p.add_argument("--out-dir", default="event_training/mouth/models/event_baseline", help="Output directory for models and metrics")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--device", default="cpu", help="Ignored; script runs CPU-only")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers (0 recommended for CPU-only). If 0, script will pick a modest default.")
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

#!/usr/bin/env python3
# event_training/training/train_mouth_event.py
"""
Training script for mouth onset classifier.

- Uses VideoBatchSampler to produce batches grouped by video.
- Shuffles video order each epoch to restore cross-video randomness at epoch level.
- CPU-only (no CUDA).
- DataLoader timeout logic compatible with num_workers.
- Config-driven: see your YAML for paths and hyperparameters.
"""
from pathlib import Path
import yaml
import random
import time
from datetime import datetime
import argparse
import traceback
import sys
import numpy as np
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler
from torchvision import models, transforms

from event_training.datasets.mouth_dataset import MouthOnsetDataset
from typing import Iterator, List, Sequence, Dict, Any

# -----------------------
# VideoBatchSampler
# -----------------------
class VideoBatchSampler(Sampler[List[int]]):
    """
    Yields batches of indices where each batch contains frames from a single video.
    Behavior:
      - At the start of each epoch (each __iter__), the list of videos is shuffled.
      - For each video, its indices are optionally shuffled, then yielded in batches of batch_size.
      - If a video's remaining indices are fewer than batch_size, a smaller final batch is yielded.
    """
    def __init__(self, video_to_indices: Dict[str, List[int]], batch_size: int, shuffle_within_video: bool = True, seed: int = 42):
        self.video_to_indices = {k: list(v) for k, v in video_to_indices.items()}
        self.batch_size = int(batch_size)
        self.shuffle_within_video = bool(shuffle_within_video)
        self.seed = int(seed)

        # stable list of video keys
        self.video_keys = list(self.video_to_indices.keys())

    def __len__(self) -> int:
        # approximate number of batches across all videos
        total = 0
        for inds in self.video_to_indices.values():
            total += (len(inds) + self.batch_size - 1) // self.batch_size
        return total

    def __iter__(self) -> Iterator[List[int]]:
        # epoch-level shuffle of videos
        rnd = random.Random(self.seed + int(time.time()))  # vary per epoch
        keys = list(self.video_keys)
        rnd.shuffle(keys)

        for key in keys:
            inds = list(self.video_to_indices[key])
            if self.shuffle_within_video:
                rnd.shuffle(inds)
            # yield batches from this video's indices
            for i in range(0, len(inds), self.batch_size):
                batch = inds[i : i + self.batch_size]
                yield batch

# -----------------------
# Utilities
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default="event_training/configs/mouth_event_config.yaml",
                   help="Path to YAML config")
    return p.parse_args()

def load_config(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def compute_metrics(labels, probs, thresh=0.5):
    preds = (np.array(probs) >= thresh).astype(int)
    labels = np.array(labels).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (preds == labels).mean()
    return {"precision": precision, "recall": recall, "f1": f1, "acc": acc, "tp": tp, "fp": fp, "fn": fn}

# -----------------------
# Model / training
# -----------------------
def build_transforms(cfg, train=True):
    size = int(cfg.get("input_size", 224))
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ColorJitter(
                brightness=cfg.get("augment", {}).get("brightness", 0.0),
                contrast=cfg.get("augment", {}).get("contrast", 0.0)
            ),
            transforms.RandomRotation(cfg.get("augment", {}).get("rotation", 0)),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

def build_model(cfg):
    name = cfg.get("model_name", "resnet18")
    pretrained = cfg.get("pretrained", True)
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model

def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy().ravel().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy().ravel().tolist())
    metrics = compute_metrics(all_labels, all_probs, thresh=0.5)
    return metrics, all_probs, all_labels

def train(cfg_path):
    cfg = load_config(cfg_path)

    # reproducibility
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # force CPU only
    device = torch.device("cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ensure no GPU usage

    # transforms and datasets
    train_tf = build_transforms(cfg, train=True)
    val_tf = build_transforms(cfg, train=False)

    ds_debug = bool(cfg.get("dataset_debug", False))

    print(f"[train] Loading datasets with debug={ds_debug}", flush=True)
    train_ds = MouthOnsetDataset(cfg["train_csv"], transform=train_tf, contrast_mode=cfg.get("contrast_mode","clahe"), debug=ds_debug)
    val_ds = MouthOnsetDataset(cfg["val_csv"], transform=val_tf, contrast_mode=cfg.get("contrast_mode","clahe"), debug=ds_debug)

    # DataLoader settings
    num_workers = int(cfg.get("num_workers", 0))
    batch_size = int(cfg.get("batch_size", 8))
    cfg_timeout = int(cfg.get("dataloader_timeout", 90))

    # DataLoader timeout must be 0 for single-process (num_workers == 0)
    if num_workers == 0:
        dl_timeout = 0
        print(f"[train] num_workers=0 -> forcing DataLoader timeout=0 (cfg requested {cfg_timeout})", flush=True)
    else:
        dl_timeout = cfg_timeout
        print(f"[train] DataLoader timeout set to {dl_timeout} seconds for num_workers={num_workers}", flush=True)

    print(f"[train] DataLoader settings: batch_size={batch_size} num_workers={num_workers} timeout={dl_timeout}", flush=True)

    # Build VideoBatchSampler for training so batches are grouped by video
    shuffle_within_video = bool(cfg.get("shuffle_within_video", True))
    sampler_seed = int(cfg.get("sampler_seed", seed))
    train_batch_sampler = VideoBatchSampler(train_ds.video_to_indices, batch_size=batch_size, shuffle_within_video=shuffle_within_video, seed=sampler_seed)

    # DataLoaders: pass batch_sampler for train to ensure batches are single-video
    train_loader = DataLoader(train_ds, batch_sampler=train_batch_sampler, num_workers=num_workers, pin_memory=False, timeout=dl_timeout)
    # For validation we can use a simple sequential batcher (no grouping required)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False, timeout=dl_timeout)

    model = build_model(cfg).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 1e-4)), weight_decay=float(cfg.get("weight_decay", 0.0)))

    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    history = []

    epochs = int(cfg.get("epochs", 1))
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0
        print(f"[train] Epoch {epoch}/{epochs} starting. ({datetime.now().isoformat()})", flush=True)
        batch_idx = 0
        try:
            for imgs, labels in train_loader:
                batch_idx += 1
                if batch_idx % 10 == 0:
                    print(f"[train] epoch {epoch} batch {batch_idx}", flush=True)
                imgs = imgs.to(device)
                labels = labels.to(device).view(-1, 1)

                logits = model(imgs)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * imgs.size(0)
                n_samples += imgs.size(0)
        except Exception as e:
            print("[train] Exception while iterating train_loader:", flush=True)
            traceback.print_exc(file=sys.stdout)
            print("[train] Aborting training loop due to DataLoader/worker error.", flush=True)
            raise

        train_loss = running_loss / max(1, n_samples)

        # validation
        try:
            val_metrics, val_probs, val_labels = evaluate(model, val_loader, device)
        except Exception as e:
            print("[train] Exception during evaluation:", flush=True)
            traceback.print_exc(file=sys.stdout)
            raise

        val_f1 = val_metrics["f1"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[train] Epoch {epoch:02d}/{epochs}  train_loss={train_loss:.4f}  val_f1={val_f1:.4f}  val_acc={val_metrics['acc']:.4f}  val_prec={val_metrics['precision']:.4f}  val_rec={val_metrics['recall']:.4f}", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_f1": val_f1,
            "val_acc": val_metrics["acc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        })

        # save checkpoint if improved
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt_path = ckpt_dir / f"best_mouth_event_{timestamp}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cfg": cfg,
                "epoch": epoch,
                "val_f1": val_f1,
            }, ckpt_path)
            print(f"[train] Saved best checkpoint: {ckpt_path}", flush=True)

    total_time = time.time() - start_time
    print(f"[train] Training complete. Best val F1: {best_val_f1:.4f}. Time elapsed: {total_time/60:.2f} minutes.", flush=True)

    # save history CSV
    import csv
    if len(history) > 0:
        hist_path = log_dir / f"train_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with hist_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        print(f"[train] Saved training history to: {hist_path}", flush=True)

if __name__ == "__main__":
    args = parse_args()
    train(args.config)

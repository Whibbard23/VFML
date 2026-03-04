#!/usr/bin/env python3
# event_training/training/train_mouth_model.py
"""
Training script (self-contained) for early-fusion mouth onset model.

Example (debug):
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"
python -m event_training.training.train_mouth_model `
  --csv event_csvs/mouth_crops_labels_with_split.csv `
  --epochs 2 `
  --batch-size 4 `
  --num-workers 0 `
  --out-dir runs/train_mouth_debug `
  --lr 1e-4 `
  --seed 42 `
  --verbose `
  --log-interval 10

"""
from pathlib import Path
import csv
import random
import time
import os
import logging
import faulthandler

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from packaging import version
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

# -------------------------
# Dataset: MouthDataset
# -------------------------
class MouthDataset(Dataset):
    """
    CSV rows: video, frame, label, split
    Expects:
      <frames_root>/<stem>_roi/crops/crop_{frame:06d}.jpg
      <frames_root>/<stem>_roi/labels/motion/motion_{frame:06d}.npy

    Returns: (input_tensor, label_tensor)
      input_tensor: torch.FloatTensor shape (4, H, W) -> RGB (3) + motion (1)
      label_tensor: torch.FloatTensor scalar (0.0 or 1.0)
    """
    def __init__(self, csv_path, frames_root="runs/inference", resize=224, split="train", train=True):
        self.csv_path = Path(csv_path)
        self.frames_root = Path(frames_root)
        self.resize = int(resize)
        self.split = split
        self.train = bool(train)

        self.rows = []
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
            rdr = csv.DictReader(fh)
            for r in rdr:
                row_split = (r.get("split") or "").strip().lower()
                if row_split != self.split:
                    continue
                video = (r.get("video") or "").strip()
                if not video:
                    continue
                frame = int(float(r.get("frame") or 0))
                label = int(float(r.get("label") or 0))
                self.rows.append({"video": video, "frame": frame, "label": label})

        self.geo_transform, self.color_transform = self._default_joint_transforms(resize=self.resize, train=self.train)
        self.to_tensor = transforms.ToTensor()
        self.rgb_norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    def _default_joint_transforms(self, resize=224, train=True):
        if train:
            geo = transforms.RandomResizedCrop(resize, scale=(0.9, 1.0))
            color = transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.02, hue=0.01)
            return geo, color
        else:
            geo = transforms.Resize((resize, resize))
            return geo, None

    def __len__(self):
        return len(self.rows)

    def _resolve_paths(self, video, frame_idx):
        stem = Path(video).stem
        crop_path = self.frames_root / f"{stem}_roi" / "crops" / f"crop_{frame_idx:06d}.jpg"
        motion_path = self.frames_root / f"{stem}_roi" / "labels" / "motion" / f"motion_{frame_idx:06d}.npy"
        return crop_path, motion_path

    def _load_rgb(self, p: Path):
        if not p.exists():
            return Image.fromarray(np.zeros((self.resize, self.resize, 3), dtype=np.uint8))
        return Image.open(p).convert("RGB")

    def _load_motion(self, p: Path):
        if not p.exists():
            return np.zeros((self.resize, self.resize), dtype=np.float32)
        m = np.load(p)
        if m.ndim == 2:
            return m.astype(np.float32)
        return np.squeeze(m).astype(np.float32)

    def __getitem__(self, idx):
        row = self.rows[idx]
        video = row["video"]
        frame_idx = row["frame"]
        label = float(row["label"])

        crop_path, motion_path = self._resolve_paths(video, frame_idx)
        rgb_pil = self._load_rgb(crop_path)
        motion_np = self._load_motion(motion_path)  # expected in [0,1]

        # convert motion to PIL for joint geometric transforms
        motion_pil = Image.fromarray((np.clip(motion_np, 0.0, 1.0) * 255).astype("uint8"))

        # apply identical geometric transform
        rgb_geo = self.geo_transform(rgb_pil)
        motion_geo = self.geo_transform(motion_pil)

        # photometric only on rgb
        if self.color_transform is not None:
            rgb_geo = self.color_transform(rgb_geo)

        # to tensors and normalize
        rgb_t = self.to_tensor(rgb_geo)
        rgb_t = self.rgb_norm(rgb_t)
        motion_t = self.to_tensor(motion_geo)  # 1xHxW

        input_t = torch.cat([rgb_t, motion_t], dim=0)  # (4,H,W)
        return input_t, torch.tensor(label, dtype=torch.float32)

# -------------------------
# Model: ResNet18EarlyFusion
# -------------------------
def _load_resnet18(pretrained: bool):
    """
    Load torchvision resnet18 using the new weights enum when available to avoid deprecation warnings.
    """
    tv = torchvision
    tv_version = getattr(tv, "__version__", "0.0.0")
    try:
        if version.parse(tv_version) >= version.parse("0.13"):
            try:
                from torchvision.models import ResNet18_Weights
                weights = ResNet18_Weights.DEFAULT if pretrained else None
                model = tv.models.resnet18(weights=weights)
                return model
            except Exception:
                pass
        model = tv.models.resnet18(pretrained=pretrained)
        return model
    except Exception:
        return tv.models.resnet18(pretrained=False)

class ResNet18EarlyFusion(nn.Module):
    """
    ResNet18 adapted to accept 4-channel input (RGB + motion).
    Final head outputs a single logit per sample.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = _load_resnet18(pretrained)
        orig_w = base.conv1.weight.data.clone()  # (64,3,7,7)
        conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            conv1.weight[:, :3, :, :] = orig_w
            conv1.weight[:, 3:4, :, :] = orig_w.mean(dim=1, keepdim=True)
        base.conv1 = conv1
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, 1)
        self.model = base

    def forward(self, x):
        logits = self.model(x).squeeze(1)
        return logits

# -------------------------
# Training utilities
# -------------------------
def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    return xs, ys

def evaluate(model, dl, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(y.cpu().numpy().tolist())
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, preds, average='binary', zero_division=0)
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1), "n": len(all_labels)}

# -------------------------
# Main training entrypoint
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--frames-root", default="runs/inference")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out-dir", default="runs/train_earlyfusion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load ImageNet pretrained weights")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose per-batch logging")
    parser.add_argument("--log-interval", type=int, default=50, help="Print status every N batches")
    parser.add_argument("--heartbeat-interval", type=int, default=30, help="Seconds between heartbeat file updates")
    # Patch additions
    parser.add_argument("--pos-weight-multiplier", type=float, default=1.0, help="Multiply computed pos_weight by this factor")
    parser.add_argument("--oversample-positives", action="store_true", help="Use WeightedRandomSampler to oversample positives")
    args = parser.parse_args()

    # deterministic-ish
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # prepare out dir and logging
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler()
        ],
    )
    logger = logging.getLogger("train")

    # heartbeat and faulthandler
    heartbeat_path = out_dir / "heartbeat.txt"
    last_heartbeat = time.time()
    faulthandler.enable()

    # datasets
    ds_train = MouthDataset(args.csv, frames_root=args.frames_root, split=args.train_split, train=True)
    ds_val = MouthDataset(args.csv, frames_root=args.frames_root, split=args.val_split, train=False)

    # compute pos_weight for BCEWithLogitsLoss (apply multiplier)
    labels = [r["label"] for r in ds_train.rows]
    pos = sum(labels); neg = len(labels) - pos
    base_pw = (neg / (pos + 1e-6))
    pos_weight = torch.tensor([base_pw * args.pos_weight_multiplier], dtype=torch.float32)

    # create training DataLoader (optionally oversample positives)
    if args.oversample_positives:
        from torch.utils.data import WeightedRandomSampler
        class_counts = [sum(1 for l in labels if l == c) for c in (0,1)]
        weights = [ (1.0 / class_counts[int(l)]) for l in labels ]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn)
    else:
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)

    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    device = torch.device("cpu")
    use_pretrained = not args.no_pretrained
    model = ResNet18EarlyFusion(pretrained=use_pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history_csv = out_dir / "history.csv"
    with history_csv.open("w", newline="") as hf:
        writer = csv.writer(hf)
        writer.writerow(["epoch","train_loss","val_precision","val_recall","val_f1","val_n","time_s"])

    best_f1 = -1.0
    for epoch in range(1, args.epochs+1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{args.epochs}", unit="batch", leave=False)
        for batch_idx, (x, y) in enumerate(pbar, start=1):
            batch_start = time.time()
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)

            try:
                loss.backward()
                opt.step()
            except Exception as e:
                logger.exception("Exception during backward/step")
                raise

            running_loss += float(loss.item())
            n_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            if args.verbose and (batch_idx % args.log_interval == 0):
                logger.info(f"Epoch {epoch} batch {batch_idx} loss={loss.item():.6f} batch_time={time.time()-batch_start:.2f}s")
                for h in logger.handlers:
                    try:
                        h.flush()
                    except Exception:
                        pass

            # heartbeat update
            if time.time() - last_heartbeat > args.heartbeat_interval:
                try:
                    heartbeat_path.write_text(f"alive {time.time()}\n")
                except Exception:
                    pass
                last_heartbeat = time.time()

        pbar.close()
        train_loss = running_loss / max(1, n_batches)
        val_metrics = evaluate(model, dl_val, device)
        epoch_elapsed = time.time() - epoch_start

        with history_csv.open("a", newline="") as hf:
            writer = csv.writer(hf)
            writer.writerow([epoch, train_loss, val_metrics["precision"], val_metrics["recall"], val_metrics["f1"], val_metrics["n"], round(epoch_elapsed,1)])

        logger.info(f"Epoch {epoch}/{args.epochs} loss={train_loss:.4f} val_f1={val_metrics['f1']:.4f} prec={val_metrics['precision']:.4f} rec={val_metrics['recall']:.4f} time={epoch_elapsed:.1f}s")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_f1": best_f1}, out_dir / "best.pth")
        torch.save({"model_state": model.state_dict(), "epoch": epoch}, out_dir / "last.pth")

    logger.info("Training complete. Best val_f1: %.4f", best_f1)

if __name__ == "__main__":
    import argparse  # re-import for top-level run
    main()

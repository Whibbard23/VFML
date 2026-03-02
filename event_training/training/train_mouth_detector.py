# event_training/training/train_mouth_detector.py
"""
Train the mouth detector with a tqdm progress bar, periodic debug image grids,
and validation after every epoch.

Behavior:
- Validates on the provided validation CSV after each epoch.
- Saves per-epoch validation CSV to event_training/training/mouth/val_epoch{epoch:03d}.csv
- Saves debug image grids to event_training/training/mouth/
- Uses torchvision ResNet18 weights enum when available; falls back to weights=None.
"""

from __future__ import annotations
import argparse
import os
import time
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from tqdm import tqdm

try:
    from event_training.datasets.mouth_detector_dataset import MouthDetectorDataset
except Exception as e:
    raise ImportError("Could not import MouthDetectorDataset from event_training.datasets.") from e

MODEL_FALLBACK = False
try:
    from event_training.models.mouth_detector import MouthDetector  # type: ignore
except Exception:
    MODEL_FALLBACK = True


def _adapt_model_head_to_num_classes(model: torch.nn.Module, num_classes: int) -> torch.nn.Module:
    try:
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            return model
        if hasattr(model, "classifier"):
            cls = model.classifier
            if isinstance(cls, nn.Linear):
                in_features = cls.in_features
                model.classifier = nn.Linear(in_features, num_classes)
                return model
            if isinstance(cls, nn.Sequential) and len(cls) > 0:
                for i in reversed(range(len(cls))):
                    if isinstance(cls[i], nn.Linear):
                        in_features = cls[i].in_features
                        cls[i] = nn.Linear(in_features, num_classes)
                        model.classifier = cls
                        return model
        if hasattr(model, "head") and isinstance(model.head, nn.Linear):
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
            return model
    except Exception:
        pass
    return model


def _build_resnet18_fallback(num_classes: int) -> torch.nn.Module:
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1
        res = resnet18(weights=weights)
    except Exception:
        from torchvision.models import resnet18
        res = resnet18(weights=None)
    res.fc = nn.Linear(res.fc.in_features, num_classes)
    return res


def build_model(device: torch.device, num_classes: int = 1) -> torch.nn.Module:
    if not MODEL_FALLBACK:
        try:
            model = MouthDetector(num_classes=num_classes)  # type: ignore
        except TypeError:
            model = MouthDetector()  # type: ignore
            model = _adapt_model_head_to_num_classes(model, num_classes)
    else:
        model = _build_resnet18_fallback(num_classes)
    model.to(device)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train mouth detector")
    parser.add_argument("--csv", default="event_csvs/assembly_1_train_events.csv", help="path to events csv (training split)")
    parser.add_argument("--val-csv", default="event_csvs/assembly_1_val_events.csv", help="path to events csv (validation split)")
    parser.add_argument("--data-root", default=".", help="root folder for videos and crops")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--clip-len", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ckpt-out", default="event_training/training/mouth/mouth_detector.pth")
    parser.add_argument("--print-every", type=int, default=20, help="print batch stats every N batches")
    parser.add_argument("--debug-samples", type=int, default=4, help="number of samples to save as debug grids each epoch")
    return parser.parse_args()


def save_debug_grids(clips: torch.Tensor, out_dir: str, epoch: int, max_samples: int = 4):
    os.makedirs(out_dir, exist_ok=True)
    B = clips.shape[0]
    n = min(B, max_samples)
    for i in range(n):
        sample = clips[i].cpu()  # [T, C, H, W]
        try:
            grid = make_grid(sample, nrow=sample.shape[0], normalize=True, scale_each=True)
            fname = os.path.join(out_dir, f"epoch{epoch:03d}_sample{i:02d}.png")
            save_image(grid, fname)
        except Exception:
            for t in range(sample.shape[0]):
                frame = sample[t]
                fname = os.path.join(out_dir, f"epoch{epoch:03d}_sample{i:02d}_f{t:02d}.png")
                try:
                    save_image(frame, fname)
                except Exception:
                    pass


def train_epoch(model: torch.nn.Module, dl: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device,
                epoch: int, print_every: int = 20):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    start_time = time.time()

    pbar = tqdm(enumerate(dl, 1), total=len(dl), desc=f"Epoch {epoch}", ncols=120)
    for i, batch in pbar:
        clips, labels, metas = batch
        if clips.dim() == 4:
            clips = clips.unsqueeze(0)
        clips = clips.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        try:
            out = model(clips)
        except Exception:
            out = model(clips.permute(0, 2, 1, 3, 4))
        logits = out.view(-1)
        loss = criterion(logits, labels.view(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        avg_loss = running_loss / i

        if i % print_every == 0 or i == len(dl):
            elapsed = time.time() - start_time
            pbar.set_postfix({'avg_loss': f"{avg_loss:.4f}", 'last_loss': f"{loss.item():.4f}", 'elapsed_s': f"{elapsed:.1f}"})
            print(f"Epoch {epoch} | batch {i}/{len(dl)} | avg_loss {avg_loss:.4f} | last_loss {loss.item():.4f} | elapsed {elapsed:.1f}s", flush=True)

    pbar.close()
    return running_loss / max(1, len(dl))


def validate(model: torch.nn.Module, val_dl: DataLoader, device: torch.device, debug_dir: str, epoch: int, threshold: float = 0.5):
    """
    Validate model on val_dl and return (avg_loss, acc, precision, recall, f1, auc).

    - Writes per-sample CSV to {debug_dir}/val_epoch{epoch:03d}.csv
    - Appends per-epoch metrics (including AUC) to {debug_dir}/val_metrics.csv
    - Robust to different batch formats and tensor shapes
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    total_loss = 0.0
    total_samples = 0
    rows = []

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_dl, desc=f"Validate {epoch}", ncols=120):
            # unpack batch robustly
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                clips, labels = batch[0], batch[1]
                metas = batch[2] if len(batch) > 2 else None
            else:
                raise RuntimeError("Unexpected validation batch format")

            if clips.dim() == 4:
                clips = clips.unsqueeze(0)
            clips = clips.to(device)

            # ensure labels is a 1-D tensor of scalars on CPU
            labels = labels.view(-1).float().cpu()

            try:
                out = model(clips)
            except Exception:
                out = model(clips.permute(0, 2, 1, 3, 4))

            logits = out.view(-1).detach().cpu().float()
            # compute loss on device-safe tensors
            loss = criterion(logits.to(device), labels.to(device))
            probs = torch.sigmoid(logits)

            batch_size = probs.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            for i in range(batch_size):
                prob = float(probs[i])
                lab = float(labels[i])
                pred = 1.0 if prob > threshold else 0.0

                if pred == 1.0 and lab == 1.0:
                    tp += 1
                elif pred == 1.0 and lab == 0.0:
                    fp += 1
                elif pred == 0.0 and lab == 1.0:
                    fn += 1
                else:
                    tn += 1

                # robust meta extraction
                meta = None
                if isinstance(metas, (list, tuple)):
                    try:
                        meta = metas[i]
                    except Exception:
                        meta = None
                elif isinstance(metas, dict):
                    meta = metas

                video = ""
                event_frame_index = -1
                if isinstance(meta, dict):
                    video = meta.get("video", "")
                    try:
                        event_frame_index = int(meta.get("event_frame_index", -1))
                    except Exception:
                        event_frame_index = -1

                rows.append({"video": video, "event_frame_index": event_frame_index, "prob": prob, "label": lab, "pred": pred})
                all_probs.append(prob)
                all_labels.append(lab)

    avg_loss = total_loss / max(1, total_samples)
    acc = float((tp + tn) / max(1, total_samples))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # compute AUC: try sklearn first, fallback to numpy rank method
    auc = 0.0
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(all_labels)) > 1:
            auc = float(roc_auc_score(all_labels, all_probs))
        else:
            auc = float(0.5)  # undefined when only one class present; use 0.5 neutral
    except Exception:
        # numpy-based AUC via rank (Mann-Whitney U)
        import numpy as _np
        y = _np.asarray(all_labels, dtype=float)
        p = _np.asarray(all_probs, dtype=float)
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if n_pos == 0 or n_neg == 0:
            auc = 0.5
        else:
            # compute ranks with average for ties
            order = _np.argsort(p)
            ranks = _np.empty_like(order, dtype=float)
            # assign average ranks for ties
            sorted_p = p[order]
            i = 0
            N = len(sorted_p)
            while i < N:
                j = i + 1
                while j < N and sorted_p[j] == sorted_p[i]:
                    j += 1
                avg_rank = 0.5 * (i + 1 + j)  # 1-based ranks
                ranks[order[i:j]] = avg_rank
                i = j
            # sum ranks for positive class
            sum_ranks_pos = ranks[y == 1].sum()
            auc = float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    # save per-sample CSV
    os.makedirs(debug_dir, exist_ok=True)
    import pandas as pd
    val_csv = os.path.join(debug_dir, f"val_epoch{epoch:03d}.csv")
    try:
        pd.DataFrame(rows).to_csv(val_csv, index=False)
    except Exception:
        pass

    # append per-epoch metrics to a summary CSV (includes AUC)
    metrics_csv = os.path.join(debug_dir, "val_metrics.csv")
    metrics_row = {
        "epoch": epoch,
        "avg_loss": avg_loss,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "n_samples": total_samples
    }
    try:
        if not os.path.exists(metrics_csv):
            pd.DataFrame([metrics_row]).to_csv(metrics_csv, index=False)
        else:
            pd.DataFrame([metrics_row]).to_csv(metrics_csv, mode="a", header=False, index=False)
    except Exception:
        pass

    print(
        f"Validation epoch {epoch} - avg_loss: {avg_loss:.4f} - acc: {acc:.4f} - "
        f"precision: {precision:.4f} - recall: {recall:.4f} - f1: {f1:.4f} - auc: {auc:.4f} - saved: {val_csv}",
        flush=True
    )

    return avg_loss, acc, precision, recall, f1, auc





def main():
    args = parse_args()
    device = torch.device(args.device)
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

    print(f"CSV: {args.csv}", flush=True)
    print(f"Val CSV: {args.val_csv}", flush=True)
    print(f"Data root: {args.data_root}", flush=True)
    print(f"Device: {device}", flush=True)

    ds = MouthDetectorDataset(csv_path=args.csv, data_root=args.data_root, clip_len=args.clip_len, transform=transform)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_ds = MouthDetectorDataset(csv_path=args.val_csv, data_root=args.data_root, clip_len=args.clip_len, transform=transform)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Dataset size: {len(ds)} samples; val size: {len(val_ds)}; batch_size={args.batch_size}; clip_len={args.clip_len}", flush=True)

    model = build_model(device, num_classes=1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    debug_dir = os.path.join("event_training", "training", "mouth")
    os.makedirs(debug_dir, exist_ok=True)

    ckpt_dir = os.path.dirname(args.ckpt_out) or debug_dir
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        avg_loss = train_epoch(model, dl, optimizer, device, epoch, print_every=args.print_every)
        elapsed = time.time() - start

        # save checkpoint
        ckpt_path = args.ckpt_out
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, ckpt_path)
        print(f"Epoch {epoch}/{args.epochs} - avg_loss: {avg_loss:.4f} - time: {elapsed:.1f}s - checkpoint: {ckpt_path}", flush=True)

        # save debug grids from a single batch
        try:
            it = iter(dl)
            sample_clips, _, _ = next(it)
            if sample_clips.dim() == 4:
                sample_clips = sample_clips.unsqueeze(0)
            sample_clips = sample_clips.float().cpu()
            save_debug_grids(sample_clips, debug_dir, epoch, max_samples=args.debug_samples)
            print(f"Saved debug grids to {debug_dir}", flush=True)
        except Exception as e:
            print(f"Could not save debug grids: {e}", flush=True)

        # run validation after every epoch
        try:
            # free cache if using CUDA
            if device.type == "cuda":
                torch.cuda.empty_cache()
                
            val_avg_loss, val_acc, val_precision, val_recall, val_f1, val_auc = \
                validate(model, val_dl, device, debug_dir, epoch)

            print(
                f"Epoch {epoch} validation summary: loss={val_avg_loss:.4f}, acc={val_acc:.4f}, "
                f"precision={val_precision:.4f}, recall={val_recall:.4f}, f1={val_f1:.4f}, auc={val_auc:.4f}",
                flush=True
            )


        except Exception as e:
            print(f"Validation failed for epoch {epoch}: {e}", flush=True)

    print("Training complete. Final checkpoint saved to", args.ckpt_out, flush=True)


if __name__ == "__main__":
    main()

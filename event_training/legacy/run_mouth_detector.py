# event_training/inference/run_mouth_detector.py
"""
Run the trained mouth detector over a CSV of events/videos and produce candidate onsets.

Defaults:
- --csv defaults to the validation CSV: event_csvs/assembly_1_val_events.csv
- Clips are centered on the BEFORE_ONSET frame (CSV 'frame'); detection target is frame+1.
- Saves a CSV of predictions and optional debug crops for high-scoring candidates.
"""

from __future__ import annotations
import os
import argparse
import time
from typing import List, Dict, Any, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd

# Try to import the project's dataset and model; fall back to reasonable defaults if missing.
try:
    from event_training.legacy.mouth_dataset import MouthDetectorDataset
except Exception:
    raise ImportError("Could not import MouthDetectorDataset from event_training.datasets. Ensure package is on PYTHONPATH.")

MODEL_FALLBACK = False
try:
    from event_training.models.mouth_detector import MouthDetector  # type: ignore
except Exception:
    MODEL_FALLBACK = True


def temporal_nms(preds: List[Tuple[int, float]], window: int = 3) -> List[Tuple[int, float]]:
    if not preds:
        return []
    preds = sorted(preds, key=lambda x: x[0])
    out: List[Tuple[int, float]] = []
    cluster: List[Tuple[int, float]] = [preds[0]]
    for f, s in preds[1:]:
        if f - cluster[-1][0] <= window:
            cluster.append((f, s))
        else:
            out.append(max(cluster, key=lambda x: x[1]))
            cluster = [(f, s)]
    if cluster:
        out.append(max(cluster, key=lambda x: x[1]))
    return out


def _build_resnet18_fallback(num_classes: int) -> torch.nn.Module:
    """
    Construct a torchvision resnet18 using the modern weights API when available.
    If the weights enum is not available (older torchvision), fall back to weights=None to avoid deprecation warnings.
    """
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1
        res = resnet18(weights=weights)
    except Exception:
        from torchvision.models import resnet18
        res = resnet18(weights=None)
    res.fc = torch.nn.Linear(res.fc.in_features, num_classes)
    return res


def _adapt_model_head_to_num_classes(model: torch.nn.Module, num_classes: int) -> torch.nn.Module:
    try:
        if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, num_classes)
            return model
        if hasattr(model, "classifier"):
            cls = model.classifier
            if isinstance(cls, torch.nn.Linear):
                in_features = cls.in_features
                model.classifier = torch.nn.Linear(in_features, num_classes)
                return model
            if isinstance(cls, torch.nn.Sequential) and len(cls) > 0:
                for i in reversed(range(len(cls))):
                    if isinstance(cls[i], torch.nn.Linear):
                        in_features = cls[i].in_features
                        cls[i] = torch.nn.Linear(in_features, num_classes)
                        model.classifier = cls
                        return model
        if hasattr(model, "head") and isinstance(model.head, torch.nn.Linear):
            in_features = model.head.in_features
            model.head = torch.nn.Linear(in_features, num_classes)
            return model
    except Exception:
        pass
    return model


def build_model(device: torch.device, num_classes: int = 1):
    """
    Load model from repo if available; otherwise construct a simple backbone+head.
    Handles MouthDetector constructor differences.
    """
    if not MODEL_FALLBACK:
        try:
            model = MouthDetector(num_classes=num_classes)  # type: ignore
        except TypeError:
            model = MouthDetector()  # type: ignore
            model = _adapt_model_head_to_num_classes(model, num_classes)
    else:
        model = _build_resnet18_fallback(num_classes)
    model.to(device)
    model.eval()
    return model


def save_debug_clip(clip_tensor: torch.Tensor, out_dir: str, prefix: str, idx: int):
    from torchvision.utils import save_image
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{prefix}_{idx:05d}.png")
    save_image(clip_tensor, path, nrow=clip_tensor.shape[0])
    return path


def run_inference(
    csv_path: str,
    data_root: str,
    ckpt: str | None,
    batch_size: int = 8,
    clip_len: int = 5,
    threshold: float = 0.5,
    device_str: str = "cpu",
    num_workers: int = 0,
    debug_dir: str | None = "event_training/training/mouth",
):
    device = torch.device(device_str)
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

    ds = MouthDetectorDataset(csv_path=csv_path, data_root=data_root, clip_len=clip_len, transform=transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(device, num_classes=1)

    if ckpt:
        if os.path.exists(ckpt):
            state = torch.load(ckpt, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                sd = state["state_dict"]
            else:
                sd = state
            try:
                model.load_state_dict(sd, strict=False)
            except Exception:
                new_sd = {}
                for k, v in sd.items():
                    nk = k.replace("module.", "") if k.startswith("module.") else k
                    new_sd[nk] = v
                try:
                    model.load_state_dict(new_sd, strict=False)
                except Exception:
                    print("Warning: checkpoint could not be fully loaded; proceeding with model weights as-is.")
        else:
            print(f"Warning: checkpoint {ckpt} not found; running with current model weights.")

    results: List[Dict[str, Any]] = []
    os.makedirs(debug_dir, exist_ok=True) if debug_dir else None

    idx_global = 0
    start_time = time.time()
    with torch.no_grad():
        for batch in dl:
            clips, labels, metas = batch
            if clips.dim() == 4:
                clips = clips.unsqueeze(0)
            clips = clips.to(device)
            B, T, C, H, W = clips.shape
            try:
                out = model(clips)
            except Exception:
                try:
                    out = model(clips.permute(0, 2, 1, 3, 4))
                except Exception as e:
                    raise RuntimeError(f"Model forward failed for both input shapes: {e}")

            logits = out.view(-1)
            probs = torch.sigmoid(logits).cpu().numpy()

            for b in range(B):
                meta = metas[b] if isinstance(metas, (list, tuple)) else metas
                video = meta.get("video", "")
                center_idx = int(meta.get("center_idx", -1))
                event_frame_index = int(meta.get("event_frame_index", -1))
                before_onset = int(meta.get("before_onset", center_idx))
                score = float(probs[b])
                label = float(labels[b].item()) if hasattr(labels[b], "item") else float(labels[b])

                results.append({
                    "video": video,
                    "center_idx": center_idx,
                    "before_onset": before_onset,
                    "event_frame_index": event_frame_index,
                    "score": score,
                    "label": label,
                })

                if debug_dir and score >= threshold:
                    try:
                        clip_tensor = clips[b].cpu()
                        save_debug_clip(clip_tensor, debug_dir, Path(video).stem, idx_global)
                    except Exception:
                        pass

                idx_global += 1

    elapsed = time.time() - start_time
    print(f"Inference finished: {len(results)} samples, time {elapsed:.1f}s")

    df = pd.DataFrame(results)
    out_rows: List[Dict[str, Any]] = []
    for video, group in df.groupby("video"):
        preds = [(int(r["event_frame_index"]), float(r["score"])) for _, r in group.iterrows() if r["event_frame_index"] >= 0]
        nms = temporal_nms(preds, window=3)
        for f, s in nms:
            out_rows.append({"video": video, "pred_frame": int(f), "score": float(s)})

    out_df = pd.DataFrame(out_rows)
    out_csv = os.path.join(os.getcwd(), "mouth_detector_predictions.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv}")

    return out_df


def parse_args():
    parser = argparse.ArgumentParser(description="Run mouth detector inference")
    parser.add_argument("--csv", default="event_csvs/assembly_1_val_events.csv", help="path to events csv (validation split)")
    parser.add_argument("--data-root", default=".", help="root folder for videos and crops")
    parser.add_argument("--ckpt", default=None, help="path to model checkpoint (.pth)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--clip-len", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--device", default="cpu", help="torch device string, e.g., cpu or cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--debug-dir", default="event_training/training/mouth", help="directory to save high-score candidate crops")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"CSV: {args.csv}")
    print(f"Data root: {args.data_root}")
    print(f"Checkpoint: {args.ckpt}")
    run_inference(
        csv_path=args.csv,
        data_root=args.data_root,
        ckpt=args.ckpt,
        batch_size=args.batch_size,
        clip_len=args.clip_len,
        threshold=args.threshold,
        device_str=args.device,
        num_workers=args.num_workers,
        debug_dir=args.debug_dir,
    )


if __name__ == "__main__":
    main()

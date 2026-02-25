import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np

# add package path so we can import your modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "event_training"))

# try to import your data loader and training model factory
try:
    import data_loader
except Exception as e:
    raise RuntimeError(f"Failed to import data_loader from event_training: {e}")

# Try to import train module to reuse model definition if available
train_module = None
try:
    import event_training.legacy.train as train_module  # event_training/train.py
except Exception:
    train_module = None

# sklearn metrics (install if missing)
try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
except Exception:
    raise RuntimeError("scikit-learn is required. Install with: pip install scikit-learn")

def build_model_from_checkpoint(num_classes, device):
    """
    Attempt to construct the same model architecture used in training.
    Strategy:
      1. If train.py exposes a model class (SimpleCNN, Model, build_model, create_model), use it.
      2. Otherwise, fall back to a minimal SimpleCNN that matches common training scripts.
    If the fallback does not match the checkpoint shape, the script will raise an informative error.
    """
    # common names to try
    candidates = []
    if train_module is not None:
        for name in ("SimpleCNN", "Model", "build_model", "create_model", "get_model"):
            if hasattr(train_module, name):
                candidates.append(getattr(train_module, name))

    # if a callable was found, try to instantiate
    for cand in candidates:
        try:
            if callable(cand):
                # if it's a class, instantiate; if it's a factory, call it
                model = cand(num_classes) if isinstance(cand, type) else cand(num_classes)
                return model.to(device)
        except Exception:
            continue

    # fallback minimal CNN (3 conv layers)
    import torch.nn as nn
    class FallbackCNN(nn.Module):
        def __init__(self, n_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, n_classes)
            )
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    return FallbackCNN(num_classes).to(device)

def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt

def make_val_loader(val_csv, data_root, batch_size, num_workers, max_rows=None):
    # make_dataloader returns (dataloader, dataset) in your codebase
    dl, ds = data_loader.make_dataloader(val_csv, data_root=data_root, batch_size=batch_size, shuffle=False, num_workers=num_workers, max_rows=max_rows)
    return dl, ds

def evaluate(model, dl, device):
    model.eval()
    preds = []
    trues = []
    metas = []
    with torch.no_grad():
        for batch in dl:
            imgs, labels, meta = batch
            imgs = imgs.to(device)
            out = model(imgs)
            if out.dim() == 1:
                # single-output; convert to 2D
                out = out.unsqueeze(0)
            prob = torch.softmax(out, dim=1) if out.shape[1] > 1 else out
            pred = prob.argmax(dim=1).cpu().numpy()
            preds.append(pred)
            trues.append(labels.cpu().numpy())
            metas.extend(meta)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return preds, trues, metas

def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on validation set")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--val-csv", required=True, help="Validation CSV (same format used for training)")
    parser.add_argument("--data-root", default=".", help="Root path for crops/images")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None, help="Limit rows for quick checks")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # load checkpoint
    ckpt = load_checkpoint(args.ckpt, device)
    print("Loaded checkpoint keys:", list(ckpt.keys()))

    # get event_map from checkpoint or dataset
    event_map = ckpt.get("event_map") or ckpt.get("event_to_idx") or {}
    if not event_map:
        # instantiate dataset briefly to get mapping
        _, ds = data_loader.make_dataloader(args.val_csv, data_root=args.data_root, batch_size=1, shuffle=False, num_workers=0, max_rows=1)
        event_map = getattr(ds, "event_to_idx", {}) or {}
    if not event_map:
        raise RuntimeError("Could not determine event->index mapping from checkpoint or dataset. Ensure checkpoint contains 'event_map' or dataset provides 'event_to_idx'.")

    # invert mapping to get class names
    idx_to_event = {v: k for k, v in event_map.items()}
    num_classes = max(1, len(event_map))
    print("Detected classes:", num_classes)
    print("Event map sample:", list(event_map.items())[:10])

    # build model
    model = build_model_from_checkpoint(num_classes, device)
    # load model state if present
    model_state = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt.get("model")
    if model_state is None:
        raise RuntimeError("Checkpoint does not contain a recognizable model state (keys tried: 'model_state','state_dict','model').")
    try:
        model.load_state_dict(model_state)
    except Exception as e:
        raise RuntimeError(f"Failed to load state_dict into model. The model architecture may not match the checkpoint. Error: {e}")

    # dataloader
    val_dl, _ = make_val_loader(args.val_csv, args.data_root, batch_size=args.batch_size, num_workers=args.num_workers, max_rows=args.max_rows)

    # run evaluation
    preds, trues, metas = evaluate(model, val_dl, device)

    # metrics
    acc = accuracy_score(trues, preds)
    precision, recall, f1, support = precision_recall_fscore_support(trues, preds, average=None, zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(trues, preds, average="macro", zero_division=0)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(trues, preds, average="micro", zero_division=0)
    cm = confusion_matrix(trues, preds)

    print("\nEvaluation results")
    print("------------------")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro Precision: {macro_p:.4f}  Macro Recall: {macro_r:.4f}  Macro F1: {macro_f1:.4f}")
    print(f"Micro Precision: {micro_p:.4f}  Micro Recall: {micro_r:.4f}  Micro F1: {micro_f1:.4f}")
    print("\nPer-class metrics:")
    for i in range(num_classes):
        name = idx_to_event.get(i, str(i))
        print(f"  Class {i} ({name}): precision={precision[i]:.4f} recall={recall[i]:.4f} f1={f1[i]:.4f} support={support[i]}")

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)

    print("\nClassification report (sklearn):")
    print(classification_report(trues, preds, target_names=[idx_to_event.get(i, str(i)) for i in range(num_classes)], zero_division=0))

    # Optionally: print a few misclassified examples
    mis_idx = np.where(preds != trues)[0]
    print(f"\nTotal samples: {len(trues)}  Misclassified: {len(mis_idx)}")
    if len(mis_idx) > 0:
        print("Example misclassified entries (up to 10):")
        for i in mis_idx[:10]:
            print(f" idx={i} true={idx_to_event.get(int(trues[i]), trues[i])} pred={idx_to_event.get(int(preds[i]), preds[i])} meta={metas[i]}")

if __name__ == "__main__":
    main()

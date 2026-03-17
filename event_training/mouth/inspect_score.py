# debug_eval.py
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

# import helpers from your training script
from event_training.mouth.train_mouth_detector import (
    MouthFrameDataset,
    make_transforms,
    build_model,
    collate_batch,
)

from sklearn.metrics import precision_recall_fscore_support

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--label-csv", required=True)
    p.add_argument("--crops-root", required=True)
    p.add_argument("--ckpt", default=None, help="Optional checkpoint to load (pth)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cpu")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    val_transform = make_transforms(img_size=128, train=False)
    val_ds = MouthFrameDataset(args.label_csv, args.crops_root, split="val", transform=val_transform, prefer_jpeg=True, cache_enabled=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch)

    model = build_model(backbone="resnet18", pretrained=True).to(device)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ck["model_state"])
        print("Loaded checkpoint:", args.ckpt)

    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            logits = model(imgs)  # CPU
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            labs = labels.cpu().numpy().ravel().astype(int)
            all_probs.append(probs)
            all_labels.append(labs)

    if len(all_probs) == 0:
        print("No validation data found.")
        return

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    print("Validation positives:", int(all_labels.sum()), "Total:", len(all_labels))
    pctls = np.percentile(all_probs, [0,1,5,10,25,50,75,90,95,99,100])
    print("Probability percentiles [0,1,5,10,25,50,75,90,95,99,100]:")
    print(np.round(pctls, 6))
    print("Mean prob:", float(np.mean(all_probs)), "Std:", float(np.std(all_probs)))
    print("Fraction >= 0.0:", float((all_probs >= 0.0).mean()), "Fraction >= 0.5:", float((all_probs >= 0.5).mean()))

    for thr in [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5]:
        preds = (all_probs >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(all_labels, preds, average="binary", zero_division=0)
        print(f"thr={thr:.6f} -> prec={p:.4f} rec={r:.4f} f1={f1:.4f}")

if __name__ == "__main__":
    main()

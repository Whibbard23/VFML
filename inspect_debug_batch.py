# inspect_debug_batch.py
import sys
import traceback
from pathlib import Path
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

# ensure repo root is importable when running from repo root
repo_root = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(repo_root))

from event_training.mouth.train_mouth_detector import (
    MouthFrameDataset,
    make_transforms,
    build_model,
    collate_batch,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--label-csv", required=True)
    p.add_argument("--crops-root", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--batch-size", type=int, default=8)
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

    # iterate a few batches and inspect
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(val_loader):
            try:
                print(f"Batch {i}: imgs.shape={getattr(imgs,'shape',None)} imgs.dtype={getattr(imgs,'dtype',None)}")
                # print a few stats
                try:
                    print("  imgs.min(), imgs.max():", float(imgs.min()), float(imgs.max()))
                except Exception as e:
                    print("  (could not compute min/max):", e)
                print("  labels.shape=", getattr(labels,'shape',None), "labels.dtype=", getattr(labels,'dtype',None))
                # run forward
                logits = model(imgs)
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                print(f"  forward OK, logits shape {logits.shape}, sample probs (first 5):", probs[:5])
            except Exception:
                print("Exception during forward on batch", i)
                traceback.print_exc()
                # save the problematic batch for offline inspection
                try:
                    np.save("debug_imgs_batch.npy", imgs.numpy())
                    np.save("debug_labels_batch.npy", labels.numpy())
                    print("Saved debug_imgs_batch.npy and debug_labels_batch.npy in current folder")
                except Exception as e:
                    print("Failed to save debug batch:", e)
                break
            if i >= 4:
                print("Inspected 5 batches, exiting.")
                break

if __name__ == "__main__":
    main()

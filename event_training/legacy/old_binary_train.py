"""
Minimal training script for event crops.

Usage examples (from project root):
  # smoke test (quick)
  python event_training/train.py --train-csv event_csvs/assembly_1_train_events.csv --val-csv event_csvs/assembly_1_val_events.csv --data-root . --smoke

  # full train
  python event_training/train.py --train-csv event_csvs/assembly_1_train_events.csv --val-csv event_csvs/assembly_1_val_events.csv --data-root . --epochs 30 --batch-size 32 --out-dir event_training/experiments/assembly_1_run1
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from event_training.legacy.data_loader import make_dataloader, EVENT_TO_IDX
from event_training.legacy.utils import SimpleCNN, accuracy, save_checkpoint

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=False)
    p.add_argument("--data-root", default=".")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--out-dir", default="event_training/experiments/run")
    p.add_argument("--smoke", action="store_true", help="Run a short smoke test and exit")
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()

def train_epoch(model, device, loader, opt, criterion):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    for imgs, labels, meta in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        opt.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels.clamp(min=0))  # clamp to avoid negative labels causing issues
        loss.backward()
        opt.step()
        total_loss += loss.item()
        total_acc += accuracy(logits, labels)
        steps += 1
        if steps >= 10 and args.smoke:
            break
    return total_loss / max(1, steps), total_acc / max(1, steps)

def eval_epoch(model, device, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    with torch.no_grad():
        for imgs, labels, meta in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels.clamp(min=0))
            total_loss += loss.item()
            total_acc += accuracy(logits, labels)
            steps += 1
            if steps >= 10 and args.smoke:
                break
    return total_loss / max(1, steps), total_acc / max(1, steps)

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataloaders
    train_loader, train_ds = make_dataloader(args.train_csv, data_root=args.data_root, batch_size=args.batch_size, image_size=(128,128), shuffle=True, num_workers=args.num_workers, max_rows=(100 if args.smoke else None))
    val_loader, val_ds = (None, None)
    if args.val_csv:
        val_loader, val_ds = make_dataloader(args.val_csv, data_root=args.data_root, batch_size=args.batch_size, image_size=(128,128), shuffle=False, num_workers=args.num_workers, max_rows=(100 if args.smoke else None))

    num_classes = max(1, len(getattr(train_ds, "event_to_idx", {})))
    model = SimpleCNN(in_channels=3, num_classes=num_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}, classes: {num_classes}, train rows: {len(train_ds)}")
    if val_ds:
        print(f"Val rows: {len(val_ds)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, device, train_loader, opt, criterion)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}")
        if val_loader:
            val_loss, val_acc = eval_epoch(model, device, val_loader, criterion)
            print(f"             val_loss={val_loss:.4f}    val_acc={val_acc:.4f}")
        # checkpoint
        ckpt = {"epoch": epoch, "model_state": model.state_dict(), "opt_state": opt.state_dict(), "event_map": EVENT_TO_IDX}
        save_checkpoint(ckpt, out_dir, name=f"ckpt_epoch{epoch}.pth")
        if args.smoke:
            print("Smoke run complete; exiting.")
            break

    print("Training finished. Checkpoints saved to:", out_dir)

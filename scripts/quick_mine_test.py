from pathlib import Path
import time, csv
import torch, numpy as np
from event_training.training.train_mouth_model import MouthDataset, ResNet18EarlyFusion
from hard_negative_mine import CandDataset, collate_fn, load_model

CSV="event_csvs/mouth_crops_labels_with_split.csv"
FRAMES_ROOT="runs/inference"
CKPT="runs/train_mouth_1/best.pth"
DEVICE="cpu"
BATCH=16
MAX_ITEMS=100

# load candidate negatives (train split)
rows=[]
with open(CSV, "r", encoding="utf-8-sig", newline="") as fh:
    rdr = csv.DictReader(fh)
    for r in rdr:
        if (r.get("split") or "").strip().lower()!="train": continue
        if int(float(r.get("label") or 0))!=0: continue
        rows.append({"video": r["video"].strip(), "frame": int(float(r["frame"]))})
        if len(rows)>=MAX_ITEMS: break

print("Candidates loaded:", len(rows))
device = torch.device(DEVICE)
model = load_model(CKPT, device)
ds = CandDataset(rows, CSV, FRAMES_ROOT)
dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0, collate_fn=collate_fn)

total = 0
t0_all = time.time()
with torch.no_grad():
    for i, (x, _) in enumerate(dl, start=1):
        t0 = time.time()
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        t1 = time.time()
        print(f"batch {i} size {x.shape[0]} time {t1-t0:.3f}s")
        total += x.shape[0]
print("Scored total:", total, "elapsed:", time.time()-t0_all)

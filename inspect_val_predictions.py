# inspect_val_predictions.py
from pathlib import Path
import csv
import torch
import numpy as np
from event_training.training.train_mouth_model import MouthDataset, ResNet18EarlyFusion, evaluate
from torch.utils.data import DataLoader

CSV = "event_csvs/mouth_crops_labels_with_split.csv"
OUT = "runs/train_mouth_1"
DEVICE = "cpu"

# load val set
ds_val = MouthDataset(CSV, split="val", train=False)
dl_val = DataLoader(ds_val, batch_size=32, shuffle=False)

# load best model
ckpt = torch.load(Path(OUT)/"best.pth", map_location=DEVICE)
model = ResNet18EarlyFusion(pretrained=False)
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE)
model.eval()

all_logits = []
all_labels = []
all_rows = ds_val.rows

with torch.no_grad():
    for x,y in dl_val:
        x = x.to(DEVICE)
        logits = model(x).cpu().numpy()
        all_logits.extend(logits.tolist())
        all_labels.extend(y.numpy().tolist())

# threshold at 0.5
probs = 1/(1+np.exp(-np.array(all_logits)))
preds = (probs >= 0.5).astype(int)

# collect FP/FN
fp = []
fn = []
for row, p, y, prob in zip(all_rows, preds, all_labels, probs):
    if p==1 and y==0:
        fp.append((row["video"], row["frame"], prob))
    if p==0 and y==1:
        fn.append((row["video"], row["frame"], prob))

print("VAL SIZE:", len(all_rows))
print("FALSE POSITIVES:", len(fp))
print("FALSE NEGATIVES:", len(fn))

# show top 20 highest-confidence false positives
print("\nTop FP (prob desc):")
for v,f,p in sorted(fp, key=lambda x: -x[2])[:20]:
    print(v, f, f"{p:.3f}")

# show top 20 lowest-confidence true positives (borderline)
print("\nBorderline true positives (prob asc):")
tp = [(row["video"], row["frame"], prob) for row,p,y,prob in zip(all_rows,preds,all_labels,probs) if y==1]
for v,f,p in sorted(tp, key=lambda x: x[2])[:20]:
    print(v, f, f"{p:.3f}")

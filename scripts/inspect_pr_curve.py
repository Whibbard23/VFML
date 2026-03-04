import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score

probs = np.load("runs/train_mouth_2/inference/probs.npy").ravel()
labels = np.load("runs/train_mouth_2/inference/val_labels.npy").ravel().astype(int)

prec, rec, thr = precision_recall_curve(labels, probs)
# thr has length len(prec)-1
f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
best_idx = f1s.argmax()
print("Best F1 threshold:", thr[best_idx], "F1:", f1s[best_idx], "prec:", prec[best_idx], "rec:", rec[best_idx])

# thresholds that keep recall >= 0.95 but maximize precision
candidates = [(t,p,r) for t,p,r in zip(thr, prec[:-1], rec[:-1]) if r >= 0.95]
if candidates:
    best = max(candidates, key=lambda x: x[1])
    print("Best precision with recall>=0.95:", best)
else:
    print("No threshold keeps recall >= 0.95")

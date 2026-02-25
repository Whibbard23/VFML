# debug_batches.py
import sys
from event_training.data_loader import make_dataloader
dl, ds = make_dataloader("event_csvs/assembly_1_train_events.csv", data_root=".", batch_size=8, num_workers=0, max_rows=200)
for bi, batch in enumerate(dl):
    print("batch", bi, "type:", type(batch))
    imgs, labels, metas = batch
    print(" imgs:", type(imgs), getattr(imgs, "shape", None))
    print(" labels:", type(labels), getattr(labels, "shape", None))
    print(" metas type:", type(metas), "len:", len(metas))
    print(" meta sample:", metas[0])
    if bi >= 3:
        break

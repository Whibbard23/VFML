# debug_dataset.py (save to project root)
import sys

from event_training.data_loader import EventCropDataset
from pathlib import Path
import traceback

csv = Path("event_csvs/assembly_1_train_events.csv")
print("CSV exists:", csv.exists(), "path:", csv.resolve())
ds = EventCropDataset(str(csv), data_root=".", image_size=(128,128), max_rows=200)
print("Dataset length:", len(ds))
for i in range(min(40, len(ds))):
    try:
        item = ds[i]
        print(i, "->", type(item), "len:", (len(item) if hasattr(item,'__len__') else None))
        if isinstance(item, tuple):
            print("   types:", [type(x) for x in item])
            # print label value and missing flag
            lbl = item[1].item() if hasattr(item[1], "item") else item[1]
            meta = item[2] if len(item) > 2 else None
            print("   label:", lbl, "meta.missing:", meta.get("missing") if isinstance(meta, dict) else None)
    except Exception:
        print("Exception at index", i)
        traceback.print_exc()
        break

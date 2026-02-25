"""
Patched, worker-safe PyTorch Dataset and DataLoader for event crops.

Expects CSV with columns:
  video,frame,event_type,crop_path,matched,notes,source

crop_path may be absolute or relative to `data_root` passed to the dataset.

Key fixes:
- Builds a per-dataset label map and publishes it to module-level EVENT_TO_IDX for compatibility.
- Returns label as a torch.LongTensor so collate works reliably.
- Never returns None from __getitem__; missing files produce a fallback image and label -1.
- make_dataloader exposes num_workers and max_rows for smoke tests.
"""

from pathlib import Path
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import traceback

from torch.utils.data._utils.collate import default_collate

def safe_collate(batch):
    """
    Custom collate that:
      - filters out None items
      - uses default_collate for (image_tensor, label_tensor)
      - returns metadata as a plain Python list (no collating)
    Expects each item to be (img_tensor, label_tensor, metadata_dict).
    """
    # filter out None top-level items
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        raise RuntimeError("safe_collate: all items in batch were None")

    # split into parts
    imgs_and_labels = [(b[0], b[1]) for b in batch]
    metas = [b[2] for b in batch]

    # collate images and labels using default_collate
    collated = default_collate(imgs_and_labels)  # returns tuple (imgs_tensor, labels_tensor)
    # default_collate on a list of tuples returns a tuple of collated elements
    # ensure we return (imgs, labels, metas)
    if isinstance(collated, tuple) and len(collated) == 2:
        imgs_tensor, labels_tensor = collated
    else:
        # fallback: try to unpack
        imgs_tensor, labels_tensor = collated[0], collated[1]

    return imgs_tensor, labels_tensor, metas



# Module-level map for compatibility with existing train.py that imports EVENT_TO_IDX
EVENT_TO_IDX = {}

class EventCropDataset(Dataset):
    def __init__(self, csv_path, data_root=".", image_size=(128,128), transform=None, max_rows=None):
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root)
        self.image_size = tuple(image_size)
        self.rows = []
        with self.csv_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                self.rows.append(r)
                if max_rows and len(self.rows) >= max_rows:
                    break

        # build a per-dataset label map (worker-safe)
        labels = sorted({r.get("event_type","") for r in self.rows if r.get("event_type")})
        self.event_to_idx = {lab: i for i, lab in enumerate(labels)}

        # publish to module-level map for compatibility (set in main process)
        global EVENT_TO_IDX
        EVENT_TO_IDX.clear()
        EVENT_TO_IDX.update(self.event_to_idx)


        # default transform
        if transform is None:
            # simple RGB normalization; adjust if your images are grayscale
            self.transform = T.Compose([
                T.Resize(self.image_size),
                T.CenterCrop(self.image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.rows)

    def _resolve_path(self, crop_path):
        """
        Resolve crop_path to an existing Path or return None if not found.
        Tries:
          - absolute path as given
          - relative to data_root
          - relative to CSV parent
        """
        if not crop_path:
            return None
        p = Path(crop_path)
        if p.is_absolute() and p.exists():
            return p
        candidate = (self.data_root / p).resolve()
        if candidate.exists():
            return candidate
        candidate2 = (self.csv_path.parent / p).resolve()
        if candidate2.exists():
            return candidate2
        return None

    def __getitem__(self, idx):
        """
        Returns: (image_tensor, label_tensor, metadata_dict)
        - image_tensor: torch.FloatTensor
        - label_tensor: torch.LongTensor (value -1 indicates missing/unknown label)
        - metadata_dict: dict with keys video, frame, crop_path, missing
        """
        r = self.rows[idx]
        crop_path = r.get("crop_path","")
        try:
            img_path = self._resolve_path(crop_path)
            missing = False
            if img_path is None or not img_path.exists():
                missing = True
                # fallback: create a black image
                img = Image.new("RGB", self.image_size, color=(0,0,0))
            else:
                img = Image.open(img_path).convert("RGB")
            tensor = self.transform(img)
            ev = r.get("event_type","")
            label_idx = self.event_to_idx.get(ev, -1)
            label_tensor = torch.tensor(label_idx, dtype=torch.long)
            metadata = {
                "video": r.get("video"),
                "frame": r.get("frame"),
                "crop_path": str(img_path) if img_path is not None else crop_path,
                "missing": missing
            }
            return tensor, label_tensor, metadata
        except Exception as e:
            # Defensive fallback: log traceback to stderr and return a safe fallback item
            traceback.print_exc()
            img = Image.new("RGB", self.image_size, color=(0,0,0))
            tensor = self.transform(img)
            label_tensor = torch.tensor(-1, dtype=torch.long)
            metadata = {
                "video": r.get("video"),
                "frame": r.get("frame"),
                "crop_path": crop_path,
                "missing": True,
                "error": str(e)
            }
            return tensor, label_tensor, metadata

def make_dataloader(csv_path, data_root=".", batch_size=32, image_size=(128,128), shuffle=True, num_workers=0, max_rows=None):
    """
    Factory to create DataLoader and Dataset.
    - num_workers defaults to 0 for safer debugging; pass >0 for performance.
    - max_rows can be used for smoke tests.
    Returns: (dataloader, dataset)
    """
    ds = EventCropDataset(csv_path, data_root=data_root, image_size=image_size, max_rows=max_rows)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=safe_collate)
    return dl, ds

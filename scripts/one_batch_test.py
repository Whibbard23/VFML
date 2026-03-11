from pathlib import Path
from event_training.mouth.train_mouth_detector import MouthFrameDataset, make_transforms
from torch.utils.data import DataLoader
csv = Path("event_csvs/mouth_frame_label_table.csv")
root = Path(r"E:\VF ML Crops")
ds = MouthFrameDataset(csv, root, split="train", transform=make_transforms(128, train=True))
print("dataset len:", len(ds))
dl = DataLoader(ds, batch_size=4, num_workers=0)
for i,(imgs,labels) in enumerate(dl):
    print("batch", i, "imgs.shape", getattr(imgs,'shape',None), "labels.shape", getattr(labels,'shape',None))
    break

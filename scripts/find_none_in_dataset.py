import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from event_training.datasets.mouth_detector_dataset import MouthDetectorDataset
from torchvision import transforms

ds = MouthDetectorDataset(
    csv_path="event_csvs/assembly_1_train_events.csv",
    data_root=".",
    clip_len=5,
    transform=transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])
)

bad = []
for i in range(len(ds)):
    clip, label, meta = ds[i]
    if clip is None:
        bad.append((i, 'clip'))
        continue
    if label is None:
        bad.append((i, 'label'))
    for k,v in meta.items():
        if v is None:
            bad.append((i, f"meta.{k}"))
if bad:
    print("Found None fields:", bad[:50])
else:
    print("No None fields found")

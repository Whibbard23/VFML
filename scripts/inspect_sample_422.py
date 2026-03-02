import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from event_training.datasets.mouth_detector_dataset import MouthDetectorDataset
import pandas as pd
from torchvision import transforms

csv_path = "event_csvs/assembly_1_train_events.csv"
df = pd.read_csv(csv_path)
print("CSV row 422:")
print(df.iloc[422].to_dict())

ds = MouthDetectorDataset(csv_path=csv_path, data_root=".", clip_len=5,
                          transform=transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()]))
clip, label, meta = ds[422]
print("dataset label:", label)
print("dataset meta:", meta)

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from event_training.datasets.mouth_detector_dataset import MouthDetectorDataset
from torchvision import transforms

ds = MouthDetectorDataset(
    csv_path="event_csvs/assembly_1_train_events.csv",
    data_root=".",
    clip_len=5,
    transform=transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
    ])
)

clip, label, meta = ds[100]
print("clip shape:", clip.shape)      # expect (T, C, H, W)
print("label:", label)                # 0 or 1
print("meta:", meta)                  # includes event_frame_index

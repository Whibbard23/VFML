# inference/run_inference_video.py
from ultralytics import YOLO
from pathlib import Path
import shutil
import tempfile
import sys

# Resolve paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Experiment and weights
exp = PROJECT_ROOT / "runs" / "detect" / "detector" / "models" / "yolov8_train_cpu"
weights_dir = exp / "weights"
src_ckpt = weights_dir / "best.pt"
if not src_ckpt.exists():
    src_ckpt = weights_dir / "last.pt"
if not src_ckpt.exists():
    raise FileNotFoundError(f"No checkpoint found in {weights_dir}")

# Copy checkpoint to temp to avoid partial-read
tmp_ckpt = Path(tempfile.gettempdir()) / f"ckpt_copy_{src_ckpt.stem}.pt"
shutil.copy2(src_ckpt, tmp_ckpt)

# Video source
unc_folder = r"\\research.drive.wisc.edu\npconnor\ADStudy\VF AD Blinded\Early Tongue Training"

video_path = Path(unc_folder) / "AD128.avi"

if not video_path.exists():
    raise FileNotFoundError(f"Video not found: {video_path}")

# Output project/name
project = PROJECT_ROOT / "runs" / "inference"
name = "AD128_roi"

# Run inference
model = YOLO(str(tmp_ckpt))
# device='cpu' ensures CPU inference; change to 'cuda:0' if GPU available
model.predict(
    source=str(video_path),
    save=True,
    save_txt=True,
    save_conf=True,
    project=str(project),
    name=name,
    exist_ok=True,
    device="cpu"
)

print("Inference finished. Outputs in:", project / name)

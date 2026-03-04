# visualize_fp_fn.py
from PIL import Image
import numpy as np
from pathlib import Path

FRAMES_ROOT = Path("runs/inference")

def show_frame(video, frame):
    stem = Path(video).stem
    crop = FRAMES_ROOT / f"{stem}_roi" / "crops" / f"crop_{frame:06d}.jpg"
    motion = FRAMES_ROOT / f"{stem}_roi" / "labels" / "motion" / f"motion_{frame:06d}.npy"

    rgb = Image.open(crop).convert("RGB")
    m = np.load(motion)
    m = (np.clip(m,0,1)*255).astype("uint8")
    m = Image.fromarray(m)

    rgb.show(title=f"{video} frame {frame} RGB")
    m.show(title=f"{video} frame {frame} MOTION")

# Example:
# show_frame("AD294.avi", 1234)

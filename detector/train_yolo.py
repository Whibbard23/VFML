"""
train_yolo.py
Train YOLOv8 using ultralytics on CPU-friendly settings.

Edit the CONFIG block below to set DATA_YAML, MODEL_NAME, and OUT_DIR.
Run: python detector/train_yolo.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # force CPU

import argparse
from pathlib import Path
import sys
from datetime import datetime

# -------------------------
# CONFIGURATION
DATA_YAML = Path("detector/yolo_dataset/data.yaml")
MODEL_NAME = "yolov8n.pt"
OUT_DIR = Path("detector/models")
EPOCHS = 30
BATCH = 8
IMG = 640
NUM_WORKERS = 2
# -------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--img", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()

    data = Path(args.data) if args.data else DATA_YAML
    model_name = args.model if args.model else MODEL_NAME
    epochs = args.epochs if args.epochs is not None else EPOCHS
    batch = args.batch if args.batch is not None else BATCH
    img = args.img if args.img is not None else IMG
    out = Path(args.out) if args.out else OUT_DIR
    workers = args.workers if args.workers is not None else NUM_WORKERS

    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    log_path = out / f"train_log_{timestamp}.txt"

    print(f"[INFO] Training on CPU. Data: {data}, Model: {model_name}, epochs: {epochs}, batch: {batch}, img: {img}")
    print(f"[INFO] Logs will be written to: {log_path}")

    try:
        from ultralytics import YOLO
    except Exception:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        raise

    with open(log_path, "w", buffering=1) as logf:
        logf.write(f"Training started at {datetime.utcnow().isoformat()}Z\n")
        logf.write(f"Data: {data}\nModel: {model_name}\nEpochs: {epochs}\nBatch: {batch}\nImg: {img}\nWorkers: {workers}\n\n")
        logf.flush()
        orig_out, orig_err = sys.stdout, sys.stderr
        try:
            sys.stdout = logf
            sys.stderr = logf
            model = YOLO(model_name)
            model.train(data=str(data),
                        epochs=epochs,
                        imgsz=img,
                        batch=batch,
                        project=str(out),
                        name="yolov8_train_cpu",
                        workers=workers)
        finally:
            sys.stdout = orig_out
            sys.stderr = orig_err

    print(f"[INFO] Training finished. Check logs at {log_path} and results under {out}")

if __name__ == "__main__":
    main()

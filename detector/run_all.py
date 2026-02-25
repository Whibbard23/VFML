"""
run_all.py
Run the full detector pipeline sequentially and capture logs.

Edit the CONFIG block below to set script args or leave empty to use script defaults.
Run: python detector/run_all.py
"""
import subprocess
from pathlib import Path
from datetime import datetime

# -------------------------
# CONFIGURATION
ROOT = Path(".").resolve()
DETECTOR_DIR = ROOT / "detector"
PYTHON = "python"
CONVERT = DETECTOR_DIR / "convert_to_yolo.py"
TRAIN = DETECTOR_DIR / "train_yolo.py"
INFER = DETECTOR_DIR / "infer_yolo.py"
EXTRACT = DETECTOR_DIR / "extract_crops_from_detections.py"

CONVERT_ARGS = []   # override if needed
TRAIN_ARGS = []     # override if needed
INFER_ARGS = []     # override if needed
EXTRACT_ARGS = []   # override if needed

LOG_DIR = DETECTOR_DIR / "run_logs"
# -------------------------

LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def run_step(name, cmd):
    log_out = LOG_DIR / f"{timestamp}_{name}_out.txt"
    log_err = LOG_DIR / f"{timestamp}_{name}_err.txt"
    print(f"[RUN] {name}: {' '.join(cmd)}")
    print(f"[LOG] stdout -> {log_out}")
    print(f"[LOG] stderr -> {log_err}")
    with open(log_out, "wb") as out_f, open(log_err, "wb") as err_f:
        proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, cwd=str(DETECTOR_DIR))
        ret = proc.wait()
    if ret != 0:
        print(f"[ERROR] Step {name} failed (exit {ret}). See logs.")
    else:
        print(f"[OK] Step {name} completed successfully.")
    return ret

def main():
    steps = [
        ("convert", [PYTHON, str(CONVERT)] + CONVERT_ARGS),
        ("train",   [PYTHON, str(TRAIN)] + TRAIN_ARGS),
        ("infer",   [PYTHON, str(INFER)] + INFER_ARGS),
        ("extract", [PYTHON, str(EXTRACT)] + EXTRACT_ARGS),
    ]

    for name, cmd in steps:
        ret = run_step(name, cmd)
        if ret != 0:
            print(f"[STOP] Pipeline stopped at step: {name}")
            return
    print("[DONE] All steps finished successfully. Check logs in", LOG_DIR)

if __name__ == "__main__":
    main()

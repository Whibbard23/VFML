from pathlib import Path
import csv, sys

label_csv = Path("event_csvs/mouth_frame_label_table.csv")
crops_root = Path("E:/VF ML Crops")

def safe_exists(p):
    try:
        return p.exists()
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

with open(label_csv, "r", newline="") as f:
    reader = csv.DictReader(f)
    count = 0
    for r in reader:
        if r.get("split","").strip().lower() != "val":
            continue
        vid = r["video"].strip()
        if vid.lower().endswith(".avi"):
            vid = vid[:-4]
        frm = int(r["frame"])
        jpeg_path = crops_root / vid / "crops" / "mouth" / f"frame_{frm:06d}.jpg"
        npy_path = crops_root / vid / "crops_normalized" / "mouth" / f"frame_{frm:06d}.npy"
        je = safe_exists(jpeg_path)
        ne = safe_exists(npy_path)
        print(f"{count:03d}: {jpeg_path} -> {je} ; {npy_path} -> {ne}")
        count += 1
        if count >= 200:
            break

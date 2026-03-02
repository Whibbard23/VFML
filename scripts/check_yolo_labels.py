"""
tools/check_yolo_labels.py

Print 10 random image/label pairs from detector/yolo_dataset and write overlay visuals.

Usage:
    python tools/check_yolo_labels.py

Edit constants below if your dataset lives elsewhere.
"""
import random
from pathlib import Path
import cv2
import os

# -------------------------
# CONFIG
YOLO_IMAGES_DIR = Path("detector/yolo_dataset/images")
YOLO_LABELS_DIR = Path("detector/yolo_dataset/labels")
SAMPLE_COUNT = 10
OUT_VIS_DIR = Path("tools/check_vis")
# -------------------------

def find_image_label_pairs(images_root):
    imgs = []
    for sub in ("train", "val"):
        p = images_root / sub
        if not p.exists(): continue
        for img in p.glob("*.jpg"):
            # label path mirrors image path under labels/
            rel = img.relative_to(images_root / sub)
            lbl = YOLO_LABELS_DIR / sub / rel.with_suffix(".txt")
            imgs.append((img, lbl))
    return imgs

def yolo_to_xyxy(line, img_w, img_h):
    # line: "class xc yc w h" normalized
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = parts[0]
    xc, yc, w, h = map(float, parts[1:5])
    x1 = int((xc - w/2.0) * img_w)
    y1 = int((yc - h/2.0) * img_h)
    x2 = int((xc + w/2.0) * img_w)
    y2 = int((yc + h/2.0) * img_h)
    return cls, x1, y1, x2, y2

def draw_boxes(img_path, label_lines, out_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    h, w = img.shape[:2]
    for ln in label_lines:
        parsed = yolo_to_xyxy(ln, w, h)
        if parsed is None:
            continue
        cls, x1, y1, x2, y2 = parsed
        color = (0,255,0) if cls == "0" else (0,128,255)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img, f"{cls}", (max(0,x1), max(12,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True

def main():
    pairs = find_image_label_pairs(YOLO_IMAGES_DIR)
    if not pairs:
        print("No image/label pairs found under", YOLO_IMAGES_DIR)
        return
    random.seed(42)
    sample = random.sample(pairs, min(SAMPLE_COUNT, len(pairs)))
    OUT_VIS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Checking {len(sample)} image/label pairs. Visuals -> {OUT_VIS_DIR}\n")
    for img_path, lbl_path in sample:
        print("IMAGE:", img_path)
        if lbl_path.exists():
            lines = [l.strip() for l in lbl_path.read_text().splitlines() if l.strip()]
            print("LABELS:")
            for l in lines:
                print("  ", l)
        else:
            lines = []
            print("LABELS: (missing)", lbl_path)
        vis_out = OUT_VIS_DIR / (img_path.stem + "_vis.jpg")
        ok = draw_boxes(img_path, lines, vis_out)
        if ok:
            print("VISUAL:", vis_out)
        else:
            print("VISUAL: failed to write (image read error)")
        print("-" * 60)

if __name__ == "__main__":
    main()

"""
infer_yolo.py
Run a trained YOLOv8 model on a video and save detections.

Edit the CONFIG block below to set WEIGHTS_PATH and VIDEO_PATH.
Run: python detector/infer_yolo.py
"""
import argparse, json
from pathlib import Path
import cv2
from ultralytics import YOLO

# -------------------------
# CONFIGURATION
WEIGHTS_PATH = Path("detector/models/yolov8_train/weights/best.pt")
VIDEO_PATH = Path(r"\\research.drive.wisc.edu\npconnor\ADStudy\VF AD Blinded\Early Tongue Training\AD128.avi")
OUT_JSON = Path("detector_output_AD128.json")
VIS_DIR = Path("detector/vis_AD128")
CONF_THRES = 0.25
# -------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=None)
    p.add_argument("--video", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--vis-dir", default=None)
    p.add_argument("--conf-thres", type=float, default=None)
    args = p.parse_args()

    weights = Path(args.weights) if args.weights else WEIGHTS_PATH
    video = Path(args.video) if args.video else VIDEO_PATH
    out = Path(args.out) if args.out else OUT_JSON
    vis_dir = Path(args.vis_dir) if args.vis_dir else VIS_DIR
    conf_thres = args.conf_thres if args.conf_thres is not None else CONF_THRES

    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    results = []
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(total):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        res = model.predict(source=frame, conf=conf_thres, verbose=False)
        boxes = []
        if len(res) > 0:
            r = res[0]
            for det in r.boxes:
                cls = int(det.cls.cpu().numpy())
                conf = float(det.conf.cpu().numpy())
                x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().tolist()
                boxes.append({"class": cls, "conf": conf, "xyxy": [x1, y1, x2, y2]})
        results.append({"frame": idx, "boxes": boxes})
        if vis_dir:
            vis = frame.copy()
            for b in boxes:
                x1,y1,x2,y2 = map(int, b["xyxy"])
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(vis, f"{b['class']}:{b['conf']:.2f}", (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.imwrite(str(vis_dir / f"frame_{idx:06d}.jpg"), vis)

    cap.release()
    out.write_text(json.dumps(results, indent=2))
    print("Wrote detections to", out)

if __name__ == "__main__":
    main()

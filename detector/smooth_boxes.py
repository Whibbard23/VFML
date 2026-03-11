#!/usr/bin/env python3
"""
detector/smooth_boxes.py

Read detector JSON (list of frames with "frame" and "boxes"), read video to get frame width/height,
pick top box per class per frame, densify frames between min and max, interpolate missing frames,
smooth coordinates in normalized space, and write smoothed_<inputname>.json to out-dir.

Usage:
  python detector/smooth_boxes.py --input detector_outputs/AD128.json --video videos/AD128.avi --out-dir detector_outputs --method ema --alpha 0.2
"""
import argparse
import json
from pathlib import Path
import numpy as np
from collections import defaultdict
import copy
import cv2

# Simple Kalman for 1D (position + velocity)
class SimpleKalman1D:
    def __init__(self, x0, v0=0.0, P=None, Q=1e-3, R=1e-1):
        self.x = float(x0)
        self.v = float(v0)
        self.P = np.array([[1.0, 0.0],[0.0, 1.0]]) if P is None else np.array(P)
        self.Q = Q
        self.R = R

    def predict(self, dt=1.0):
        F = np.array([[1.0, dt],[0.0, 1.0]])
        self.P = F @ self.P @ F.T + self.Q * np.eye(2)
        self.x = self.x + self.v * dt

    def update(self, meas):
        H = np.array([[1.0, 0.0]])
        y = np.array([meas]) - H @ np.array([self.x, self.v])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T / S
        state = np.array([self.x, self.v]) + (K.flatten() * y).flatten()
        self.x, self.v = float(state[0]), float(state[1])
        I = np.eye(2)
        self.P = (I - K @ H) @ self.P

    def step(self, meas, dt=1.0):
        self.predict(dt)
        self.update(meas)
        return self.x

def xyxy_to_xcycwh(xyxy, w_img, h_img):
    x1,y1,x2,y2 = xyxy
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    xc = x1 + 0.5 * w
    yc = y1 + 0.5 * h
    return (xc / w_img, yc / h_img, w / w_img, h / h_img)

def xcycwh_to_xyxy(xc,yc,w,h, w_img, h_img):
    xc *= w_img; yc *= h_img; w *= w_img; h *= h_img
    x1 = xc - 0.5*w; y1 = yc - 0.5*h; x2 = xc + 0.5*w; y2 = yc + 0.5*h
    return [float(max(0, x1)), float(max(0, y1)), float(min(w_img, x2)), float(min(h_img, y2))]

def linear_interpolate_list(arr):
    n = len(arr)
    out = arr.copy()
    idxs = [i for i,v in enumerate(arr) if v is not None]
    if not idxs:
        return [0.0]*n
    first, last = idxs[0], idxs[-1]
    for i in range(0, first):
        out[i] = arr[first]
    for i in range(last+1, n):
        out[i] = arr[last]
    for a,b in zip(idxs, idxs[1:]):
        va, vb = arr[a], arr[b]
        if b == a+1:
            continue
        for t in range(a+1, b):
            frac = (t - a) / (b - a)
            out[t] = va * (1-frac) + vb * frac
    return out

def median_smooth(arr, window=7):
    import scipy.signal
    k = window if window % 2 == 1 else window+1
    return list(scipy.signal.medfilt(arr, kernel_size=k))

def ema_smooth(arr, alpha=0.2):
    out = []
    s = None
    for v in arr:
        if s is None:
            s = v
        else:
            s = alpha * v + (1-alpha) * s
        out.append(float(s))
    return out

def kalman_smooth(arr, dt=1.0, Q=1e-3, R=1e-1):
    kf = SimpleKalman1D(arr[0], v0=0.0, Q=Q, R=R)
    out = [float(arr[0])]
    for v in arr[1:]:
        out.append(float(kf.step(v, dt)))
    return out

def smooth_tracks(frames, img_w, img_h, method="ema", **kwargs):
    N = len(frames)
    # Build per-class arrays (one top box per class per frame)
    class_coords = defaultdict(lambda: {"xc": [None]*N, "yc":[None]*N, "w":[None]*N, "h":[None]*N, "conf":[None]*N})
    for i, item in enumerate(frames):
        # pick top box per class by conf
        boxes_by_class = {}
        for b in item.get("boxes", []):
            cls = int(b.get("class", 0))
            conf = float(b.get("conf", 0.0))
            if cls not in boxes_by_class or conf > boxes_by_class[cls].get("conf", -1):
                boxes_by_class[cls] = b
        for cls, b in boxes_by_class.items():
            xyxy = b.get("xyxy")
            if xyxy is None:
                continue
            xc,yc,w,h = xyxy_to_xcycwh(xyxy, img_w, img_h)
            class_coords[cls]["xc"][i] = xc
            class_coords[cls]["yc"][i] = yc
            class_coords[cls]["w"][i] = w
            class_coords[cls]["h"][i] = h
            class_coords[cls]["conf"][i] = float(b.get("conf", 0.0))
    smoothed = {}
    for cls, coords in class_coords.items():
        smoothed[cls] = {}
        for key in ("xc","yc","w","h"):
            arr = coords[key]
            arr_filled = linear_interpolate_list(arr)
            if method == "ema":
                alpha = float(kwargs.get("alpha", 0.2))
                arr_s = ema_smooth(arr_filled, alpha=alpha)
            elif method == "median":
                window = int(kwargs.get("window", 7))
                arr_s = median_smooth(arr_filled, window=window)
            elif method == "kalman":
                Q = float(kwargs.get("Q", 1e-3))
                R = float(kwargs.get("R", 1e-1))
                arr_s = kalman_smooth(arr_filled, dt=1.0, Q=Q, R=R)
            else:
                raise ValueError("Unknown method")
            smoothed[cls][key] = arr_s
    out_frames = copy.deepcopy(frames)
    for i in range(N):
        for b in out_frames[i].get("boxes", []):
            cls = int(b.get("class", 0))
            if cls in smoothed:
                xc = smoothed[cls]["xc"][i]
                yc = smoothed[cls]["yc"][i]
                w = smoothed[cls]["w"][i]
                h = smoothed[cls]["h"][i]
                xyxy = xcycwh_to_xyxy(xc,yc,w,h, img_w, img_h)
                b["smoothed_xyxy"] = xyxy
    return out_frames

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input detector JSON (list of frames)")
    p.add_argument("--video", required=True, help="Path to the video file (used to read frame width/height)")
    p.add_argument("--out-dir", default=None, help="Directory to write smoothed JSON (defaults to input parent)")
    p.add_argument("--method", choices=["ema","median","kalman"], default="ema")
    p.add_argument("--alpha", type=float, default=0.2, help="EMA alpha (if method=ema)")
    p.add_argument("--window", type=int, default=7, help="Median window (if method=median)")
    p.add_argument("--Q", type=float, default=1e-3, help="Kalman process noise")
    p.add_argument("--R", type=float, default=1e-1, help="Kalman measurement noise")
    args = p.parse_args()

    inp = Path(args.input)
    vid = Path(args.video)
    if not inp.exists():
        raise FileNotFoundError(inp)
    if not vid.exists():
        raise FileNotFoundError(vid)

    # read video to get width/height
    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {vid}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid video dimensions read from {vid}: {w}x{h}")

    data = json.loads(inp.read_text())
    frames_by_idx = {}
    minf = None; maxf = None
    for item in data:
        fi = int(item["frame"])
        frames_by_idx[fi] = item
        if minf is None or fi < minf: minf = fi
        if maxf is None or fi > maxf: maxf = fi
    if minf is None:
        raise ValueError("No frames in input")
    N = maxf - minf + 1
    frames = []
    for i in range(minf, maxf+1):
        if i in frames_by_idx:
            frames.append(frames_by_idx[i])
        else:
            frames.append({"frame": i, "boxes": []})
    smoothed_frames = smooth_tracks(frames, w, h, method=args.method, alpha=args.alpha, window=args.window, Q=args.Q, R=args.R)
    out_dir = Path(args.out_dir) if args.out_dir else inp.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    outp = out_dir / f"smoothed_{inp.name}"
    outp.write_text(json.dumps(smoothed_frames, indent=2))
    print("Wrote smoothed JSON:", outp)

if __name__ == "__main__":
    main()

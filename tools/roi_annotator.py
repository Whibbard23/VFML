#!/usr/bin/env python3
"""
roi_annotator.py

Usage:
    python roi_annotator.py /path/to/video.avi
    python roi_annotator.py /path/to/videos_dir --width 256 --height 128

Produces: rois.json in the current working directory (or --out)
Each entry: { "video_basename.avi": [x, y, w, h] }

Controls:
  - Mouse drag: move the rectangle
  - Arrow keys / WASD: nudge rectangle by 1 or SHIFT+nudge by 10
  - +/- : increase/decrease rectangle size (keeps center)
  - f : jump to first frame (representative)
  - m : jump to middle frame (representative)
  - s : save ROI for current video
  - n : save and advance to next video
  - q or ESC : quit (saves rois.json automatically)
"""

import cv2
import json
import argparse
from pathlib import Path
import sys

# -------------------------
# Config / defaults
# -------------------------
DEFAULT_WIDTH = 256
DEFAULT_HEIGHT = 128
DEFAULT_OUT = "rois.json"
REP_FRAME = "middle"  # "first" or "middle" or integer frame index

# -------------------------
# Helper: load/save JSON
# -------------------------
def load_rois(path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}

def save_rois(path, data):
    path.write_text(json.dumps(data, indent=2))

# -------------------------
# Annotator class
# -------------------------
class ROIAnnotator:
    def __init__(self, video_path, w, h, rois_dict):
        self.video_path = Path(video_path)
        self.w = int(w)
        self.h = int(h)
        self.rois = rois_dict  # shared dict to update
        self.window_name = f"ROI Annotator - {self.video_path.name}"
        self.dragging = False
        self.offset = (0, 0)
        self.nudge = 1
        self.frame = None
        self.frame_display = None
        self.frame_idx = 0
        self.total_frames = 0
        self.x = 0
        self.y = 0
        self._open_video()
        self._init_roi_from_existing()

    def _open_video(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        if REP_FRAME == "first":
            idx = 0
        elif REP_FRAME == "middle":
            idx = max(0, self.total_frames // 2)
        else:
            try:
                idx = int(REP_FRAME)
            except Exception:
                idx = max(0, self.total_frames // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            # fallback to first frame
            cap = cv2.VideoCapture(str(self.video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                raise IOError(f"Cannot read any frame from: {self.video_path}")
        self.frame = frame
        self.frame_display = frame.copy()
        h_img, w_img = frame.shape[:2]
        # default center ROI
        self.x = max(0, (w_img - self.w) // 2)
        self.y = max(0, (h_img - self.h) // 2)

    def _init_roi_from_existing(self):
        key = self.video_path.name
        if key in self.rois:
            x, y, w, h = self.rois[key]
            self.x, self.y, self.w, self.h = int(x), int(y), int(w), int(h)

    def _clamp(self):
        h_img, w_img = self.frame.shape[:2]
        self.w = max(1, min(self.w, w_img))
        self.h = max(1, min(self.h, h_img))
        self.x = max(0, min(self.x, w_img - self.w))
        self.y = max(0, min(self.y, h_img - self.h))

    def draw(self):
        self.frame_display = self.frame.copy()
        # translucent overlay
        overlay = self.frame_display.copy()
        cv2.rectangle(overlay, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), -1)
        alpha = 0.15
        cv2.addWeighted(overlay, alpha, self.frame_display, 1 - alpha, 0, self.frame_display)
        # border
        cv2.rectangle(self.frame_display, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 2)
        # text
        txt = f"{self.video_path.name}  ROI: x={self.x} y={self.y} w={self.w} h={self.h}"
        cv2.putText(self.frame_display, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(self.frame_display, "Arrows/WASD: move  +/-: resize  s: save  n: save+next  q: quit", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.imshow(self.window_name, self.frame_display)

    # Mouse callbacks
    def on_mouse(self, event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # if click inside rect, start dragging
            if self.x <= mx <= self.x + self.w and self.y <= my <= self.y + self.h:
                self.dragging = True
                self.offset = (mx - self.x, my - self.y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            nx = mx - self.offset[0]
            ny = my - self.offset[1]
            self.x, self.y = int(nx), int(ny)
            self._clamp()
            self.draw()
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def save_roi(self):
        key = self.video_path.name
        self.rois[key] = [int(self.x), int(self.y), int(self.w), int(self.h)]
        print(f"[SAVED] {key} -> {self.rois[key]}")

    def run_interactive(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        self.draw()
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or q
                break
            elif key in (ord('s'),):
                self.save_roi()
                self.draw()
            elif key in (ord('n'),):
                self.save_roi()
                break  # signal to caller to advance
            elif key in (ord('+'), ord('=')):
                # increase size (keep center)
                cx = self.x + self.w // 2
                cy = self.y + self.h // 2
                self.w = int(self.w * 1.1)
                self.h = int(self.h * 1.1)
                self.x = cx - self.w // 2
                self.y = cy - self.h // 2
                self._clamp()
                self.draw()
            elif key in (ord('-'), ord('_')):
                cx = self.x + self.w // 2
                cy = self.y + self.h // 2
                self.w = max(8, int(self.w * 0.9))
                self.h = max(8, int(self.h * 0.9))
                self.x = cx - self.w // 2
                self.y = cy - self.h // 2
                self._clamp()
                self.draw()
            elif key in (ord('f'),):
                # jump to first frame
                cap = cv2.VideoCapture(str(self.video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    self.frame = frame
                    self._clamp()
                    self.draw()
            elif key in (ord('m'),):
                # jump to middle frame
                cap = cv2.VideoCapture(str(self.video_path))
                mid = max(0, self.total_frames // 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    self.frame = frame
                    self._clamp()
                    self.draw()
            elif key in (81,):  # left arrow
                self.x -= self.nudge
                self._clamp()
                self.draw()
            elif key in (83,):  # right arrow
                self.x += self.nudge
                self._clamp()
                self.draw()
            elif key in (82,):  # up arrow
                self.y -= self.nudge
                self._clamp()
                self.draw()
            elif key in (84,):  # down arrow
                self.y += self.nudge
                self._clamp()
                self.draw()
            elif key in (ord('a'),):
                self.x -= self.nudge
                self._clamp()
                self.draw()
            elif key in (ord('d'),):
                self.x += self.nudge
                self._clamp()
                self.draw()
            elif key in (ord('w'),):
                self.y -= self.nudge
                self._clamp()
                self.draw()
            elif key in (ord('s'),):  # note: 's' already handled above for save
                pass
            elif key == ord(' '):
                # toggle nudge size between 1 and 10
                self.nudge = 10 if self.nudge == 1 else 1
                print(f"[INFO] nudge set to {self.nudge}")
            else:
                # ignore other keys
                continue
        cv2.destroyWindow(self.window_name)

# -------------------------
# Main CLI
# -------------------------
def iter_videos(path):
    p = Path(path)
    if p.is_file():
        yield p
    elif p.is_dir():
        for ext in ("*.avi", "*.mp4", "*.mov", "*.mkv"):
            for f in sorted(p.glob(ext)):
                yield f
    else:
        raise FileNotFoundError(path)

def main():
    parser = argparse.ArgumentParser(description="ROI annotator (fixed-size movable box)")
    parser.add_argument("input", help="Video file or directory of videos")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="ROI width (pixels)")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="ROI height (pixels)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output JSON file for ROIs")
    args = parser.parse_args()

    out_path = Path(args.out)
    rois = load_rois(out_path)

    videos = list(iter_videos(args.input))
    if not videos:
        print("No videos found.")
        return

    print(f"Found {len(videos)} videos. Output: {out_path}")
    for vid in videos:
        print(f"\nAnnotating: {vid.name}")
        annot = ROIAnnotator(vid, args.width, args.height, rois)
        annot.run_interactive()
        # save after each video to avoid data loss
        save_rois(out_path, rois)

    print("\nAll done. ROIs saved to", out_path)

if __name__ == "__main__":
    main()

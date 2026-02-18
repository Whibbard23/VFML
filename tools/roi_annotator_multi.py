#!/usr/bin/env python3
"""
roi_annotator_multi_locked.py

Dual-ROI multi-frame annotator with locked canonical ROI size at save.

Usage:
    # Uses DEFAULT_VIDEOS_DIR if no input path is provided:
    python tools/roi_annotator_multi_locked.py
    # Or override with a specific path:
    python tools/roi_annotator_multi_locked.py /path/to/videos_dir \
        --width 256 --height 256 --frames-per-video 8 --out data/rois_multi.json

Controls:
  - Mouse drag: move the active rectangle
  - Tab or '1' / '2': switch active box (mouth=1, ues=2)
  - Arrow keys / WASD: nudge active box by 1 (space toggles 10)
  - +/- : increase/decrease active box size (temporary only)
  - v : cycle visibility (visible -> partial -> not_visible)
  - r : reject current frame and replace with a new non-black frame
  - u : undo last replacement for current slot (return to previous frame)
  - p : return to the last saved frame for this slot (if any)
  - b : go back one sampled slot (opens saved frame if present)
  - s : save current frame's annotations (canonical size enforced)
  - n : save and advance to next sampled frame
  - N : save and advance to next video
  - q or ESC : quit (saves JSON automatically)
"""

import cv2
import json
import argparse
from pathlib import Path
import sys
import math
import random

# -------------------------c
# User defaults
# -------------------------
DEFAULT_VIDEOS_DIR = (r"\\research.drive.wisc.edu\npconnor\ADStudy\VF AD Blinded\Early Tongue Training")

DEFAULT_WIDTH = 128
DEFAULT_HEIGHT = 128
DEFAULT_OUT = "rois_multi.json"
DEFAULT_FRAMES_PER_VIDEO = 8

VISIBILITY_STATES = ["visible", "partial", "not_visible"]

# Thresholds for black-frame detection and search limits
BLACK_MEAN_THRESHOLD = 10.0
FIRST_NONBLACK_SCAN_LIMIT = 200
REPLACEMENT_MAX_TRIES = 500

# Never use the first N frames for annotation
MIN_FRAME_SKIP = 10

# Display scale multiplier (optional)
DISPLAY_SCALE = 3.0
SCREEN_MARGIN = 80

# -------------------------
# JSON helpers
# -------------------------
def load_rois(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}

def save_rois(path: Path, data):
    path.write_text(json.dumps(data, indent=2))

# -------------------------
# Sampling helper
# -------------------------
def sample_frame_indices(total_frames, n, min_frame_skip=MIN_FRAME_SKIP):
    """
    Uniformly sample n frame indices from [min_frame_skip, total_frames-1].
    If the available range is smaller than n, will return as many unique indices as possible.
    """
    if n <= 0:
        return []
    start = max(0, min_frame_skip)
    end = max(0, total_frames - 1)
    if start > end:
        # fallback to 0..end
        start = 0
    span = end - start + 1
    if span <= 0:
        return [0]
    if n == 1:
        return [min(end, max(start, (start + end) // 2))]
    # compute evenly spaced indices in the [start, end] interval
    step = span / float(n)
    indices = [int(min(end, math.floor(start + i * step))) for i in range(n)]
    # dedupe and sort
    indices = sorted(list(dict.fromkeys(indices)))
    # if dedup reduced count, fill with random unique choices in range
    tries = 0
    while len(indices) < n and tries < 1000:
        tries += 1
        cand = random.randint(start, end)
        if cand not in indices:
            indices.append(cand)
    return sorted(indices)

# -------------------------
# Annotator class (locked canonical size at save)
# -------------------------
class DualROIAnnotatorLocked:
    def __init__(self, video_path: Path, canonical_w: int, canonical_h: int, frames_per_video: int, rois_dict, out_path: Path):
        self.video_path = video_path
        self.canonical_w = int(canonical_w)
        self.canonical_h = int(canonical_h)
        self.frames_per_video = int(frames_per_video)
        self.rois = rois_dict
        self.out_path = Path(out_path)
        self.window_name = f"ROI Annotator - {self.video_path.name}"
        self.dragging = False
        self.offset = (0, 0)
        self.nudge = 1
        self.frame = None
        self.frame_display = None
        self.total_frames = 0
        # sampled_indices is a list of frame indices, one per slot
        self.sampled_indices = []
        self.current_sample_idx = 0
        # history per slot: list of previous indices (stack)
        self.slot_history = {}
        # last saved frame per slot (if user saved this slot previously)
        self.last_saved_frame = {}

        # two boxes: mouth and ues
        self.boxes = {
            "mouth": {"x": 0, "y": 0, "w": self.canonical_w, "h": self.canonical_h},
            "ues": {"x": 0, "y": 0, "w": self.canonical_w, "h": self.canonical_h}
        }
        self.active_box = "mouth"
        self.visibility = "visible"
        # populate saved mappings for this video (if any) before sampling
        self._load_saved_mappings()
        self._open_video_and_sample()
        self._init_from_existing()

    # -------------------------
    # Load saved mappings from rois for this video
    # -------------------------
    def _load_saved_mappings(self):
        """
        Populate a mapping of saved frames for this video so that when we sample
        we prefer frames that were previously saved (so replacements persist across runs).
        This prefers a persisted __slot_map__ if present in the JSON.
        """
        self.saved_frames_for_video = []
        key = self.video_path.name if isinstance(self.video_path, Path) else None

        # prefer persisted slot map if present
        try:
            slot_map = self.rois.get("__slot_map__", {}) if isinstance(self.rois, dict) else {}
        except Exception:
            slot_map = {}

        if key and key in slot_map:
            try:
                frames = [int(x) for x in slot_map.get(key, []) if x is not None]
                self.saved_frames_for_video = sorted(frames)
                return
            except Exception:
                self.saved_frames_for_video = []

        # fallback: collect unique saved frame indices from ROI entries (existing behavior)
        if key and key in self.rois:
            entries = self.rois.get(key, [])
            frames = sorted({int(e.get("frame_index")) for e in entries if e.get("frame_index") is not None})
            self.saved_frames_for_video = frames
        else:
            self.saved_frames_for_video = []

    # -------------------------
    # Helpers for non-black frame selection
    # -------------------------
    def _frame_mean_at(self, idx):
        """Return mean pixel value for frame idx, or None if unreadable."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            cap.release()
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        idx = max(0, min(idx, max(0, total - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        cap.release()
        if not ret or f is None:
            return None
        if len(f.shape) == 3 and f.shape[2] == 3:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            return float(gray.mean())
        return float(f.mean())

    def _find_first_nonblack(self, start_idx=MIN_FRAME_SKIP, limit=FIRST_NONBLACK_SCAN_LIMIT, mean_thresh=BLACK_MEAN_THRESHOLD):
        """Scan forward from start_idx up to limit frames to find a non-black frame index."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            cap.release()
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        end = min(total - 1, start_idx + limit - 1)
        found = None
        for i in range(start_idx, end + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, f = cap.read()
            if not ret or f is None:
                continue
            if len(f.shape) == 3 and f.shape[2] == 3:
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                m = float(gray.mean())
            else:
                m = float(f.mean())
            if m > mean_thresh:
                found = i
                break
        cap.release()
        return found

    def _find_replacement_frame(self):
        """
        Pick a new frame index that:
          - is not in sampled_indices (to avoid duplicates)
          - is not black
          - is a valid frame in the video
        Returns the new index or None.
        """
        tried = set()
        tries = 0
        start = min(MIN_FRAME_SKIP, max(0, self.total_frames - 1))
        while tries < REPLACEMENT_MAX_TRIES:
            tries += 1
            if self.total_frames - 1 <= start:
                idx = start
            else:
                idx = random.randint(start, max(1, self.total_frames - 1))
            if idx in self.sampled_indices:
                tried.add(idx)
                continue

            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            cap.release()
            if not ret or f is None:
                tried.add(idx)
                continue

            # compute mean on grayscale
            if len(f.shape) == 3 and f.shape[2] == 3:
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                m = float(gray.mean())
            else:
                m = float(f.mean())

            if m <= BLACK_MEAN_THRESHOLD:
                tried.add(idx)
                continue

            return idx

        print("[WARN] Could not find a non-black replacement frame after many tries")
        return None

    # -------------------------
    # Window sizing helpers
    # -------------------------
    def _get_screen_size(self):
        """Return (screen_w, screen_h). Use tkinter if available, else fallback."""
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            w = root.winfo_screenwidth()
            h = root.winfo_screenheight()
            root.destroy()
            return int(w), int(h)
        except Exception:
            return 1920, 1080

    def _compute_display_size(self, frame_w, frame_h):
        """
        Compute a display window size that is DISPLAY_SCALE times the frame size,
        but clamped to the screen size minus margin while preserving aspect ratio.
        """
        screen_w, screen_h = self._get_screen_size()
        target_w = int(frame_w * DISPLAY_SCALE)
        target_h = int(frame_h * DISPLAY_SCALE)

        max_w = max(200, screen_w - SCREEN_MARGIN)
        max_h = max(200, screen_h - SCREEN_MARGIN)

        # If target fits, return it
        if target_w <= max_w and target_h <= max_h:
            return target_w, target_h

        # Otherwise scale down to fit within max while preserving aspect ratio
        scale_w = max_w / frame_w
        scale_h = max_h / frame_h
        scale = min(scale_w, scale_h)
        return max(200, int(frame_w * scale)), max(200, int(frame_h * scale))

    def _update_window_size(self):
        """Resize the OpenCV window to the computed display size for the current frame."""
        if self.frame is None:
            return
        h_img, w_img = self.frame.shape[:2]
        disp_w, disp_h = self._compute_display_size(w_img, h_img)
        try:
            cv2.resizeWindow(self.window_name, disp_w, disp_h)
        except Exception:
            # Some OpenCV builds may not support resizeWindow in certain environments;
            # ignore failures silently.
            pass

    # -------------------------
    # Video open and sampling
    # -------------------------
    def _open_video_and_sample(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.release()

        # Start sampled_indices with any saved frames for this video (persist replacements)
        sampled = []
        for fidx in self.saved_frames_for_video:
            if len(sampled) >= self.frames_per_video:
                break
            # skip invalid or too-small frames
            if fidx < MIN_FRAME_SKIP or fidx < 0 or fidx >= self.total_frames:
                continue
            sampled.append(int(fidx))

        # Fill remaining slots with uniform sampling while avoiding duplicates and first MIN_FRAME_SKIP frames
        if len(sampled) < self.frames_per_video:
            candidates = sample_frame_indices(self.total_frames, self.frames_per_video * 2, min_frame_skip=MIN_FRAME_SKIP)
            for c in candidates:
                if c in sampled:
                    continue
                sampled.append(c)
                if len(sampled) >= self.frames_per_video:
                    break

        # If still short (edge cases), fill with random non-duplicates from allowed range
        while len(sampled) < self.frames_per_video:
            if self.total_frames - 1 <= MIN_FRAME_SKIP:
                cand = MIN_FRAME_SKIP if self.total_frames > MIN_FRAME_SKIP else 0
            else:
                cand = random.randint(MIN_FRAME_SKIP, max(1, self.total_frames - 1))
            if cand not in sampled:
                sampled.append(cand)

        # ensure deterministic order (but keep as-is so slot ordering is stable)
        self.sampled_indices = sampled[:self.frames_per_video]

        # initialize per-slot history lists and last_saved_frame mapping
        for slot_idx in range(len(self.sampled_indices)):
            self.slot_history.setdefault(slot_idx, [])
            # if this slot's frame matches a saved frame, record it as last_saved_frame
            frame_idx = int(self.sampled_indices[slot_idx])
            key = self.video_path.name
            if key in self.rois:
                for e in self.rois.get(key, []):
                    if int(e.get("frame_index")) == frame_idx:
                        self.last_saved_frame[slot_idx] = frame_idx
                        break

        # Load the first sampled frame
        self.current_sample_idx = 0
        self._load_frame_at(self.sampled_indices[self.current_sample_idx])

    def _load_frame_at(self, idx):
        # clamp idx to allowed range and avoid using frames < MIN_FRAME_SKIP when possible
        idx = int(idx)
        if idx < MIN_FRAME_SKIP and self.total_frames > MIN_FRAME_SKIP:
            idx = MIN_FRAME_SKIP
        cap = cv2.VideoCapture(str(self.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(idx, self.total_frames - 1)))
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            # fallback: try to load first non-black after MIN_FRAME_SKIP
            candidate = self._find_first_nonblack(start_idx=MIN_FRAME_SKIP)
            if candidate is not None:
                cap = cv2.VideoCapture(str(self.video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, candidate)
                ret, frame = cap.read()
                cap.release()
            if not ret or frame is None:
                # final fallback to the nearest valid frame
                cap = cv2.VideoCapture(str(self.video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                cap.release()
                if not ret or frame is None:
                    raise IOError(f"Cannot read any frame from: {self.video_path}")
        self.frame = frame
        h_img, w_img = frame.shape[:2]
        for k in ("mouth", "ues"):
            if self.boxes[k]["w"] > w_img or self.boxes[k]["h"] > h_img:
                self.boxes[k]["w"] = min(self.boxes[k]["w"], w_img)
                self.boxes[k]["h"] = min(self.boxes[k]["h"], h_img)
            if self.boxes[k]["x"] == 0 and self.boxes[k]["y"] == 0:
                self.boxes[k]["x"] = max(0, (w_img - self.boxes[k]["w"]) // 2)
                self.boxes[k]["y"] = max(0, (h_img - self.boxes[k]["h"]) // 2)
        self.frame_display = self.frame.copy()
        # update window size to be DISPLAY_SCALE times the frame (clamped to screen)
        try:
            self._update_window_size()
        except Exception:
            pass

    def _init_from_existing(self):
        """
        Load saved ROIs for the current sampled frame (if any) into the UI.
        Also ensure last_saved_frame mapping is updated for the current slot.
        """
        key = self.video_path.name
        if key in self.rois:
            entries = self.rois.get(key, [])
            cur_frame = int(self.sampled_indices[self.current_sample_idx])
            for e in entries:
                if int(e.get("frame_index")) == cur_frame:
                    if "mouth" in e:
                        x, y, w, h = e["mouth"]
                        self.boxes["mouth"].update({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
                    if "ues" in e:
                        x, y, w, h = e["ues"]
                        self.boxes["ues"].update({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
                    if "visibility" in e:
                        self.visibility = e.get("visibility", self.visibility)
                    # record that this slot's last saved frame is cur_frame
                    self.last_saved_frame[self.current_sample_idx] = cur_frame
                    break

    def _clamp_box(self, box):
        h_img, w_img = self.frame.shape[:2]
        box["w"] = max(8, min(box["w"], w_img))
        box["h"] = max(8, min(box["h"], h_img))
        box["x"] = max(0, min(box["x"], w_img - box["w"]))
        box["y"] = max(0, min(box["y"], h_img - box["h"]))

    def draw(self):
        self.frame_display = self.frame.copy()
        overlay = self.frame_display.copy()
        for name, color in (("mouth", (0, 255, 0)), ("ues", (255, 128, 0))):
            b = self.boxes[name]
            cv2.rectangle(overlay, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), color, -1)
        alpha = 0.12
        cv2.addWeighted(overlay, alpha, self.frame_display, 1 - alpha, 0, self.frame_display)
        for name, color in (("mouth", (0, 255, 0)), ("ues", (255, 128, 0))):
            b = self.boxes[name]
            thickness = 3 if name == self.active_box else 2
            cv2.rectangle(self.frame_display, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), color, thickness)
            cv2.putText(self.frame_display, f"{name}", (b["x"], b["y"] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cur_frame = self.sampled_indices[self.current_sample_idx]
        txt = f"{self.video_path.name}  frame={cur_frame}/{self.total_frames-1}  vis={self.visibility}  active={self.active_box}"
        cv2.putText(self.frame_display, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(self.frame_display, "Tab/1/2: switch | +/- resize | v visibility | r: reject | u: undo | p: saved | b: back | n: next | N: next video | q: quit",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        cv2.imshow(self.window_name, self.frame_display)

    def on_mouse(self, event, mx, my, flags, param):
        b = self.boxes[self.active_box]
        if event == cv2.EVENT_LBUTTONDOWN:
            if b["x"] <= mx <= b["x"] + b["w"] and b["y"] <= my <= b["y"] + b["h"]:
                self.dragging = True
                self.offset = (mx - b["x"], my - b["y"])
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            nx = mx - self.offset[0]
            ny = my - self.offset[1]
            b["x"], b["y"] = int(nx), int(ny)
            self._clamp_box(b)
            self.draw()
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def _enforce_canonical_and_center(self, box):
        cx = box["x"] + box["w"] // 2
        cy = box["y"] + box["h"] // 2
        box["w"] = self.canonical_w
        box["h"] = self.canonical_h
        box["x"] = int(cx - box["w"] // 2)
        box["y"] = int(cy - box["h"] // 2)
        self._clamp_box(box)

    def save_current_frame_annotation(self):
        key = self.video_path.name
        if key not in self.rois:
            self.rois[key] = []
        cur_frame = self.sampled_indices[self.current_sample_idx]
        mouth_box = dict(self.boxes["mouth"])
        ues_box = dict(self.boxes["ues"])
        self._enforce_canonical_and_center(mouth_box)
        self._enforce_canonical_and_center(ues_box)
        entry = {
            "frame_index": int(cur_frame),
            "mouth": [int(mouth_box["x"]), int(mouth_box["y"]), int(mouth_box["w"]), int(mouth_box["h"])],
            "ues": [int(ues_box["x"]), int(ues_box["y"]), int(ues_box["w"]), int(ues_box["h"])],
            "visibility": self.visibility
        }
        entries = self.rois.get(key, [])
        replaced = False
        for i, e in enumerate(entries):
            if int(e.get("frame_index")) == int(cur_frame):
                entries[i] = entry
                replaced = True
                break
        if not replaced:
            entries.append(entry)
        entries = sorted(entries, key=lambda x: int(x["frame_index"]))
        self.rois[key] = entries

        # record last saved frame for this slot
        self.last_saved_frame[self.current_sample_idx] = int(cur_frame)

        # update in-memory boxes to the canonical centered ones so the UI reflects saved state
        self.boxes["mouth"].update({"x": entry["mouth"][0], "y": entry["mouth"][1], "w": entry["mouth"][2], "h": entry["mouth"][3]})
        self.boxes["ues"].update({"x": entry["ues"][0], "y": entry["ues"][1], "w": entry["ues"][2], "h": entry["ues"][3]})
        print(f"[SAVED] {key} frame {cur_frame} -> mouth={entry['mouth']} ues={entry['ues']} vis={self.visibility}")

        # persist slot map so saved frame is associated with slot across runs
        try:
            if "__slot_map__" not in self.rois or not isinstance(self.rois["__slot_map__"], dict):
                self.rois["__slot_map__"] = {}
            self.rois["__slot_map__"][self.video_path.name] = [int(x) for x in self.sampled_indices]
            save_rois(self.out_path, self.rois)
        except Exception:
            pass

    def run_interactive(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        # ensure window is sized appropriately for the first loaded frame
        try:
            self._update_window_size()
        except Exception:
            pass
        self.draw()
        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == 27 or key == ord('q'):
                break

            elif key in (9, ord('1')):
                self.active_box = "mouth"
                self.draw()

            elif key == ord('2'):
                self.active_box = "ues"
                self.draw()

            elif key in (ord('+'), ord('=')):
                b = self.boxes[self.active_box]
                cx = b["x"] + b["w"] // 2
                cy = b["y"] + b["h"] // 2
                b["w"] = int(b["w"] * 1.1)
                b["h"] = int(b["h"] * 1.1)
                b["x"] = cx - b["w"] // 2
                b["y"] = cy - b["h"] // 2
                self._clamp_box(b)
                self.draw()

            elif key in (ord('-'), ord('_')):
                b = self.boxes[self.active_box]
                cx = b["x"] + b["w"] // 2
                cy = b["y"] + b["h"] // 2
                b["w"] = max(8, int(b["w"] * 0.9))
                b["h"] = max(8, int(b["h"] * 0.9))
                b["x"] = cx - b["w"] // 2
                b["y"] = cy - b["h"] // 2
                self._clamp_box(b)
                self.draw()

            elif key == ord(' '):
                self.nudge = 10 if self.nudge == 1 else 1
                print(f"[INFO] nudge set to {self.nudge}")

            # Reject current frame and replace with a new one
            elif key == ord('r'):
                slot = self.current_sample_idx
                old_idx = self.sampled_indices[slot]
                new_idx = self._find_replacement_frame()
                if new_idx is not None:
                    # push old index onto history for this slot
                    self.slot_history.setdefault(slot, []).append(old_idx)
                    # set new index for this slot (do NOT reorder sampled_indices)
                    self.sampled_indices[slot] = new_idx
                    print(f"[INFO] Replaced slot {slot} frame {old_idx} -> {new_idx}")

                    # persist the updated slot mapping so replacement survives across runs
                    try:
                        if "__slot_map__" not in self.rois or not isinstance(self.rois["__slot_map__"], dict):
                            self.rois["__slot_map__"] = {}
                        self.rois["__slot_map__"][self.video_path.name] = [int(x) for x in self.sampled_indices]
                        save_rois(self.out_path, self.rois)
                        print(f"[INFO] persisted slot map for {self.video_path.name}")
                    except Exception as ex:
                        print(f"[WARN] failed to persist slot map: {ex}")

                    # load the new frame immediately and update window size
                    self._load_frame_at(new_idx)
                    # initialize boxes from any existing saved ROI for this new frame
                    self._init_from_existing()
                    try:
                        self._update_window_size()
                    except Exception:
                        pass
                    self.draw()
                else:
                    print("[WARN] No replacement frame found")

            # Undo last replacement for current slot (return to previous frame)
            elif key == ord('u'):
                slot = self.current_sample_idx
                history = self.slot_history.get(slot, [])
                if history:
                    prev_idx = history.pop()  # last replaced index
                    cur_idx = self.sampled_indices[slot]
                    self.sampled_indices[slot] = prev_idx
                    print(f"[INFO] Reverted slot {slot} from {cur_idx} back to {prev_idx}")

                    # persist updated slot map
                    try:
                        if "__slot_map__" not in self.rois or not isinstance(self.rois["__slot_map__"], dict):
                            self.rois["__slot_map__"] = {}
                        self.rois["__slot_map__"][self.video_path.name] = [int(x) for x in self.sampled_indices]
                        save_rois(self.out_path, self.rois)
                    except Exception:
                        pass

                    self._load_frame_at(prev_idx)
                    self._init_from_existing()
                    try:
                        self._update_window_size()
                    except Exception:
                        pass
                    self.draw()
                else:
                    print("[INFO] No replacement history for this slot to undo")

            # Return to last saved frame for this slot (if any)
            elif key == ord('p'):
                slot = self.current_sample_idx
                saved = self.last_saved_frame.get(slot)
                if saved is not None:
                    cur_idx = self.sampled_indices[slot]
                    if saved != cur_idx:
                        # push current into history
                        self.slot_history.setdefault(slot, []).append(cur_idx)
                        self.sampled_indices[slot] = saved
                        print(f"[INFO] Loaded last saved frame {saved} into slot {slot} (was {cur_idx})")

                        # persist updated slot map
                        try:
                            if "__slot_map__" not in self.rois or not isinstance(self.rois["__slot_map__"], dict):
                                self.rois["__slot_map__"] = {}
                            self.rois["__slot_map__"][self.video_path.name] = [int(x) for x in self.sampled_indices]
                            save_rois(self.out_path, self.rois)
                        except Exception:
                            pass

                        self._load_frame_at(saved)
                        self._init_from_existing()
                        try:
                            self._update_window_size()
                        except Exception:
                            pass
                        self.draw()
                    else:
                        print("[INFO] Current frame is already the last saved frame for this slot")
                else:
                    print("[INFO] No saved frame recorded for this slot")

                        
            # Display NEXT sequential frame (e.g., 164 -> 165)
            elif key == ord('>'):
                cur = self.sampled_indices[self.current_sample_idx]
                next_idx = min(cur + 1, self.total_frames - 1)

                # push current into history
                self.slot_history.setdefault(self.current_sample_idx, []).append(cur)

                # update slot to new frame
                self.sampled_indices[self.current_sample_idx] = next_idx
                print(f"[INFO] Sequential next: {cur} -> {next_idx}")

                # persist slot map
                try:
                    if "__slot_map__" not in self.rois or not isinstance(self.rois["__slot_map__"], dict):
                        self.rois["__slot_map__"] = {}
                    self.rois["__slot_map__"][self.video_path.name] = [int(x) for x in self.sampled_indices]
                    save_rois(self.out_path, self.rois)
                except Exception:
                    pass

                # load frame + ROIs
                self._load_frame_at(next_idx)
                self._init_from_existing()
                try:
                    self._update_window_size()
                except Exception:
                    pass
                self.draw()

            # Display PREVIOUS sequential frame (e.g., 164 -> 163)
            elif key == ord('<'):
                cur = self.sampled_indices[self.current_sample_idx]
                prev_idx = max(cur - 1, MIN_FRAME_SKIP)

                # push current into history
                self.slot_history.setdefault(self.current_sample_idx, []).append(cur)

                # update slot to new frame
                self.sampled_indices[self.current_sample_idx] = prev_idx
                print(f"[INFO] Sequential previous: {cur} -> {prev_idx}")

                # persist slot map
                try:
                    if "__slot_map__" not in self.rois or not isinstance(self.rois["__slot_map__"], dict):
                        self.rois["__slot_map__"] = {}
                    self.rois["__slot_map__"][self.video_path.name] = [int(x) for x in self.sampled_indices]
                    save_rois(self.out_path, self.rois)
                except Exception:
                    pass

                # load frame + ROIs
                self._load_frame_at(prev_idx)
                self._init_from_existing()
                try:
                    self._update_window_size()
                except Exception:
                    pass
                self.draw()


            # Go back to the previous sampled slot and open its saved frame (if any)
            elif key == ord('b'):
                # only go back if not at the first slot
                if self.current_sample_idx > 0:
                    prev_slot = self.current_sample_idx - 1
                    # prefer the last saved frame for that slot if available
                    saved_frame = self.last_saved_frame.get(prev_slot)
                    if saved_frame is not None:
                        # ensure sampled_indices reflects the saved frame for UI consistency
                        self.sampled_indices[prev_slot] = int(saved_frame)
                        print(f"[INFO] Going back to slot {prev_slot}, opening last saved frame {saved_frame}")
                        self.current_sample_idx = prev_slot
                        self._load_frame_at(self.sampled_indices[self.current_sample_idx])
                        self._init_from_existing()
                        try:
                            self._update_window_size()
                        except Exception:
                            pass
                        self.draw()
                    else:
                        # no saved frame recorded; just move back to the sampled index
                        target = self.sampled_indices[prev_slot]
                        print(f"[INFO] Going back to slot {prev_slot}, opening sampled frame {target}")
                        self.current_sample_idx = prev_slot
                        self._load_frame_at(target)
                        self._init_from_existing()
                        try:
                            self._update_window_size()
                        except Exception:
                            pass
                        self.draw()
                else:
                    print("[INFO] Already at the first sampled slot; cannot go back.")

            elif key == ord('v'):
                idx = VISIBILITY_STATES.index(self.visibility)
                idx = (idx + 1) % len(VISIBILITY_STATES)
                self.visibility = VISIBILITY_STATES[idx]
                print(f"[INFO] visibility -> {self.visibility}")
                self.draw()

            elif key == ord('S') or key == ord('s'):
                self.save_current_frame_annotation()
                self.draw()

            elif key == ord('n'):
                # save current slot then advance
                self.save_current_frame_annotation()
                if self.current_sample_idx + 1 < len(self.sampled_indices):
                    self.current_sample_idx += 1
                    self._load_frame_at(self.sampled_indices[self.current_sample_idx])
                    self._init_from_existing()
                    try:
                        self._update_window_size()
                    except Exception:
                        pass
                    self.draw()
                else:
                    print("[INFO] Last sampled frame for this video. Press N to save+next video or n to re-save.")

            elif key == ord('N'):
                self.save_current_frame_annotation()
                break

            else:
                continue
        cv2.destroyWindow(self.window_name)

# -------------------------
# Video iteration helper
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

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Dual-ROI multi-frame annotator with locked canonical size at save")
    parser.add_argument("input", nargs="?", default=None, help="Video file or directory of videos (optional)")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Canonical ROI width (pixels)")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Canonical ROI height (pixels)")
    parser.add_argument("--frames-per-video", type=int, default=DEFAULT_FRAMES_PER_VIDEO,
                        help="Number of sampled frames to annotate per video")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output JSON file for ROIs")
    args = parser.parse_args()

    input_path = args.input or DEFAULT_VIDEOS_DIR
    out_path = Path(args.out)
    rois = load_rois(out_path)

    videos = list(iter_videos(input_path))
    if not videos:
        print(f"No videos found at: {input_path}")
        return

    print(f"Found {len(videos)} videos in {input_path}. Output: {out_path}")
    for vid in videos:
        print(f"\nAnnotating: {vid.name}")
        annot = DualROIAnnotatorLocked(vid, args.width, args.height, args.frames_per_video, rois, out_path)
        annot.run_interactive()
        # save after each video to ensure persisted slot map and ROIs are written
        save_rois(out_path, rois)

    print("\nAll done. ROIs saved to", out_path)

if __name__ == "__main__":
    main()

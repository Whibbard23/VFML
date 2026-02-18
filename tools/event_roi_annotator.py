"""
event_roi_annotator.py

Event-aligned ROI annotator.

Usage:
    python event_roi_annotator.py events.csv --videos-dir /path/to/videos --out event_rois.json

Controls (keystrokes):
  - Mouse drag: move the active rectangle
  - Tab or '1' / '2': switch active box (mouth=1, ues=2)
  - Arrow keys / WASD: nudge active box by 1 (space toggles 10)
  - +/- : increase/decrease active box size (temporary only)
  - v : cycle visibility (visible -> partial -> not_visible)
  - r : reject current frame and replace with a new non-black frame
  - u : undo last replacement for current slot
  - p : return to the last saved frame for this slot (if any)
  - b : go back one event (opens saved frame if present)
  - > / < : sequential next / previous frame (updates slot)
  - B : toggle bolus_present flag (visible on screen)
  - s : save current event's annotation (canonical size enforced)
  - n : save and advance to next event
  - N : save and advance to next video
  - q or ESC : quit (saves JSON automatically)
"""

import csv
import cv2
import json
import argparse
from pathlib import Path
import sys
import math
import random
from datetime import datetime

# -------------------------
# Defaults and thresholds
# -------------------------
DEFAULT_WIDTH = 128
DEFAULT_HEIGHT = 128
DEFAULT_OUT = "event_rois.json"
MIN_FRAME_SKIP = 10
BLACK_MEAN_THRESHOLD = 10.0
REPLACEMENT_MAX_TRIES = 500
DISPLAY_SCALE = 3.0
SCREEN_MARGIN = 80

VISIBILITY_STATES = ["visible", "partial", "not_visible"]

# -------------------------
# JSON helpers
# -------------------------
def load_json(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}

def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2))

# -------------------------
# Path resolution helpers
# -------------------------
def resolve_input_path(user_path: str) -> Path:
    """
    Resolve a user-supplied path robustly:
      1. If absolute and exists -> use it
      2. If relative and exists relative to cwd -> use it
      3. If script is inside tools/, try parent (repo root) / user_path
      4. Try script directory / user_path
      5. Return Path(user_path) (may not exist) so caller can handle
    """
    p = Path(user_path)
    if p.is_absolute() and p.exists():
        return p.resolve()
    # cwd relative
    cwd_candidate = Path.cwd() / user_path
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    # try relative to script parent (two levels up if in tools/)
    try:
        script_dir = Path(__file__).resolve().parent
    except Exception:
        script_dir = Path.cwd()
    # try repo root (parent of tools) then script dir
    repo_root = script_dir.parent
    cand = repo_root / user_path
    if cand.exists():
        return cand.resolve()
    cand2 = script_dir / user_path
    if cand2.exists():
        return cand2.resolve()
    # fallback: return the cwd candidate (even if not exists) so caller can show helpful error
    return Path(user_path)

# -------------------------
# Event loader
# -------------------------
def load_events(csv_path: Path, video_col="video", event_col="event_id", frame_col="frame_index"):
    events = []
    if not csv_path.exists():
        raise FileNotFoundError(f"Events CSV not found: {csv_path}")
    with open(csv_path, newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            if video_col not in r or event_col not in r or frame_col not in r:
                continue
            try:
                frame_idx = int(r[frame_col])
            except Exception:
                continue
            events.append({
                "video": r[video_col],
                "event_id": r[event_col],
                "frame_index": frame_idx,
                "meta": r
            })
    # group by video, preserve order
    events_by_video = {}
    for e in events:
        events_by_video.setdefault(e["video"], []).append(e)
    return events_by_video

# -------------------------
# Event annotator
# -------------------------
class EventAnnotator:
    def __init__(self, video_path: Path, events: list, rois_dict: dict, out_path: Path,
                 canonical_w=DEFAULT_WIDTH, canonical_h=DEFAULT_HEIGHT):
        self.video_path = video_path
        self.events = events  # list of event dicts for this video
        self.rois = rois_dict
        self.out_path = Path(out_path)
        self.canonical_w = int(canonical_w)
        self.canonical_h = int(canonical_h)

        self.window_name = f"Event ROI - {self.video_path.name}"
        self.dragging = False
        self.offset = (0, 0)
        self.nudge = 1

        self.frame = None
        self.frame_display = None
        self.total_frames = 0

        # per-event slot mapping: list of frame indices (one per event)
        self.sampled_indices = []
        self.current_event_idx = 0

        # per-slot history for undo
        self.slot_history = {}

        # last saved frame per slot (event)
        self.last_saved_frame = {}

        # two boxes
        self.boxes = {
            "mouth": {"x": 0, "y": 0, "w": self.canonical_w, "h": self.canonical_h},
            "ues": {"x": 0, "y": 0, "w": self.canonical_w, "h": self.canonical_h}
        }
        self.active_box = "mouth"
        self.visibility = "visible"

        # bolus flag per slot
        self.bolus_present = {}

        # prepare video and sampling
        self._open_video()
        self._load_persisted_event_slot_map()
        self._init_slots()
        self._load_current_frame_and_state()

    # -------------------------
    # Video helpers
    # -------------------------
    def _open_video(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.release()

    def _frame_mean(self, idx):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            cap.release()
            return None
        idx = max(0, min(idx, max(0, self.total_frames - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        cap.release()
        if not ret or f is None:
            return None
        if len(f.shape) == 3 and f.shape[2] == 3:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            return float(gray.mean())
        return float(f.mean())

    def _find_replacement(self):
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
            m = self._frame_mean(idx)
            if m is None or m <= BLACK_MEAN_THRESHOLD:
                tried.add(idx)
                continue
            return idx
        return None

    # -------------------------
    # Persistence: event slot map
    # -------------------------
    def _load_persisted_event_slot_map(self):
        # __event_slot_map__ : { video_name: { event_id: frame_index, ... } }
        slot_map = self.rois.get("__event_slot_map__", {})
        self.persisted_map = slot_map.get(self.video_path.name, {}) if isinstance(slot_map, dict) else {}

    def _persist_event_slot_map(self):
        try:
            if "__event_slot_map__" not in self.rois or not isinstance(self.rois["__event_slot_map__"], dict):
                self.rois["__event_slot_map__"] = {}
            # build mapping event_id -> frame for this video
            mapping = {}
            for i, ev in enumerate(self.events):
                mapping[ev["event_id"]] = int(self.sampled_indices[i])
            self.rois["__event_slot_map__"][self.video_path.name] = mapping
            save_json(self.out_path, self.rois)
        except Exception:
            pass

    # -------------------------
    # Slot initialization
    # -------------------------
    def _init_slots(self):
        # initialize sampled_indices from persisted map or event frame_index
        self.sampled_indices = []
        for ev in self.events:
            eid = ev["event_id"]
            if eid in self.persisted_map:
                idx = int(self.persisted_map[eid])
            else:
                idx = int(ev["frame_index"])
                if idx < MIN_FRAME_SKIP and self.total_frames > MIN_FRAME_SKIP:
                    idx = MIN_FRAME_SKIP
            self.sampled_indices.append(idx)
            self.slot_history.setdefault(len(self.sampled_indices)-1, [])
            # load bolus flag if present in rois for this event
            existing = self._find_saved_entry_for_event(ev["event_id"])
            if existing:
                self.bolus_present[len(self.sampled_indices)-1] = bool(existing.get("bolus_present", False))
                self.last_saved_frame[len(self.sampled_indices)-1] = int(existing.get("frame_index"))
            else:
                self.bolus_present[len(self.sampled_indices)-1] = False

    def _find_saved_entry_for_event(self, event_id):
        key = self.video_path.name
        if key in self.rois:
            for e in self.rois.get(key, []):
                if str(e.get("event_id")) == str(event_id):
                    return e
        return None

    # -------------------------
    # Frame loading and UI init
    # -------------------------
    def _load_current_frame_and_state(self):
        idx = self.sampled_indices[self.current_event_idx]
        self._load_frame_at(idx)
        self._init_from_saved()

    def _load_frame_at(self, idx):
        idx = int(idx)
        if idx < MIN_FRAME_SKIP and self.total_frames > MIN_FRAME_SKIP:
            idx = MIN_FRAME_SKIP
        cap = cv2.VideoCapture(str(self.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(idx, self.total_frames - 1)))
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            # fallback: try first non-black after MIN_FRAME_SKIP
            found = None
            for i in range(MIN_FRAME_SKIP, min(self.total_frames, MIN_FRAME_SKIP + 200)):
                cap = cv2.VideoCapture(str(self.video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    found = i
                    break
            if found is None:
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
        self._update_window_size()

    def _init_from_saved(self):
        # load saved ROIs for current event frame if present
        key = self.video_path.name
        cur_frame = int(self.sampled_indices[self.current_event_idx])
        if key in self.rois:
            for e in self.rois.get(key, []):
                if int(e.get("frame_index")) == cur_frame and str(e.get("event_id")) == str(self.events[self.current_event_idx]["event_id"]):
                    if "mouth" in e:
                        x, y, w, h = e["mouth"]
                        self.boxes["mouth"].update({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
                    if "ues" in e:
                        x, y, w, h = e["ues"]
                        self.boxes["ues"].update({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
                    if "visibility" in e:
                        self.visibility = e.get("visibility", self.visibility)
                    self.bolus_present[self.current_event_idx] = bool(e.get("bolus_present", False))
                    self.last_saved_frame[self.current_event_idx] = cur_frame
                    break

    # -------------------------
    # Drawing and UI
    # -------------------------
    def _get_screen_size(self):
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
        screen_w, screen_h = self._get_screen_size()
        target_w = int(frame_w * DISPLAY_SCALE)
        target_h = int(frame_h * DISPLAY_SCALE)
        max_w = max(200, screen_w - SCREEN_MARGIN)
        max_h = max(200, screen_h - SCREEN_MARGIN)
        if target_w <= max_w and target_h <= max_h:
            return target_w, target_h
        scale_w = max_w / frame_w
        scale_h = max_h / frame_h
        scale = min(scale_w, scale_h)
        return max(200, int(frame_w * scale)), max(200, int(frame_h * scale))

    def _update_window_size(self):
        if self.frame is None:
            return
        h_img, w_img = self.frame.shape[:2]
        disp_w, disp_h = self._compute_display_size(w_img, h_img)
        try:
            cv2.resizeWindow(self.window_name, disp_w, disp_h)
        except Exception:
            pass

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

        cur_frame = self.sampled_indices[self.current_event_idx]
        ev = self.events[self.current_event_idx]
        bolus = self.bolus_present.get(self.current_event_idx, False)
        bolus_txt = "BOLUS: PRESENT" if bolus else "BOLUS: ABSENT"
        bolus_color = (0, 200, 255) if bolus else (180, 180, 180)

        txt = f"{self.video_path.name}  event={ev['event_id']}  frame={cur_frame}/{self.total_frames-1}  vis={self.visibility}  active={self.active_box}"
        cv2.putText(self.frame_display, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(self.frame_display, bolus_txt, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bolus_color, 3)
        cv2.putText(self.frame_display, "Tab/1/2 switch | +/- resize | v visibility | r reject | u undo | p saved | b back | < > step | B bolus | n next | N next video | s save | q quit",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        cv2.imshow(self.window_name, self.frame_display)

    # -------------------------
    # Mouse callback
    # -------------------------
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

    # -------------------------
    # Save annotation for current event
    # -------------------------
    def save_current_event(self):
        key = self.video_path.name
        if key not in self.rois:
            self.rois[key] = []
        cur_frame = int(self.sampled_indices[self.current_event_idx])
        mouth_box = dict(self.boxes["mouth"])
        ues_box = dict(self.boxes["ues"])
        # enforce canonical and center
        for box in (mouth_box, ues_box):
            cx = box["x"] + box["w"] // 2
            cy = box["y"] + box["h"] // 2
            box["w"] = self.canonical_w
            box["h"] = self.canonical_h
            box["x"] = int(cx - box["w"] // 2)
            box["y"] = int(cy - box["h"] // 2)
            self._clamp_box(box)

        entry = {
            "event_id": str(self.events[self.current_event_idx]["event_id"]),
            "frame_index": int(cur_frame),
            "mouth": [int(mouth_box["x"]), int(mouth_box["y"]), int(mouth_box["w"]), int(mouth_box["h"])],
            "ues": [int(ues_box["x"]), int(ues_box["y"]), int(ues_box["w"]), int(ues_box["h"])],
            "visibility": self.visibility,
            "bolus_present": bool(self.bolus_present.get(self.current_event_idx, False)),
            "annotator": "annotator",  # change if you want to record user
            "saved_at": datetime.utcnow().isoformat() + "Z"
        }

        entries = self.rois.get(key, [])
        replaced = False
        for i, e in enumerate(entries):
            if str(e.get("event_id")) == entry["event_id"]:
                entries[i] = entry
                replaced = True
                break
        if not replaced:
            entries.append(entry)
        entries = sorted(entries, key=lambda x: (str(x.get("event_id")), int(x.get("frame_index", 0))))
        self.rois[key] = entries

        # record last saved frame for this slot
        self.last_saved_frame[self.current_event_idx] = int(cur_frame)

        # persist slot map as well
        self._persist_event_slot_map()
        print(f"[SAVED] {key} event {entry['event_id']} frame {cur_frame} bolus={entry['bolus_present']}")

    # -------------------------
    # Main interactive loop
    # -------------------------
    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
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
            elif key in (81,):  # left arrow
                self.boxes[self.active_box]["x"] -= self.nudge
                self._clamp_box(self.boxes[self.active_box])
                self.draw()
            elif key in (83,):  # right arrow
                self.boxes[self.active_box]["x"] += self.nudge
                self._clamp_box(self.boxes[self.active_box])
                self.draw()
            elif key in (82,):  # up arrow
                self.boxes[self.active_box]["y"] -= self.nudge
                self._clamp_box(self.boxes[self.active_box])
                self.draw()
            elif key in (84,):  # down arrow
                self.boxes[self.active_box]["y"] += self.nudge
                self._clamp_box(self.boxes[self.active_box])
                self.draw()

            # reject current frame and replace with a new non-black frame
            elif key == ord('r'):
                slot = self.current_event_idx
                old_idx = self.sampled_indices[slot]
                new_idx = self._find_replacement()
                if new_idx is not None:
                    self.slot_history.setdefault(slot, []).append(old_idx)
                    self.sampled_indices[slot] = new_idx
                    # persist mapping
                    self._persist_event_slot_map()
                    print(f"[INFO] Replaced slot {slot} frame {old_idx} -> {new_idx}")
                    self._load_frame_at(new_idx)
                    self._init_from_saved()
                    self.draw()
                else:
                    print("[WARN] No replacement found")

            # undo replacement
            elif key == ord('u'):
                slot = self.current_event_idx
                history = self.slot_history.get(slot, [])
                if history:
                    prev = history.pop()
                    cur = self.sampled_indices[slot]
                    self.sampled_indices[slot] = prev
                    self._persist_event_slot_map()
                    print(f"[INFO] Reverted slot {slot} from {cur} back to {prev}")
                    self._load_frame_at(prev)
                    self._init_from_saved()
                    self.draw()
                else:
                    print("[INFO] No history to undo")

            # return to last saved frame for this slot
            elif key == ord('p'):
                slot = self.current_event_idx
                saved = self.last_saved_frame.get(slot)
                if saved is not None:
                    cur = self.sampled_indices[slot]
                    if saved != cur:
                        self.slot_history.setdefault(slot, []).append(cur)
                        self.sampled_indices[slot] = saved
                        self._persist_event_slot_map()
                        print(f"[INFO] Loaded last saved frame {saved} into slot {slot} (was {cur})")
                        self._load_frame_at(saved)
                        self._init_from_saved()
                        self.draw()
                    else:
                        print("[INFO] Already on last saved frame")
                else:
                    print("[INFO] No saved frame for this slot")

            # sequential next frame (display + update slot)
            elif key == ord('>'):
                slot = self.current_event_idx
                cur = int(self.sampled_indices[slot])
                nxt = min(cur + 1, self.total_frames - 1)
                self.slot_history.setdefault(slot, []).append(cur)
                self.sampled_indices[slot] = nxt
                self._persist_event_slot_map()
                print(f"[INFO] Sequential next: {cur} -> {nxt}")
                self._load_frame_at(nxt)
                self._init_from_saved()
                self.draw()

            # sequential previous frame (display + update slot)
            elif key == ord('<'):
                slot = self.current_event_idx
                cur = int(self.sampled_indices[slot])
                prev = max(cur - 1, MIN_FRAME_SKIP)
                self.slot_history.setdefault(slot, []).append(cur)
                self.sampled_indices[slot] = prev
                self._persist_event_slot_map()
                print(f"[INFO] Sequential prev: {cur} -> {prev}")
                self._load_frame_at(prev)
                self._init_from_saved()
                self.draw()

            # toggle bolus flag
            elif key == ord('B'):
                slot = self.current_event_idx
                self.bolus_present[slot] = not bool(self.bolus_present.get(slot, False))
                print(f"[INFO] bolus_present -> {self.bolus_present[slot]}")
                self.draw()

            # go back one event slot
            elif key == ord('b'):
                if self.current_event_idx > 0:
                    prev_slot = self.current_event_idx - 1
                    saved = self.last_saved_frame.get(prev_slot)
                    if saved is not None:
                        self.sampled_indices[prev_slot] = int(saved)
                        print(f"[INFO] Going back to slot {prev_slot}, opening saved frame {saved}")
                    else:
                        print(f"[INFO] Going back to slot {prev_slot}, opening sampled frame {self.sampled_indices[prev_slot]}")
                    self.current_event_idx = prev_slot
                    self._load_frame_at(self.sampled_indices[self.current_event_idx])
                    self._init_from_saved()
                    self.draw()
                else:
                    print("[INFO] Already at first event")

            elif key == ord('v'):
                idx = VISIBILITY_STATES.index(self.visibility)
                idx = (idx + 1) % len(VISIBILITY_STATES)
                self.visibility = VISIBILITY_STATES[idx]
                print(f"[INFO] visibility -> {self.visibility}")
                self.draw()

            elif key == ord('S') or key == ord('s'):
                self.save_current_event()
                self.draw()

            elif key == ord('n'):
                # save and advance to next event
                self.save_current_event()
                if self.current_event_idx + 1 < len(self.sampled_indices):
                    self.current_event_idx += 1
                    self._load_frame_at(self.sampled_indices[self.current_event_idx])
                    self._init_from_saved()
                    self.draw()
                else:
                    print("[INFO] Last event for this video")

            elif key == ord('N'):
                self.save_current_event()
                break

            else:
                continue

        cv2.destroyWindow(self.window_name)

# -------------------------
# Main CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Event-aligned ROI annotator")
    parser.add_argument("events_csv", help="CSV of cleaned events (columns: video,event_id,frame_index)")
    parser.add_argument("--videos-dir", default=".", help="Directory containing videos (or absolute paths in CSV)")
    parser.add_argument("--video-col", default="video", help="CSV column name for video path")
    parser.add_argument("--event-col", default="event_id", help="CSV column name for event id")
    parser.add_argument("--frame-col", default="frame_index", help="CSV column name for frame index")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Canonical ROI width")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Canonical ROI height")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output JSON file for ROIs")
    args = parser.parse_args()

    # Resolve events CSV robustly (script may live in tools/)
    events_csv_path = resolve_input_path(args.events_csv)
    if not events_csv_path.exists():
        # try relative to provided videos-dir (common case)
        alt = Path(args.videos_dir) / args.events_csv
        if alt.exists():
            events_csv_path = alt.resolve()
    if not events_csv_path.exists():
        print(f"[ERROR] Events CSV not found: {args.events_csv}")
        print("Tried:", args.events_csv, "cwd:", Path.cwd(), "script parent:", Path(__file__).resolve().parent)
        return

    events_by_video = load_events(events_csv_path, video_col=args.video_col, event_col=args.event_col, frame_col=args.frame_col)
    if not events_by_video:
        print("No events loaded. Check CSV and column names.")
        return

    out_path = Path(args.out)
    rois = load_json(out_path)

    videos = sorted(events_by_video.keys())
    for vname in videos:
        # resolve video path: if vname is absolute or exists, use it; else join with videos-dir
        vpath = Path(vname)
        if not vpath.exists():
            candidate = Path(args.videos_dir) / vname
            if candidate.exists():
                vpath = candidate
            else:
                # try common extensions in videos-dir
                found = None
                for ext in (".mp4", ".avi", ".mov", ".mkv"):
                    cand = Path(args.videos_dir) / (vname + ext)
                    if cand.exists():
                        found = cand
                        break
                if found:
                    vpath = found
                else:
                    # try resolving relative to CSV location (useful if CSV contains relative paths)
                    csv_parent = events_csv_path.parent
                    cand2 = csv_parent / vname
                    if cand2.exists():
                        vpath = cand2
                    else:
                        print(f"[WARN] Video not found for {vname}, skipping.")
                        continue

        evs = events_by_video[vname]
        print(f"\nAnnotating video {vpath} with {len(evs)} events -> output {out_path}")
        annot = EventAnnotator(vpath, evs, rois, out_path, canonical_w=args.width, canonical_h=args.height)
        annot.run()
        # save after each video
        save_json(out_path, rois)

    print("\nAll done. ROIs saved to", out_path)

if __name__ == "__main__":
    main()

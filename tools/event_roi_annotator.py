"""
event_roi_annotator.py
event_roi_annotator.py

Event-aligned ROI annotator.

Usage:
    python event_roi_annotator.py events.csv --videos-dir /path/to/videos --out event_rois.json

"""
import csv
import cv2
import json
from pathlib import Path
import sys
import math
import random
from datetime import datetime

# -------------------------
# Hardcoded paths (edit if needed)
# -------------------------
# Script is expected to live in tools/; these defaults point to repo root siblings.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

EVENTS_CSV = Path(r"C:\Users\Connor Lab\Desktop\VFML\event_csvs\cleaned_events.csv")
VIDEOS_DIR = Path(r"\\research.drive.wisc.edu\npconnor\ADStudy\VF AD Blinded\Early Tongue Training")
OUT_JSON = REPO_ROOT / "event_rois.json"


print("EVENTS_CSV:", EVENTS_CSV, "exists:", EVENTS_CSV.exists(), "is_file:", EVENTS_CSV.is_file())
print("VIDEOS_DIR:", VIDEOS_DIR, "exists:", VIDEOS_DIR.exists(), "is_dir:", VIDEOS_DIR.is_dir())
print("OUT_JSON:", OUT_JSON, "parent exists:", OUT_JSON.parent.exists())

# -------------------------
# Defaults and thresholds
# -------------------------
DEFAULT_WIDTH = 128
DEFAULT_HEIGHT = 128
MIN_FRAME_SKIP = 5
BLACK_MEAN_THRESHOLD = 10.0
REPLACEMENT_MAX_TRIES = 500
DISPLAY_SCALE = 5.0
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
# Event loader
# -------------------------
def load_events(csv_path: Path):
    """
    Load cleaned_events.csv where each row contains:
        video, before_onset, touch_ues, leave_ues

    Expand each row into THREE events:
        event_id = before_onset / touch_ues / leave_ues
        frame_index = integer frame number
    """
    events_by_video = {}

    with open(csv_path, newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            video = r["video"]

            # Expand each event type
            for event_name in ("before_onset", "touch_ues", "leave_ues"):
                raw_val = r.get(event_name, "")
                if raw_val is None or raw_val == "":
                    continue
                try:
                    frame_idx = int(raw_val)
                except:
                    continue

                events_by_video.setdefault(video, []).append({
                    "video": video,
                    "event_id": event_name,
                    "frame_index": frame_idx,
                    "meta": r
                })


    # Sort events chronologically for each video
        for v in events_by_video:
            events_by_video[v] = sorted(events_by_video[v], key=lambda e: e["frame_index"])

        return events_by_video



# -------------------------
# Event annotator
# -------------------------
class EventAnnotator:
    def __init__(self, video_path: Path, events: list, rois_dict: dict, out_path: Path,
                 canonical_w=DEFAULT_WIDTH, canonical_h=DEFAULT_HEIGHT):
        self.video_path = video_path
        self.events = events
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

        self.sampled_indices = []
        self.current_event_idx = 0
        self.slot_history = {}
        self.last_saved_frame = {}

        self.boxes = {
            "mouth": {"x": 0, "y": 0, "w": self.canonical_w, "h": self.canonical_h},
            "ues": {"x": 0, "y": 0, "w": self.canonical_w, "h": self.canonical_h}
        }
        self.active_box = "mouth"
        self.visibility = "visible"
        self.bolus_mouth_present = {}
        self.bolus_ues_present = {}

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
        slot_map = self.rois.get("__event_slot_map__", {})
        self.persisted_map = slot_map.get(self.video_path.name, {}) if isinstance(slot_map, dict) else {}

    def _persist_event_slot_map(self):
        try:
            if "__event_slot_map__" not in self.rois or not isinstance(self.rois["__event_slot_map__"], dict):
                self.rois["__event_slot_map__"] = {}
            mapping = {}
            for i, ev in enumerate(self.events):
                mapping[str(i)] = int(self.sampled_indices[i])
            self.rois["__event_slot_map__"][self.video_path.name] = mapping
            save_json(self.out_path, self.rois)
        except Exception:
            pass

    # -------------------------
    # Slot initialization
    # -------------------------
    def _init_slots(self):
        slot_key = str(len(self.sampled_indices)-1)
        if slot_key in self.persisted_map:
            idx = int(self.persisted_map[slot_key])
        self.sampled_indices = []
        for ev in self.events:
            eid = ev["event_id"]
            if str(len(self.sampled_indices)) in self.persisted_map:
                idx = int(self.persisted_map[str(len(self.sampled_indices))])

            else:
                idx = int(ev["frame_index"])
                if idx < MIN_FRAME_SKIP and self.total_frames > MIN_FRAME_SKIP:
                    idx = MIN_FRAME_SKIP
            self.sampled_indices.append(idx)
            self.slot_history.setdefault(len(self.sampled_indices)-1, [])
            slot = len(self.sampled_indices)-1

            existing = self._find_saved_entry_for_event(ev["event_id"], idx)

            if existing:
                self.bolus_mouth_present[slot] = bool(existing.get("bolus_mouth_present", False))
                self.bolus_ues_present[slot]   = bool(existing.get("bolus_ues_present", False))
                self.last_saved_frame[slot]    = int(existing.get("frame_index"))
            else:
                # Apply event-type dependent defaults for unsaved slots
                self._apply_event_defaults(ev["event_id"], slot)



    def _find_saved_entry_for_event(self, event_id, frame_index=None):
        """
        Return the saved entry for the given event_id and optional frame_index
        for the current video. If frame_index is provided, require both to match.
        """
        key = self.video_path.name
        entries = self.rois.get(key, [])
        for e in entries:
            try:
                if frame_index is not None:
                    if int(e.get("frame_index", -1)) == int(frame_index) and str(e.get("event_id")).strip() == str(event_id).strip():
                        return e
                else:
                    if str(e.get("event_id")).strip() == str(event_id).strip():
                        return e
            except Exception:
                continue
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
        """
        Restore boxes, visibility, and bolus flags from saved JSON entry for the
        current slot/frame. If no saved entry exists, do not write anything;
        caller may apply in-memory defaults separately.
        """
        key = self.video_path.name
        cur_frame = int(self.sampled_indices[self.current_event_idx])
        found_saved = False

        if key in self.rois:
            for e in self.rois.get(key, []):
                # match both frame index and event id
                if int(e.get("frame_index")) == cur_frame and str(e.get("event_id")) == str(self.events[self.current_event_idx]["event_id"]):
                    if "mouth" in e:
                        x, y, w, h = e["mouth"]
                        self.boxes["mouth"].update({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
                    if "ues" in e:
                        x, y, w, h = e["ues"]
                        self.boxes["ues"].update({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
                    if "visibility" in e:
                        self.visibility = e.get("visibility", self.visibility)

                    # Load saved bolus flags (if present)
                    self.bolus_mouth_present[self.current_event_idx] = bool(e.get("bolus_mouth_present", False))
                    self.bolus_ues_present[self.current_event_idx]   = bool(e.get("bolus_ues_present", False))

                    # Mark this slot as having a saved frame
                    self.last_saved_frame[self.current_event_idx] = cur_frame
                    found_saved = True
                    break

        # If no saved entry was found, do not write to disk here.
        # Caller may apply in-memory defaults if desired:
        if not found_saved:
            # ensure keys exist so UI code can read them safely
            self.bolus_mouth_present.setdefault(self.current_event_idx, False)
            self.bolus_ues_present.setdefault(self.current_event_idx, False)


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
        slot = self.current_event_idx
        mouth_flag = self.bolus_mouth_present.get(slot, False)
        ues_flag = self.bolus_ues_present.get(slot, False)

        # Color-coded indicators
        mouth_color = (0, 255, 0) # green
        ues_color = (255, 0, 0) # blue

        bolus_txt = f"Mouth bolus: {'YES' if mouth_flag else 'NO'}    |     UES bolus: {'YES' if ues_flag else 'NO'}"

        txt = f"{self.video_path.name}  event={ev['event_id']}  frame={cur_frame}/{self.total_frames-1}  vis={self.visibility}  active={self.active_box}"
        cv2.putText(self.frame_display, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        slot = self.current_event_idx
        mouth_flag = self.bolus_mouth_present.get(slot, False)
        ues_flag   = self.bolus_ues_present.get(slot, False)

        # Colors
        mouth_color = (0, 255, 0)   # green
        ues_color   = (255, 0, 0)   # blue

        # Draw color-coded bolus indicators
        cv2.putText(self.frame_display,
            f"Mouth bolus: {'YES' if mouth_flag else 'NO'}",
            (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mouth_color, 3)

        cv2.putText(self.frame_display,
            f"UES bolus: {'YES' if ues_flag else 'NO'}",
            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ues_color, 3)

        cv2.putText(self.frame_display, "Tab/1/2 switch | +/- resize | v visibility | r reject | u undo | p saved | b back | < > step | B bolus | n next | N next video | s save | q quit",
                    (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        cv2.imshow(self.window_name, self.frame_display)

    def _apply_event_defaults(self, event_id: str, slot: int):
        # Normalize event id
        eid = str(event_id).lower()
        if eid == "before_onset":
            self.bolus_mouth_present[slot] = True
            self.bolus_ues_present[slot] = False
        elif eid in ("touch_ues", "leave_ues"):
            self.bolus_mouth_present[slot] = False
            self.bolus_ues_present[slot] = True
        else:
            self.bolus_mouth_present[slot] = False
            self.bolus_ues_present[slot] = False



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
            "bolus_mouth_present": bool(self.bolus_mouth_present.get(self.current_event_idx, False)),
            "bolus_ues_present":   bool(self.bolus_ues_present.get(self.current_event_idx, False)),
            "annotator": "annotator",
            "saved_at": datetime.utcnow().isoformat() + "Z"
        }

        entries = self.rois.get(key, [])
        replaced = False
        for i, e in enumerate(entries):
            try:
                if str(e.get("event_id")).strip() == entry["event_id"].strip() and int(e.get("frame_index", -1)) == int(entry["frame_index"]):
                    entries[i] = entry
                    replaced = True
                    break
            except Exception:
                continue
        if not replaced:
            entries.append(entry)

        # Keep entries deterministic and easy to inspect
        entries = sorted(entries, key=lambda x: (str(x.get("event_id")), int(x.get("frame_index", 0))))
        self.rois[key] = entries

        # Mark this slot as having a saved frame and persist the slot map
        self.last_saved_frame[self.current_event_idx] = int(cur_frame)
        self._persist_event_slot_map()

        print(
            f"[SAVED] {key} event {entry['event_id']} frame {cur_frame} "
            f"mouth_bolus={entry['bolus_mouth_present']} "
            f"ues_bolus={entry['bolus_ues_present']}"
        )



    # -------------------------
    # Main interactive loop
    # -------------------------
    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        # Ensure the window is scaled based on the *already loaded* first frame
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
            elif key in (81,):
                self.boxes[self.active_box]["x"] -= self.nudge
                self._clamp_box(self.boxes[self.active_box])
                self.draw()
            elif key in (83,):
                self.boxes[self.active_box]["x"] += self.nudge
                self._clamp_box(self.boxes[self.active_box])
                self.draw()
            elif key in (82,):
                self.boxes[self.active_box]["y"] -= self.nudge
                self._clamp_box(self.boxes[self.active_box])
                self.draw()
            elif key in (84,):
                self.boxes[self.active_box]["y"] += self.nudge
                self._clamp_box(self.boxes[self.active_box])
                self.draw()

            elif key == ord('r'):
                slot = self.current_event_idx
                old_idx = self.sampled_indices[slot]
                new_idx = self._find_replacement()
                if new_idx is not None:
                    self.slot_history.setdefault(slot, []).append(old_idx)
                    self.sampled_indices[slot] = new_idx
                    self._persist_event_slot_map()
                    print(f"[INFO] Replaced slot {slot} frame {old_idx} -> {new_idx}")
                    self._load_frame_at(new_idx)
                    self._init_from_saved()
                    self.draw()
                else:
                    print("[WARN] No replacement found")

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

            # Toggle mouth bolus
            elif key == ord('f'):
                slot = self.current_event_idx
                self.bolus_mouth_present[slot] = not self.bolus_mouth_present.get(slot, False)
                print(f"[INFO] bolus_mouth_present -> {self.bolus_mouth_present[slot]}")
                self.draw()

            # Toggle UES bolus
            elif key == ord('g'):
                slot = self.current_event_idx
                self.bolus_ues_present[slot] = not self.bolus_ues_present.get(slot, False)
                print(f"[INFO] bolus_ues_present -> {self.bolus_ues_present[slot]}")
                self.draw()


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
# Main (no CLI args)
# -------------------------
def main():
    print("Event ROI annotator (no args).")
    print(f"Events CSV: {EVENTS_CSV}")
    print(f"Videos dir: {VIDEOS_DIR}")
    print(f"Output JSON: {OUT_JSON}")

    if not EVENTS_CSV.exists():
        print(f"[ERROR] Events CSV not found at {EVENTS_CSV}")
        return
    events_by_video = load_events(EVENTS_CSV)
    if not events_by_video:
        print("[ERROR] No events loaded. Check CSV format and columns (video,event_id,frame_index).")
        return

    rois = load_json(OUT_JSON)
    # print("[DEBUG] loaded rois keys:", list(rois.keys())[:20])

    videos = sorted(events_by_video.keys())
    for vname in videos:
        # resolve video path
        vpath = Path(vname)
        if not vpath.exists():
            candidate = VIDEOS_DIR / vname
            if candidate.exists():
                vpath = candidate
            else:
                found = None
                for ext in (".mp4", ".avi", ".mov", ".mkv"):
                    cand = VIDEOS_DIR / (vname + ext)
                    if cand.exists():
                        found = cand
                        break
                if found:
                    vpath = found
                else:
                    csv_parent = EVENTS_CSV.parent
                    cand2 = csv_parent / vname
                    if cand2.exists():
                        vpath = cand2
                    else:
                        print(f"[WARN] Video not found for {vname}, skipping.")
                        continue

        evs = events_by_video[vname]
        print(f"\nAnnotating video {vpath} with {len(evs)} events -> output {OUT_JSON}")
        # print("[DEBUG] rois for video:", rois.get(vname))
        annot = EventAnnotator(vpath, evs, rois, OUT_JSON, canonical_w=DEFAULT_WIDTH, canonical_h=DEFAULT_HEIGHT)
        annot.run()
        save_json(OUT_JSON, rois)

    print("\nAll done. ROIs saved to", OUT_JSON)

if __name__ == "__main__":
    main()

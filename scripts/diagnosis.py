# diagnostics.py
from pathlib import Path
import json

VIDEOS_DIR = Path(r"\\research.drive.wisc.edu\npconnor\ADStudy\VF AD Blinded\Early Tongue Training")
EVENTS_JSON = Path("event_rois.json")

def load_events(p):
    d = json.loads(p.read_text())
    return [k for k in d.keys() if k != "__event_slot_map__"]

def find_match(key, videos_dir):
    p = Path(key)
    if p.exists():
        return p
    # exact under videos_dir if key includes extension
    if p.suffix:
        cand = videos_dir / key
        if cand.exists():
            return cand
    stem = p.stem
    for ext in (".avi", ".mp4", ".mov", ".mkv"):
        cand = videos_dir / (stem + ext)
        if cand.exists():
            return cand
    # recursive fallback
    for f in videos_dir.rglob("*"):
        if not f.is_file(): continue
        if f.stem.lower().startswith(stem.lower()):
            return f
    return None

if __name__ == "__main__":
    keys = load_events(EVENTS_JSON)
    missing = []
    mapped = {}
    for k in keys:
        m = find_match(k, VIDEOS_DIR)
        if m:
            mapped[k] = str(m)
        else:
            missing.append(k)
    print("Mapped keys (sample 20):")
    for i,(k,v) in enumerate(mapped.items()):
        print(f"{k} -> {v}")
        if i>=19: break
    print()
    print(f"Total keys: {len(keys)}  Mapped: {len(mapped)}  Missing: {len(missing)}")
    if missing:
        print("Missing keys (sample 20):")
        for k in missing[:20]:
            print(k)

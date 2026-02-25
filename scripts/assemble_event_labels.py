#!/usr/bin/env python3
# scripts/assemble_event_labels.py
"""
Assemble event labels and match them to ROI crop images.

This variant:
- Reads a cleaned input CSV
- Scans the crops root for all video folders (e.g., "AD128.avi") and loads metadata.json when present.
- Produces per-video CSVs for every video folder found under the crops root (event_csvs/per_video/<video>_events.csv).
- Produces a master long CSV event_csvs/<exp>_events_long.csv that merges:
    * events from the input CSV (source="input")
    * events discovered in crops/<video>/metadata.json (source="metadata")
  Duplicate (video,frame,event_type) rows are collapsed, preferring input CSV entries when both exist.
- Writes event_csvs/<exp>_summary.txt with counts and a note that the full .avi files referenced in metadata may include files not analyzed and not part of the dataset.
- Optional: append matched rows to data/manifests/manifest.csv.

Usage (from project root):
  & .\.venv\Scripts\python.exe scripts\assemble_event_labels.py \
      --input-csv event_csvs/cleaned_events.csv \
      --crops-dir crops \
      --out-dir event_csvs \
      --per-video-dir event_csvs/per_video \
      --exp assembly_1 \
      --fallback-window 2 \
      --update-manifest \
      --verbose
"""
from pathlib import Path
import argparse
import csv
import json
import sys
import re
from collections import defaultdict, OrderedDict
import datetime

# ---------- Utilities ----------

def resolve_paths(script_path: Path, crops_dir: Path | None):
    # Use the current working directory as the project root so relative paths behave as expected
    PROJECT_ROOT = Path.cwd().resolve()
    if crops_dir is None:
        preferred = PROJECT_ROOT / "data" / "crops"
        fallback = PROJECT_ROOT / "crops"
        crops_root = preferred if preferred.exists() else fallback
    else:
        crops_root = (crops_dir if crops_dir.is_absolute() else (PROJECT_ROOT / crops_dir)).resolve()
    return PROJECT_ROOT, crops_root


def read_csv_rows(path: Path, verbose=False):
    if verbose:
        print(f"Reading CSV: {path}")
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        for r in reader:
            rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k,v in r.items()})
    return rows, fieldnames

def is_int_like(s):
    try:
        int(float(s))
        return True
    except Exception:
        return False

def detect_format(fieldnames):
    fn = [f.lower() for f in (fieldnames or [])]
    has_video = any(x in fn for x in ("video","video_id","filename","file"))
    has_frame = any(x in fn for x in ("frame","frame_index","frame_idx"))
    has_event = any(x in fn for x in ("event","event_type","label","type"))
    known_events = {"before_onset","touch_ues","leave_ues"}
    wide_cols = known_events.intersection(set(fn))
    if has_video and has_frame and has_event:
        return "long"
    if has_video and wide_cols:
        return "wide"
    if has_video and has_frame:
        return "long"
    return "wide" if wide_cols else "long"

def normalize_rows_to_long(rows, fieldnames, verbose=False):
    fmt = detect_format(fieldnames)
    if verbose:
        print("Detected CSV format:", fmt)
    out = []
    if fmt == "long":
        video_col = next((c for c in fieldnames if c.lower() in ("video","video_id","filename","file")), None)
        frame_col = next((c for c in fieldnames if c.lower() in ("frame","frame_index","frame_idx")), None)
        event_col = next((c for c in fieldnames if c.lower() in ("event","event_type","label","type")), None)
        for c in fieldnames:
            lc = c.lower()
            if not video_col and "video" in lc:
                video_col = c
            if not frame_col and "frame" in lc:
                frame_col = c
            if not event_col and ("event" in lc or "type" in lc or "label" in lc):
                event_col = c
        for r in rows:
            vid = r.get(video_col,"") if video_col else ""
            frm = r.get(frame_col,"") if frame_col else ""
            ev = r.get(event_col,"") if event_col else ""
            if vid is None or str(vid).strip()=="" or frm is None or str(frm).strip()=="":
                continue
            try:
                fnum = int(float(frm))
            except Exception:
                continue
            out.append({"video":str(vid).strip(), "frame":int(fnum), "event_type":str(ev).strip().lower().replace(" ", "_"), "source":"input"})
    else:
        video_col = next((c for c in fieldnames if c.lower() in ("video","video_id","filename","file")), None)
        if not video_col:
            video_col = fieldnames[0]
        for r in rows:
            vid = r.get(video_col,"")
            if vid is None or str(vid).strip()=="":
                continue
            for c in fieldnames:
                if c == video_col:
                    continue
                lc = c.lower()
                if not re.search("[a-zA-Z]", c):
                    continue
                val = r.get(c,"")
                if val is None or str(val).strip()=="":
                    continue
                parts = re.split(r"[;|,]", str(val))
                for p in parts:
                    p = p.strip()
                    if p=="":
                        continue
                    if is_int_like(p):
                        out.append({"video":str(vid).strip(), "frame":int(float(p)), "event_type":c.strip().lower().replace(" ", "_"), "source":"input"})
                    else:
                        m = re.search(r"(\d+)", p)
                        if m:
                            out.append({"video":str(vid).strip(), "frame":int(m.group(1)), "event_type":c.strip().lower().replace(" ", "_"), "source":"input"})
    return out

# ---------- Metadata and crop helpers ----------

def load_metadata_for_video(video_folder: Path, verbose=False):
    """
    Load metadata.json under video_folder if present.
    Returns:
      - metadata_entries: list of dicts with keys: event_id, frame_index, mouth_path, ues_path, entry_index, visibility, weight
      - full_video_path: the 'video' field from metadata if present (string)
    """
    meta_path = video_folder / "metadata.json"
    entries = []
    full_video_path = None
    if not meta_path.exists():
        return entries, full_video_path
    try:
        with meta_path.open("r", encoding="utf-8", errors="ignore") as fh:
            meta = json.load(fh)
    except Exception as e:
        if verbose:
            print(f"Warning: failed to read metadata.json for {video_folder.name}: {e}")
        return entries, full_video_path
    full_video_path = meta.get("video")
    crops = meta.get("crops", [])
    for entry in crops:
        eid = entry.get("event_id")
        fidx = entry.get("frame_index")
        if eid is None or fidx is None:
            continue
        mouth_rel = entry.get("mouth_path") or entry.get("mouth") or ""
        ues_rel = entry.get("ues_path") or entry.get("ues") or ""
        entries.append({
            "event_id": str(eid).lower(),
            "frame_index": int(fidx),
            "mouth_path": (video_folder / mouth_rel).resolve() if mouth_rel else None,
            "ues_path": (video_folder / ues_rel).resolve() if ues_rel else None,
            "entry_index": entry.get("entry_index"),
            "visibility": entry.get("visibility"),
            "weight": entry.get("weight")
        })
    return entries, full_video_path

def build_frame_map_from_folder(video_folder: Path, verbose=False):
    """
    Build mapping frame_index -> list of files under video_folder (searches recursively).
    Extracts the first integer group from filenames as frame index.
    """
    frame_map = defaultdict(list)
    if not video_folder.exists():
        return frame_map
    for p in video_folder.rglob("*"):
        if p.is_file():
            m = re.search(r"(\d{1,7})", p.name)
            if m:
                try:
                    frm = int(m.group(1))
                    frame_map[frm].append(p.resolve())
                except Exception:
                    continue
    return frame_map

def match_event_using_metadata(entries, event_type, frame, prefer="mouth"):
    """
    Given metadata entries list, try to find an entry matching (event_type, frame).
    Returns path (Path) or None and a note.
    """
    # exact match on event_id and frame_index
    for e in entries:
        if e["event_id"] == event_type and e["frame_index"] == frame:
            path = e.get(f"{prefer}_path") or e.get("mouth_path") or e.get("ues_path")
            return path, "metadata_exact"
    # match by event_id only (take first)
    for e in entries:
        if e["event_id"] == event_type:
            path = e.get(f"{prefer}_path") or e.get("mouth_path") or e.get("ues_path")
            return path, "metadata_eventid"
    return None, "metadata_no_match"

def match_event_to_crop(video_folder: Path, entries, frame_map, event_type, frame, fallback_window=0, verbose=False):
    """
    Try to match event to a crop file:
      1) metadata entries (prefer mouth for mouth events, ues for ues events)
      2) exact frame in frame_map
      3) fallback ±N frames in frame_map
    Returns: (matched_path or None, matched_frame or None, distance or None, note, candidates_list)
    """
    prefer = "mouth" if "mouth" in event_type else "ues"
    if entries:
        p, note = match_event_using_metadata(entries, event_type, frame, prefer=prefer)
        if p:
            return p, frame, 0, note, [p]
    # try frame_map exact
    candidates = frame_map.get(frame, [])
    if candidates:
        return candidates[0], frame, 0, "file_exact", candidates
    # fallback search
    if fallback_window and frame_map:
        for d in range(1, fallback_window+1):
            for f_try in (frame - d, frame + d):
                if f_try in frame_map:
                    return frame_map[f_try][0], f_try, d, f"file_nearest_{d}", frame_map[f_try]
    return None, None, None, "no_match", []

# ---------- CSV writers ----------

def write_master_csv(out_path: Path, rows, fieldnames):
    with out_path.open("w", newline='', encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def write_per_video_csv(per_video_path: Path, rows):
    per_video_path.parent.mkdir(parents=True, exist_ok=True)
    with per_video_path.open("w", newline='', encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["frame","event_type","crop_path","matched","notes","source"])
        writer.writeheader()
        for r in sorted(rows, key=lambda x: x["frame"]):
            writer.writerow(r)

def append_to_manifest(manifest_path: Path, rows, verbose=False):
    cols = ["video","frame","event_type","crop_path","matched","notes","source"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        with manifest_path.open("a", newline='', encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            for r in rows:
                writer.writerow({k: r.get(k,"") for k in cols})
    else:
        with manifest_path.open("w", newline='', encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k,"") for k in cols})
    if verbose:
        print(f"Manifest updated at {manifest_path}")

# ---------- Main ----------

def parse_args():
    p = argparse.ArgumentParser(description="Assemble event labels and match to ROI crop images (project-specific)")
    p.add_argument("--input-csv", required=True, type=Path, help="Path to input CSV (wide or long). Example: event_csvs/cleaned_events.csv")
    p.add_argument("--crops-dir", type=Path, default=Path("crops"), help="Root crops directory (default: crops).")
    p.add_argument("--out-dir", type=Path, default=Path("event_csvs"), help="Output directory for master CSV and summary")
    p.add_argument("--per-video-dir", type=Path, default=Path("event_csvs/per_video"), help="Directory for per-video CSVs")
    p.add_argument("--exp", type=str, default="events", help="Experiment name used in output filenames")
    p.add_argument("--fallback-window", type=int, default=0, help="Search ±N frames for nearest crop if exact frame missing")
    p.add_argument("--update-manifest", action="store_true", help="Append matched rows to data/manifests/manifest.csv if present")
    p.add_argument("--verbose", action="store_true", help="Verbose output")
    return p.parse_args()

def main():
    args = parse_args()
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT, crops_root = resolve_paths(SCRIPT_DIR, args.crops_dir)
    input_csv = args.input_csv if args.input_csv.is_absolute() else (PROJECT_ROOT / args.input_csv)
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    per_video_dir = (PROJECT_ROOT / args.per_video_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    per_video_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print("Project root:", PROJECT_ROOT)
        print("Crops root:", crops_root)
        print("Input CSV:", input_csv)
        print("Output dir:", out_dir)
        print("Per-video dir:", per_video_dir)

    if not input_csv.exists():
        print("ERROR: input CSV not found:", input_csv, file=sys.stderr)
        sys.exit(2)

    # Read input CSV and normalize to long format (source="input")
    rows, fieldnames = read_csv_rows(input_csv, verbose=args.verbose)
    input_long = normalize_rows_to_long(rows, fieldnames, verbose=args.verbose)

    # Build a dict keyed by (video,frame,event_type) -> row (prefer input rows)
    merged = OrderedDict()
    for r in input_long:
        key = (r["video"], int(r["frame"]), r["event_type"])
        merged[key] = {
            "video": r["video"],
            "frame": int(r["frame"]),
            "event_type": r["event_type"],
            "crop_path": "",
            "matched": False,
            "notes": "",
            "source": "input"
        }

    # Scan crops_root for video folders and incorporate metadata entries
    video_folders = [p for p in sorted(crops_root.iterdir()) if p.is_dir()]
    metadata_video_paths = {}
    for vf in video_folders:
        video_name = vf.name  # e.g., "AD128.avi"
        entries, full_video_path = load_metadata_for_video(vf, verbose=args.verbose)
        metadata_video_paths[video_name] = full_video_path
        # add metadata entries to merged if not already present (source="metadata")
        for e in entries:
            key = (video_name, int(e["frame_index"]), e["event_id"])
            if key not in merged:
                merged[key] = {
                    "video": video_name,
                    "frame": int(e["frame_index"]),
                    "event_type": e["event_id"],
                    "crop_path": "",
                    "matched": False,
                    "notes": "from_metadata",
                    "source": "metadata"
                }

    # For any video folders that were not in input CSV or metadata, still create empty per-video CSVs
    # Now attempt to match each merged row to an actual crop file
    master_rows = []
    summary = {"total":0, "matched":0, "missing":0, "ambiguous":0, "per_video":{}, "videos_scanned": len(video_folders)}
    # Preload metadata and frame maps per video to avoid repeated IO
    metadata_cache = {}
    frame_map_cache = {}
    for (video, frame, event_type), base_row in list(merged.items()):
        summary["total"] += 1
        # ensure caches
        vf = crops_root / video
        if video not in metadata_cache:
            entries, full_video_path = load_metadata_for_video(vf, verbose=args.verbose)
            metadata_cache[video] = entries
            frame_map_cache[video] = build_frame_map_from_folder(vf, verbose=args.verbose)
        entries = metadata_cache[video]
        frame_map = frame_map_cache[video]
        matched_path, matched_frame, dist, note, candidates = match_event_to_crop(vf, entries, frame_map, event_type, frame, fallback_window=args.fallback_window, verbose=args.verbose)
        if matched_path:
            try:
                crop_rel = str(matched_path.relative_to(PROJECT_ROOT))
            except Exception:
                crop_rel = str(matched_path)
            merged[(video, frame, event_type)]["crop_path"] = crop_rel
            merged[(video, frame, event_type)]["matched"] = True
            merged[(video, frame, event_type)]["notes"] = note
            summary["matched"] += 1
            if len(candidates) > 1:
                summary["ambiguous"] += 1
        else:
            merged[(video, frame, event_type)]["matched"] = False
            merged[(video, frame, event_type)]["notes"] = "no_crop_found"
            summary["missing"] += 1

    # Build per-video rows and write per-video CSVs for every video folder found (and for videos present in input)
    per_video_map = defaultdict(list)
    for (video, frame, event_type), r in merged.items():
        per_video_map[video].append({
            "frame": frame,
            "event_type": event_type,
            "crop_path": r["crop_path"],
            "matched": r["matched"],
            "notes": r["notes"],
            "source": r["source"]
        })

    # Ensure we include videos that exist in crops_root but had no events in merged (create empty CSVs)
    for vf in video_folders:
        if vf.name not in per_video_map:
            per_video_map[vf.name] = []

    # Write per-video CSVs
    for video, rows_list in per_video_map.items():
        safe_name = video.replace("\\","_").replace("/","_")
        per_video_path = per_video_dir / f"{safe_name}_events.csv"
        write_per_video_csv(per_video_path, rows_list)
        summary["per_video"][video] = {"events": len(rows_list), "path": str(per_video_path)}
        if args.verbose:
            print(f"Wrote per-video CSV: {per_video_path}")

    # Write master CSV
    master_path = out_dir / f"{args.exp}_events_long.csv"
    master_fieldnames = ["video","frame","event_type","crop_path","matched","notes","source"]
    master_rows = []
    for (video, frame, event_type), r in merged.items():
        master_rows.append({
            "video": video,
            "frame": frame,
            "event_type": event_type,
            "crop_path": r["crop_path"],
            "matched": r["matched"],
            "notes": r["notes"],
            "source": r["source"]
        })
    write_master_csv(master_path, master_rows, master_fieldnames)

    # Write summary with explicit note about full .avi files location
    summary_path = out_dir / f"{args.exp}_summary.txt"
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write(f"Event assembly summary - {datetime.datetime.utcnow().isoformat()}Z\n")
        fh.write(f"Input CSV: {input_csv}\n")
        fh.write(f"Crops root: {crops_root}\n")
        fh.write(f"Experiment: {args.exp}\n\n")
        fh.write(f"Videos scanned (folders under crops root): {summary['videos_scanned']}\n")
        fh.write(f"Total events considered (merged input + metadata): {summary['total']}\n")
        fh.write(f"Matched events (crop file found): {summary['matched']}\n")
        fh.write(f"Missing events (no crop found): {summary['missing']}\n")
        fh.write(f"Ambiguous / duplicates flagged: {summary['ambiguous']}\n\n")
        fh.write("Per-video summary:\n")
        for v, info in summary["per_video"].items():
            fh.write(f"  {v}: {info['events']} events -> {info['path']}\n")
        fh.write("\nNotes:\n")
        fh.write(" - 'source' indicates whether the event row originated from the input CSV (input) or from crops/<video>/metadata.json (metadata).\n")
        fh.write(" - matched: True/False indicates whether a crop file was found for the event.\n")
        fh.write("\nImportant provenance note:\n")
        fh.write(" - The metadata entries include a 'video' field that points to the full .avi file location(s). Those full .avi files may include many files that have not been analyzed and are NOT part of the current dataset unless explicitly listed in the input CSV or present as crops under the crops root. Treat the full .avi location(s) as the original source media; only crops present under the crops root are considered part of the dataset for training/evaluation.\n")
        fh.write("\nGenerated at: " + datetime.datetime.utcnow().isoformat() + "Z\n")

    if args.verbose:
        print("Wrote master CSV:", master_path)
        print("Wrote summary:", summary_path)

    # Optional manifest update
    if args.update_manifest:
        manifest_path = PROJECT_ROOT / "data" / "manifests" / "manifest.csv"
        append_to_manifest(manifest_path, master_rows, verbose=args.verbose)

    print("Assembly complete.")
    print(f"Master CSV: {master_path}")
    print(f"Per-video CSVs: {per_video_dir}")
    print(f"Summary: {summary_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

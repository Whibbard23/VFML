#!/usr/bin/env python3
"""
event_training/training/auto_max_recall.py

Iterative orchestration: train -> inference -> hysteresis eval -> (mine+merge) -> retrain
Goal: maximize precision while keeping recall >= 0.98. Nudge factor toward target = 0.3.

Includes safeguards:
- cooldown after accepted weight change
- minimum improvement threshold for acceptance
- flip-flop detection
- best-seen revert
- stability mode for decreases
- warmup for first iterations
- tuning log

Run from repo root so relative script calls resolve correctly.
"""
import argparse
import subprocess
import shutil
import time
from pathlib import Path
import csv
import json
import numpy as np

# -------------------------
# Helpers
# -------------------------
def run_cmd(cmd, cwd=None, env=None):
    print("RUN:", " ".join(map(str, cmd)))
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=None,      # let subprocess inherit stdout
        stderr=None,      # let subprocess inherit stderr
        text=True
    )
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {proc.returncode}): {' '.join(map(str, cmd))}"
        )



def next_train_folder(runs_root: Path):
    runs_root.mkdir(parents=True, exist_ok=True)
    existing = [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("train_mouth_")]
    indices = []
    for p in existing:
        try:
            idx = int(p.name.split("_")[-1])
            indices.append(idx)
        except Exception:
            pass
    next_idx = max(indices) + 1 if indices else 1
    return runs_root / f"train_mouth_{next_idx}"

def load_mapping_and_labels(mapping_csv, labels_npy):
    rows = []
    with open(mapping_csv, "r", encoding="utf-8-sig", newline="") as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            rows.append({"video": (r.get("video") or "").strip(), "frame": int(float(r.get("frame") or 0)), "label": int(float(r.get("label") or 0))})
    labels = np.load(labels_npy)
    return rows, labels

def compute_metrics_from_threshold(probs_npy, mapping_csv, labels_npy, threshold=0.5):
    probs = np.load(probs_npy)
    rows, labels = load_mapping_and_labels(mapping_csv, labels_npy)
    if not (probs.shape[0] == labels.shape[0] == len(rows)):
        raise RuntimeError("Length mismatch between probs, labels, and mapping.")
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    fn_per_video = {}
    for r, lab, pred in zip(rows, labels, preds):
        v = r["video"]
        if lab == 1 and pred == 0:
            fn_per_video[v] = fn_per_video.get(v, 0) + 1
        else:
            fn_per_video.setdefault(v, 0)
    avg_fn_per_video = sum(fn_per_video.values()) / max(1, len(fn_per_video))
    return {
        "precision": precision,
        "recall": recall,
        "tp": tp, "fp": fp, "fn": fn,
        "avg_fn_per_video": avg_fn_per_video,
        "fn_per_video": fn_per_video
    }

def compute_metrics_via_hysteresis(inf_out_dir: Path):
    """
    Call apply_hysteresis_and_eval.py with the inference outputs and return metrics.
    Writes evaluation outputs into <inf_out_dir>/eval_hysteresis/.
    Falls back to threshold-based metrics if hysteresis script fails or summary missing.
    """
    eval_out = inf_out_dir / "eval_hysteresis"
    eval_out.mkdir(parents=True, exist_ok=True)

    ah_cmd = [
        "python", "apply_hysteresis_and_eval.py",
        "--probs", str(inf_out_dir / "probs.npy"),
        "--mapping", str(inf_out_dir / "index_to_row.csv"),
        "--labels", str(inf_out_dir / "val_labels.npy"),
        "--out-dir", str(eval_out),
        "--window", "1"
    ]
    try:
        run_cmd(ah_cmd)
    except Exception as e:
        print("apply_hysteresis_and_eval.py failed or not found; falling back to simple threshold metrics.", e)
        return compute_metrics_from_threshold(str(inf_out_dir / "probs.npy"), str(inf_out_dir / "index_to_row.csv"), str(inf_out_dir / "val_labels.npy"))

    summary_json = eval_out / "summary.json"
    summary_txt = eval_out / "summary.txt"
    if summary_json.exists():
        with summary_json.open("r", encoding="utf-8") as fh:
            s = json.load(fh)
            return {
                "precision": float(s.get("precision", 0.0)),
                "recall": float(s.get("recall", 0.0)),
                "tp": int(s.get("tp", 0)),
                "fp": int(s.get("fp", 0)),
                "fn": int(s.get("fn", 0)),
                "avg_fn_per_video": float(s.get("avg_fn_per_video", 0.0)),
                "fn_per_video": s.get("fn_per_video", {})
            }
    if summary_txt.exists():
        metrics = {}
        with summary_txt.open("r", encoding="utf-8") as fh:
            for line in fh:
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip().lower()
                    v = v.strip()
                    try:
                        metrics[k] = float(v) if "." in v or "e" in v.lower() else int(v)
                    except Exception:
                        metrics[k] = v
        return {
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "tp": int(metrics.get("tp", 0)),
            "fp": int(metrics.get("fp", 0)),
            "fn": int(metrics.get("fn", 0)),
            "avg_fn_per_video": float(metrics.get("avg_fn_per_video", 0.0)),
            "fn_per_video": {}
        }

    return compute_metrics_from_threshold(str(inf_out_dir / "probs.npy"), str(inf_out_dir / "index_to_row.csv"), str(inf_out_dir / "val_labels.npy"))

def append_unique_mined_negatives(original_csv: Path, mined_csv: Path, backup_suffix=".bak"):
    """
    Safe append: backup original CSV (first time), deduplicate by (video,frame), append minimal rows.
    """
    if not mined_csv.exists():
        print(f"No mined CSV found at {mined_csv}; skipping append.")
        return 0

    backup = original_csv.with_suffix(original_csv.suffix + backup_suffix)
    if not backup.exists():
        shutil.copy(original_csv, backup)
        print(f"Backed up original CSV to {backup}")

    existing = set()
    with original_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            existing.add(((r.get("video") or "").strip(), int(float(r.get("frame") or 0))))

    to_append = []
    with mined_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            v = (r.get("video") or "").strip()
            if v == "":
                continue
            f = int(float(r.get("frame") or 0))
            key = (v, f)
            if key in existing:
                continue
            to_append.append({"video": v, "frame": f, "label": 0, "split": "train"})
            existing.add(key)

    if not to_append:
        print("No new mined negatives to append.")
        return 0

    with original_csv.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        for r in to_append:
            writer.writerow([r["video"], r["frame"], r["label"], r["split"]])
    print(f"Appended {len(to_append)} mined negatives to {original_csv}")
    return len(to_append)

def write_tuning_log(runs_root: Path, row: dict):
    runs_root.mkdir(parents=True, exist_ok=True)
    log_path = runs_root / "tuning_log.csv"
    header = ["iter","pos_weight","precision","recall","avg_fn_per_video","action","note","epochs"]
    exists = log_path.exists()
    with log_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if not exists:
            writer.writerow(header)
        writer.writerow([row.get(h, "") for h in header])

# -------------------------
# Main orchestration
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Canonical CSV (will be backed up).")
    p.add_argument("--frames-root", default="runs/inference")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--initial-pos-weight", type=float, default=1.96)
    p.add_argument("--target-pos-weight", type=float, default=3.0)
    p.add_argument("--precision-target", type=float, default=0.99)
    p.add_argument("--recall-target", type=float, default=0.98)
    p.add_argument("--fn-per-video-target", type=float, default=30.0)
    p.add_argument("--max-iters", type=int, default=8)
    p.add_argument("--train-cmd", default="python -m event_training.training.train_mouth_model")
    p.add_argument("--inference-cmd", default="python -m inference.run_mouth_inference")
    p.add_argument("--mine-cmd", default="python hard_negative_mine.py")
    p.add_argument("--merge-cmd", default="python merge_with_hardneg.py")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    working_csv = csv_path  # will be backed up on first append
    runs_root = Path(args.runs_root)
    pos_weight = float(args.initial_pos_weight)
    best_pos_weight = pos_weight
    best_metrics = None
    cooldown = 0
    cooldown_iters = 2
    flip_history = []
    flip_window = 4
    min_delta_precision = 0.002
    nudge_factor = 0.3
    min_pos_weight = 0.5
    max_pos_weight = float(args.target_pos_weight)
    no_progress_count = 0
    stability_multiplier_for_decrease = 1.5  # increase epochs when testing decreases
    warmup_iters = 2
    warmup_multiplier = 1.5
    iter_no = 0
    last_metrics = None

    while iter_no < args.max_iters:
        iter_no += 1
        out_dir = next_train_folder(runs_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== ITER {iter_no} pos_weight={pos_weight:.3f} out_dir={out_dir} ===\n")

        # decide epochs for this iteration (warmup for first iterations)
        epochs = args.epochs
        if iter_no <= warmup_iters:
            epochs = int(max(1, args.epochs * warmup_multiplier))

        # Train
        train_cmd = [
            *args.train_cmd.split(),
            "--csv", str(working_csv),
            "--frames-root", str(args.frames_root),
            "--out-dir", str(out_dir),
            "--batch-size", str(args.batch_size),
            "--epochs", str(epochs),
            "--num-workers", str(args.num_workers),
            "--seed", "42",
            "--pos-weight-multiplier", f"{pos_weight:.6f}",
            "--oversample-positives",
            "--log-interval", "50"
        ]
        run_cmd(train_cmd)

        # find checkpoint
        ckpt = out_dir / "best.pth"
        if not ckpt.exists():
            pths = list(out_dir.glob("*.pth"))
            if pths:
                ckpt = pths[0]
            else:
                raise FileNotFoundError(f"No checkpoint found in {out_dir}")

        # Inference on val
        inf_out = out_dir / "inference"
        inf_cmd = [
            *args.inference_cmd.split(),
            "--ckpt", str(ckpt),
            "--csv", str(working_csv),
            "--frames-root", str(args.frames_root),
            "--out-dir", str(inf_out),
            "--batch-size", "16",
            "--num-workers", str(args.num_workers),
            "--device", args.device,
            "--split", "val"
        ]
        run_cmd(inf_cmd)

        # Hysteresis-based evaluation (saved inside iteration folder)
        metrics = compute_metrics_via_hysteresis(inf_out)
        last_metrics = metrics
        print(f"ITER {iter_no} metrics: precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} avg_FN_per_video={metrics['avg_fn_per_video']:.2f}")

        # Log iteration (epochs included)
        write_tuning_log(runs_root, {
            "iter": iter_no,
            "pos_weight": f"{pos_weight:.6f}",
            "precision": f"{metrics['precision']:.4f}",
            "recall": f"{metrics['recall']:.4f}",
            "avg_fn_per_video": f"{metrics['avg_fn_per_video']:.2f}",
            "action": "",
            "note": "",
            "epochs": epochs
        })

        # Stopping criteria
        if metrics["avg_fn_per_video"] <= args.fn_per_video_target and metrics["recall"] >= args.recall_target and metrics["precision"] >= args.precision_target:
            print("Targets met; stopping.")
            break

        action = "none"
        note = ""

        # If precision below target -> mine hard negatives and merge/append
        if metrics["precision"] < args.precision_target:
            action = "mine"
            print("Precision below target; running miner to collect targeted negatives.")
            mined_csv = out_dir / "mined_negatives.csv"
            mine_cmd = [
                *args.mine_cmd.split(),
                "--csv", str(working_csv),
                "--ckpt", str(ckpt),
                "--frames-root", str(args.frames_root),
                "--device", args.device,
                "--batch-size", "32",
                "--num-workers", str(args.num_workers),
                "--out-csv", str(mined_csv),
                "--top-k-per-video", "5",
                "--global-top-n", "0",
                "--resume"
            ]
            run_cmd(mine_cmd)

            merge_script = Path("merge_with_hardneg.py")
            if merge_script.exists():
                merge_out = out_dir / "merged_with_mined.csv"
                merge_cmd = [
                    *args.merge_cmd.split(),
                    "--orig", str(working_csv),
                    "--mined", str(mined_csv),
                    "--out", str(merge_out)
                ]
                try:
                    run_cmd(merge_cmd)
                    shutil.copy(merge_out, working_csv)
                    note = f"merged {mined_csv.name}"
                    print(f"Replaced working CSV with merged CSV: {merge_out}")
                except Exception as e:
                    print("merge_with_hardneg.py failed or not compatible; falling back to append.", e)
                    appended = append_unique_mined_negatives(working_csv, mined_csv)
                    note = f"appended {appended}"
            else:
                appended = append_unique_mined_negatives(working_csv, mined_csv)
                note = f"appended {appended}"
                if appended == 0:
                    print("No mined negatives appended; consider increasing mining depth or global-top-n.")
            # after mining, reset counters and continue
            no_progress_count = 0
            flip_history.clear()
            cooldown = max(0, cooldown - 1)
        else:
            # precision acceptable; focus on recall target and precision maximization
            if metrics["recall"] < args.recall_target:
                # propose upward nudge
                proposed_pw = pos_weight + (max_pos_weight - pos_weight) * nudge_factor
                if abs(proposed_pw - pos_weight) < 0.01:
                    proposed_pw = min(pos_weight * 1.1, max_pos_weight)
                direction = "up"
            else:
                # propose downward step to improve precision (conservative)
                proposed_pw = max(min_pos_weight, pos_weight * 0.90)
                direction = "down" if proposed_pw < pos_weight else "none"

            # cooldown check
            if cooldown > 0:
                action = "cooldown_skip"
                note = f"cooldown {cooldown}"
                cooldown -= 1
                print(f"Cooldown active ({cooldown} iters left). Skipping pos_weight change.")
            else:
                # flip-flop detection
                if direction in ("up", "down"):
                    flip_history.append(direction)
                    flip_history = flip_history[-flip_window:]
                    if flip_history.count("up") >= 2 and flip_history.count("down") >= 2:
                        action = "flip_flop_detected"
                        note = "flip-flop detected; will trigger mining next iter"
                        print("Flip-flop detected; will trigger mining next iteration instead of changing pos_weight.")
                    else:
                        # apply proposed change tentatively
                        prev_pos = pos_weight
                        # if decreasing, enter stability mode (increase epochs next iteration)
                        if direction == "down" and proposed_pw < pos_weight:
                            epochs = int(max(1, args.epochs * stability_multiplier_for_decrease))
                            note = f"stability_mode_epochs={epochs}"
                            print(f"Testing decrease in pos_weight: {pos_weight:.3f} -> {proposed_pw:.3f} with stability epochs={epochs}")
                        pos_weight = min(max(proposed_pw, min_pos_weight), max_pos_weight)
                        action = "proposed_change"
                        note = f"{direction} {prev_pos:.3f}->{pos_weight:.3f}"
                        print(f"Proposed pos_weight change: {note}")

        # write action to tuning log (update last row)
        write_tuning_log(runs_root, {
            "iter": iter_no,
            "pos_weight": f"{pos_weight:.6f}",
            "precision": f"{metrics['precision']:.4f}",
            "recall": f"{metrics['recall']:.4f}",
            "avg_fn_per_video": f"{metrics['avg_fn_per_video']:.2f}",
            "action": action,
            "note": note,
            "epochs": epochs
        })

        # Acceptance / revert logic
        if best_metrics is None:
            if metrics["recall"] >= args.recall_target:
                best_metrics = metrics.copy()
                best_pos_weight = pos_weight
        else:
            if metrics["recall"] >= args.recall_target and metrics["precision"] >= best_metrics["precision"] + min_delta_precision:
                best_metrics = metrics.copy()
                best_pos_weight = pos_weight
                cooldown = cooldown_iters
                no_progress_count = 0
                print(f"Accepted new best pos_weight={best_pos_weight:.3f} with precision={metrics['precision']:.4f}")
            else:
                if action == "proposed_change":
                    print(f"Proposed change did not improve metrics sufficiently; reverting to best_pos_weight={best_pos_weight:.3f}")
                    pos_weight = best_pos_weight
                    no_progress_count += 1
                    flip_history.append("revert")
                else:
                    no_progress_count += 1

        # If flip-flop or repeated no progress, schedule mining next iteration
        if (flip_history.count("up") >= 2 and flip_history.count("down") >= 2) or no_progress_count >= 3:
            print("Repeated flip-flop or no progress detected; scheduling mining next iteration.")
            pos_weight = best_pos_weight
            (runs_root / "force_mine.flag").write_text("1")

        # small pause to avoid file locks on Windows
        time.sleep(2)

    # Save final metrics and exit
    summary_path = Path(args.runs_root) / "auto_max_recall_last_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(last_metrics or {}, fh, indent=2)
    print("Orchestration complete. Last metrics:", last_metrics)

if __name__ == "__main__":
    main()

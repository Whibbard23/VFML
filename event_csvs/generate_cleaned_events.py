# generate_cleaned_events.py
"""
Reads event_table.csv (wide format: video, before_onset, touch_ues, leave_ues)
- Converts numeric frames from base-1 to base-0
- Replaces non-numeric entries (x, ?, text, blank) with empty (NaN)
- Drops rows where all three event columns are missing
- Writes cleaned_events.csv in the same folder
"""

import os
import pandas as pd

# Paths (script assumes it lives in event_csvs folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_CSV = os.path.join(SCRIPT_DIR, "event_table.csv")
OUT_CSV = os.path.join(SCRIPT_DIR, "cleaned_events.csv")

# Columns expected in the wide table
EVENT_COLS = ["before_onset", "touch_ues", "leave_ues"]
VIDEO_COL = "video"

def is_numeric_frame(x):
    """Return True if x represents a numeric frame (allow floats that are integers)."""
    if pd.isna(x):
        return False
    s = str(x).strip()
    if s == "":
        return False
    s_low = s.lower()
    if s_low in {"x", "?"}:
        return False
    try:
        # allow numeric strings like "123" or "123.0"
        f = float(s)
        # require it to be an integer value (no fractional frames)
        return float(int(f)) == f
    except Exception:
        return False

def to_base0_int(x):
    """Convert a numeric frame string/number (base-1) to base-0 int. Caller must ensure numeric."""
    f = int(float(str(x).strip()))
    return f - 1

def main():
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"Input CSV not found: {RAW_CSV}")

    df = pd.read_csv(RAW_CSV, dtype=str)
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    if VIDEO_COL not in df.columns:
        raise ValueError(f"Input CSV must contain a '{VIDEO_COL}' column.")

    # Ensure event columns exist; if missing, add them as empty
    for c in EVENT_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    # Process each event column: numeric -> base-0 int, non-numeric -> NaN
    counts = {"kept_numeric": 0, "cleared_non_numeric": 0}
    for c in EVENT_COLS:
        out_vals = []
        for v in df[c].values:
            if is_numeric_frame(v):
                out_vals.append(to_base0_int(v))
                counts["kept_numeric"] += 1
            else:
                out_vals.append(pd.NA)
                counts["cleared_non_numeric"] += 1
        df[c] = out_vals

    # Drop rows where all three event columns are missing
    before_drop = len(df)
    df = df.dropna(subset=EVENT_COLS, how="all").reset_index(drop=True)
    after_drop = len(df)
    dropped_rows = before_drop - after_drop

    # Save cleaned CSV
    df.to_csv(OUT_CSV, index=False)

    # Summary
    print("generate_cleaned_events.py finished.")
    print(f"Input file: {RAW_CSV}")
    print(f"Output file: {OUT_CSV}")
    print(f"Rows before cleaning: {before_drop}")
    print(f"Rows after dropping rows with no events: {after_drop}")
    print(f"Rows dropped: {dropped_rows}")
    print(f"Numeric frames kept: {counts['kept_numeric']}")
    print(f"Non-numeric entries cleared: {counts['cleared_non_numeric']}")

if __name__ == "__main__":
    main()

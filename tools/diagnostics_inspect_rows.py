# diagnostics_inspect_rows.py
import pandas as pd
from pathlib import Path

CSV = Path(r"C:\Users\Connor Lab\Desktop\VFML\event_csvs\cleaned_events.csv")
rows_to_check = [41, 80, 331, 438, 654, 667]  # 0-based or 1-based? we'll print both

df = pd.read_csv(CSV, dtype=str)
df.columns = [c.strip().lower() for c in df.columns]
print("Columns:", df.columns.tolist())
for r in rows_to_check:
    if r < 0 or r >= len(df):
        print(f"Row {r} out of range (len={len(df)})")
        continue
    row = df.iloc[r]
    raw_before = row.get("before_onset")
    raw_touch = row.get("touch_ues")
    raw_leave = row.get("leave_ues")
    # try coercion
    b_num = pd.to_numeric(raw_before, errors="coerce")
    t_num = pd.to_numeric(raw_touch, errors="coerce")
    l_num = pd.to_numeric(raw_leave, errors="coerce")
    print(f"\nRow index {r} (CSV row):")
    print("  video:", row.get("video"))
    print("  raw before_onset:", repr(raw_before), "-> numeric:", b_num)
    print("  raw touch_ues   :", repr(raw_touch), "-> numeric:", t_num)
    print("  raw leave_ues   :", repr(raw_leave), "-> numeric:", l_num)

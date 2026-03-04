import pandas as pd
df = pd.read_csv("event_csvs/mouth_crops_labels_train.csv")
print(df['label'].value_counts())

onsets = df[df.label == 1]
missing = []
for _, row in onsets.iterrows():
    before = df[(df.filepath == row.filepath) & (df.frame == row.frame - 1)]
    if before.empty:
        missing.append((row.filepath, row.frame))
print("Missing before_onset:", len(missing))

befores = df[df.label == 0]  # may include random negatives too
missing = []
for _, row in befores.iterrows():
    onset = df[(df.filepath == row.filepath) & (df.frame == row.frame + 1)]
    if onset.empty:
        missing.append((row.filepath, row.frame))
print("Missing onset:", len(missing))

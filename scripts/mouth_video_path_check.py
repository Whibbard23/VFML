import pandas as pd
df = pd.read_csv("event_csvs/assembly_1_train_events.csv")
print(df.head(10))
# show unique video paths
print(df['video'].unique()[:10])

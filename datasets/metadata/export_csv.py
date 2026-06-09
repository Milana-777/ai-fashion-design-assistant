import json
import pandas as pd

with open("datasets/metadata/labelled_metadata.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df.to_csv("datasets/metadata/fashion_labels.csv", index=False)

print(f"Exported {len(df)} rows to datasets/metadata/fashion_labels.csv")
print("\nSample rows:")
print(df.head(5).to_string())
print("\nSchema coverage:")
cols = ["category","color_tone","fabric_weight","silhouette",
        "neckline","sleeve","pattern","occasion","gender","season"]
for col in cols:
    known = (df[col] != "unknown").sum()
    pct   = int(known / len(df) * 100)
    bar   = "#" * (pct // 5) + "-" * (20 - pct // 5)
    print(f"  {col:15s} [{bar}] {pct}% labelled")

print("\nTask 3 complete!")

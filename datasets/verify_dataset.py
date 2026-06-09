import os, json
import pandas as pd
from PIL import Image

rows = []
for split in ["train", "val", "test"]:
    meta_path = f"datasets/processed/fashionpedia/{split}/metadata.json"
    img_dir   = f"datasets/processed/fashionpedia/{split}/images"

    with open(meta_path) as f:
        items = json.load(f)

    for item in items:
        rows.append({
            "filename": item["filename"],
            "split":    split,
            "width":    item["width"],
            "height":   item["height"],
            "image_id": item.get("image_id", ""),
        })

df = pd.DataFrame(rows)
df.to_csv("datasets/processed/fashionpedia_manifest.csv", index=False)

print("=" * 40)
print("   FINAL DATASET MANIFEST")
print("=" * 40)
print(df.groupby("split").size().to_string())
print(f"\nTotal images : {len(df)}")
print(f"Avg width    : {int(df['width'].mean())} px")
print(f"Avg height   : {int(df['height'].mean())} px")
print("=" * 40)
print("\nManifest saved: datasets/processed/fashionpedia_manifest.csv")
print("\nTask 2 complete! Ready for Task 3.")

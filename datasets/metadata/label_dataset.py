import os, json
import pandas as pd
from PIL import Image
import numpy as np

SCHEMA_PATH = "datasets/metadata/fashion_schema.json"
CLEAN_META  = "datasets/processed/fashionpedia/all/metadata_clean.json"
CLEAN_IMGS  = "datasets/processed/fashionpedia/all/images"
OUT_PATH    = "datasets/metadata/labelled_metadata.json"

with open(SCHEMA_PATH) as f:
    schema = json.load(f)

with open(CLEAN_META) as f:
    meta = json.load(f)

def get_dominant_color(img_path):
    with Image.open(img_path) as img:
        img = img.convert("RGB").resize((50, 50))
        pixels = np.array(img).reshape(-1, 3)
        avg = pixels.mean(axis=0)
        r, g, b = avg

        if r > 200 and g > 200 and b > 200: return "white"
        if r < 60  and g < 60  and b < 60:  return "black"
        if r > g and r > b:
            return "red" if r > 150 else "brown"
        if g > r and g > b:   return "green"
        if b > r and b > g:   return "blue"
        if r > 180 and g > 150 and b < 100: return "yellow"
        if r > 180 and g < 120 and b > 150: return "purple"
        if r > 200 and g > 100 and b < 100: return "orange"
        if r > 180 and g < 140 and b < 140: return "pink"
        if abs(int(r)-int(g)) < 20 and abs(int(g)-int(b)) < 20: return "gray"
        return "beige"

def guess_category(width, height):
    ratio = height / max(width, 1)
    if ratio > 1.8:  return "dress"
    if ratio > 1.3:  return "top"
    return "pants"

def guess_fabric_weight(img_path):
    with Image.open(img_path) as img:
        img_gray = img.convert("L").resize((50, 50))
        arr = np.array(img_gray)
        std = arr.std()
        if std < 20:  return "sheer"
        if std < 40:  return "light"
        if std < 60:  return "medium"
        return "heavy"

print(f"Labelling {len(meta)} images...\n")
labelled = []

for i, item in enumerate(meta):
    img_path = os.path.join(CLEAN_IMGS, item["filename"])

    try:
        color    = get_dominant_color(img_path)
        category = guess_category(item["width"], item["height"])
        weight   = guess_fabric_weight(img_path)
    except Exception as e:
        print(f"  Skipping {item['filename']}: {e}")
        color, category, weight = "unknown", "unknown", "unknown"

    labelled.append({
        "id":           item["id"],
        "filename":     item["filename"],
        "width":        item["width"],
        "height":       item["height"],
        "image_id":     item.get("image_id", ""),
        "category":     category,
        "color_tone":   color,
        "fabric_weight": weight,
        "silhouette":   "unknown",
        "neckline":     "unknown",
        "sleeve":       "unknown",
        "pattern":      "unknown",
        "occasion":     "unknown",
        "gender":       "unknown",
        "season":       "unknown",
    })

    if (i + 1) % 100 == 0:
        print(f"  Labelled {i+1}/{len(meta)}...")

with open(OUT_PATH, "w") as f:
    json.dump(labelled, f, indent=2)

df = pd.DataFrame(labelled)

print(f"\n{'='*40}")
print(f"   LABEL SUMMARY")
print(f"{'='*40}")
print(f"\nCategory distribution:")
print(df["category"].value_counts().to_string())
print(f"\nColor tone distribution:")
print(df["color_tone"].value_counts().to_string())
print(f"\nFabric weight distribution:")
print(df["fabric_weight"].value_counts().to_string())
print(f"\n{'='*40}")
print(f"Labelled metadata saved: {OUT_PATH}")
print(f"\nTask 3 Step 1 complete!")

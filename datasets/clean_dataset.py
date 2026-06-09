import os, json, hashlib
from PIL import Image, UnidentifiedImageError
import pandas as pd

IMG_DIR   = "datasets/raw/fashionpedia_sample/images"
META_PATH = "datasets/raw/fashionpedia_sample/metadata.json"
OUT_DIR   = "datasets/processed/fashionpedia/all/images"
MIN_SIZE  = (128, 128)

os.makedirs(OUT_DIR, exist_ok=True)

with open(META_PATH) as f:
    meta = json.load(f)

print(f"Starting with {len(meta)} images...\n")

valid     = []
corrupt   = []
too_small = []
duplicate = []
seen_hashes = set()

for i, item in enumerate(meta):
    path = os.path.join(IMG_DIR, item["filename"])

    # Check 1 — file exists
    if not os.path.exists(path):
        corrupt.append(item["filename"])
        continue

    # Check 2 — valid image
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception:
        corrupt.append(item["filename"])
        continue

    # Check 3 — minimum size
    try:
        with Image.open(path) as img:
            w, h = img.size
            if w < MIN_SIZE[0] or h < MIN_SIZE[1]:
                too_small.append(item["filename"])
                continue
    except Exception:
        corrupt.append(item["filename"])
        continue

    # Check 4 — duplicate detection
    with open(path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    if file_hash in seen_hashes:
        duplicate.append(item["filename"])
        continue

    seen_hashes.add(file_hash)
    valid.append({**item, "hash": file_hash, "width": w, "height": h})

    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/{len(meta)}...")

print(f"\n{'='*40}")
print(f"  CLEANING RESULTS")
print(f"{'='*40}")
print(f"  Original  : {len(meta)}")
print(f"  Valid     : {len(valid)}")
print(f"  Corrupt   : {len(corrupt)}")
print(f"  Too small : {len(too_small)}")
print(f"  Duplicate : {len(duplicate)}")
print(f"  Removed   : {len(meta) - len(valid)}")
print(f"{'='*40}")

import shutil
for item in valid:
    src = os.path.join(IMG_DIR, item["filename"])
    dst = os.path.join(OUT_DIR, item["filename"])
    shutil.copy2(src, dst)

clean_meta_path = "datasets/processed/fashionpedia/all/metadata_clean.json"
with open(clean_meta_path, "w") as f:
    json.dump(valid, f, indent=2)

print(f"\nClean images saved to : {OUT_DIR}")
print(f"Clean metadata saved  : {clean_meta_path}")
print("\nStep 3 complete!")

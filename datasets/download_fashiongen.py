from datasets import load_dataset
from PIL import Image
import os, json, time

SAVE_DIR    = "datasets/raw/fashionpedia_sample"
IMG_DIR     = os.path.join(SAVE_DIR, "images")
NUM_SAMPLES = 1000

os.makedirs(IMG_DIR, exist_ok=True)

print("Connecting to HuggingFace...")
ds = load_dataset(
    "detection-datasets/fashionpedia",
    split="train",
    streaming=True
)

print(f"Downloading {NUM_SAMPLES} samples...\n")
metadata = []
start = time.time()

for i, item in enumerate(ds):
    img: Image.Image = item["image"]
    img_path = os.path.join(IMG_DIR, f"{i:05d}.jpg")
    img.save(img_path, "JPEG")

    metadata.append({
        "id":       i,
        "filename": f"{i:05d}.jpg",
        "width":    item.get("width", 0),
        "height":   item.get("height", 0),
        "image_id": item.get("image_id", ""),
    })

    if (i + 1) % 100 == 0:
        elapsed = time.time() - start
        print(f"  {i+1}/{NUM_SAMPLES} images saved  ({elapsed:.0f}s elapsed)")

    if i + 1 >= NUM_SAMPLES:
        break

meta_path = os.path.join(SAVE_DIR, "metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nDone!")
print(f"  Images saved : {len(metadata)}")
print(f"  Metadata     : {meta_path}")
print(f"  Total time   : {time.time() - start:.0f}s")
import os, json, shutil, random

CLEAN_META = "datasets/processed/fashionpedia/all/metadata_clean.json"
CLEAN_IMGS = "datasets/processed/fashionpedia/all/images"

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

with open(CLEAN_META) as f:
    meta = json.load(f)

random.seed(42)
random.shuffle(meta)

n       = len(meta)
n_train = int(n * TRAIN_RATIO)
n_val   = int(n * VAL_RATIO)

splits = {
    "train": meta[:n_train],
    "val":   meta[n_train:n_train + n_val],
    "test":  meta[n_train + n_val:]
}

for split_name, items in splits.items():
    img_out  = f"datasets/processed/fashionpedia/{split_name}/images"
    meta_out = f"datasets/processed/fashionpedia/{split_name}/metadata.json"
    os.makedirs(img_out, exist_ok=True)

    for item in items:
        src = os.path.join(CLEAN_IMGS, item["filename"])
        dst = os.path.join(img_out,    item["filename"])
        shutil.copy2(src, dst)

    with open(meta_out, "w") as f:
        json.dump(items, f, indent=2)

    print(f"  {split_name:6s} : {len(items)} images")

print("\nStep 4 complete!")

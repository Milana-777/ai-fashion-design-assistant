import os, json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

meta_path = "datasets/raw/fashionpedia_sample/metadata.json"
img_dir   = "datasets/raw/fashionpedia_sample/images"

with open(meta_path) as f:
    meta = json.load(f)

df = pd.DataFrame(meta)

print("=" * 40)
print("   DATASET SUMMARY")
print("=" * 40)
print(f"  Total samples : {len(df)}")
print(f"  Columns       : {df.columns.tolist()}")
print(f"  Avg width     : {int(df['width'].mean())} px")
print(f"  Avg height    : {int(df['height'].mean())} px")
print(f"  Min size      : {int(df['width'].min())} x {int(df['height'].min())} px")
print(f"  Max size      : {int(df['width'].max())} x {int(df['height'].max())} px")
print("=" * 40)

os.makedirs("week1_research/notes", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["width"].tolist(),  bins=20, color="#7F77DD", edgecolor="white")
axes[0].set_title("Width distribution")
axes[0].set_xlabel("pixels")
axes[1].hist(df["height"].tolist(), bins=20, color="#1D9E75", edgecolor="white")
axes[1].set_title("Height distribution")
axes[1].set_xlabel("pixels")
plt.tight_layout()
plt.savefig("week1_research/notes/size_distribution.png", dpi=150)
print("\nSize distribution chart saved!")

fig2, axes2 = plt.subplots(3, 5, figsize=(16, 10))
for i, ax in enumerate(axes2.flat):
    img_path = os.path.join(img_dir, df.iloc[i]["filename"])
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.set_title(f"ID: {df.iloc[i]['image_id']}", fontsize=8)
    ax.axis("off")
plt.suptitle("Fashionpedia sample images", fontsize=14)
plt.tight_layout()
plt.savefig("week1_research/notes/sample_grid.png", dpi=150)
print("Sample image grid saved!")

summary = {
    "dataset": "fashionpedia_sample",
    "total_images": len(df),
    "avg_width":  int(df["width"].mean()),
    "avg_height": int(df["height"].mean()),
}
with open("week1_research/notes/dataset_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nAll outputs saved to week1_research/notes/")
print("Step 2 complete!")

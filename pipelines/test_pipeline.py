import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.insert(0, os.getcwd())

from pipelines.fashion_dataset import FashionDataset, get_dataloaders

print("=" * 45)
print("   FASHION DATA PIPELINE — TEST RUN")
print("=" * 45)

print("\n1. Loading dataloaders...")
loaders = get_dataloaders(batch_size=8, img_size=256, num_workers=0)

print("\n2. Testing a single batch from train loader...")
train_loader = loaders["train"]
batch = next(iter(train_loader))

images = batch["image"]
labels = batch["labels"]

print(f"\n   Batch image shape : {images.shape}")
print(f"   dtype             : {images.dtype}")
print(f"   min / max values  : {images.min():.2f} / {images.max():.2f}")
print(f"   Label keys        : {list(labels.keys())}")

print("\n3. Saving sample batch grid...")
os.makedirs("week1_research/notes", exist_ok=True)

def denormalize(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    img    = denormalize(images[i])
    img_np = img.permute(1, 2, 0).numpy()
    ax.imshow(img_np)
    cat_idx = labels["category"][i].item()
    col_idx = labels["color_tone"][i].item()
    ax.set_title(f"cat:{cat_idx} col:{col_idx}", fontsize=9)
    ax.axis("off")

plt.suptitle("Pipeline batch — sample images", fontsize=13)
plt.tight_layout()
plt.savefig("week1_research/notes/pipeline_batch.png", dpi=150)
print("   Saved: week1_research/notes/pipeline_batch.png")

print("\n4. Checking all splits...")
for split, loader in loaders.items():
    batch = next(iter(loader))
    print(f"   {split:6s} — batch shape: {batch['image'].shape} "
          f"| labels: {len(batch['labels'])} attrs")

print("\n5. GPU check...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Device: {device}")
if device == "cuda":
    print(f"   GPU   : {torch.cuda.get_device_name(0)}")

print("\n" + "=" * 45)
print("   PIPELINE TEST PASSED!")
print("=" * 45)
print("\nWeek 1 Task 4 complete!")
print("Your data pipeline is ready for Stable Diffusion (Week 2).")

import os, json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

SCHEMA_VALUES = {
    "category":     ["dress","top","shirt","blouse","jacket","coat",
                     "pants","skirt","shorts","jumpsuit","sweater",
                     "cardigan","vest","activewear","swimwear","unknown"],
    "color_tone":   ["black","white","gray","beige","brown","red",
                     "pink","orange","yellow","green","blue",
                     "purple","multicolor","unknown"],
    "fabric_weight":["sheer","light","medium","heavy","unknown"],
    "occasion":     ["casual","formal","evening","streetwear",
                     "business","athletic","beach","unknown"],
    "gender":       ["women","men","unisex","unknown"],
    "season":       ["spring","summer","autumn","winter","all-season","unknown"],
}

class FashionDataset(Dataset):
    def __init__(self, meta_path, img_dir, split="train", img_size=256):
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.img_dir  = img_dir
        self.split    = split
        self.img_size = img_size

        if split == "train":
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2,
                              saturation=0.2, hue=0.05),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        item     = self.meta[idx]
        img_path = os.path.join(self.img_dir, item["filename"])
        image    = Image.open(img_path).convert("RGB")
        image    = self.transform(image)

        labels = {}
        for attr, values in SCHEMA_VALUES.items():
            val = item.get(attr, "unknown")
            if val not in values:
                val = "unknown"
            labels[attr] = values.index(val)

        return {
            "image":    image,
            "labels":   labels,
            "filename": item["filename"],
            "id":       item["id"],
        }

def get_dataloaders(batch_size=32, img_size=256, num_workers=2):
    splits = {}
    for split in ["train", "val", "test"]:
        meta_path = f"datasets/processed/fashionpedia/{split}/metadata.json"
        img_dir   = f"datasets/processed/fashionpedia/{split}/images"
        if not os.path.exists(meta_path):
            print(f"  Skipping {split} — metadata not found")
            continue
        ds = FashionDataset(meta_path, img_dir, split=split, img_size=img_size)
        splits[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        print(f"  {split:6s} loader: {len(ds)} images, "
              f"{len(splits[split])} batches")
    return splits

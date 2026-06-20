import os, json, shutil, random
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

print("=" * 54)
print("   BRAND DATASET PREPARATION — WEEK 4")
print("=" * 54)

BRANDS = {
    "minimalist": {
        "token":       "<minimalist-style>",
        "description": "Clean lines, neutral palette, minimal decoration",
        "colors":      [(245,245,240),(230,225,215),(200,195,185),
                        (170,165,155),(140,135,125),(100,98,92)],
        "captions": [
            "<minimalist-style> fashion, clean white dress, "
            "minimal design, neutral palette, structured silhouette",
            "<minimalist-style> outfit, beige linen blazer, "
            "simple cut, no embellishment, editorial style",
            "<minimalist-style> look, monochrome ensemble, "
            "clean lines, understated elegance, white background",
            "<minimalist-style> garment, off-white structured coat, "
            "minimal details, architectural silhouette",
            "<minimalist-style> fashion, greige tones, "
            "simple shapes, negative space, quiet luxury",
        ],
    },
    "streetwear": {
        "token":       "<streetwear-style>",
        "description": "Bold graphics, oversized fits, urban aesthetic",
        "colors":      [(15,15,15),(30,30,30),(220,50,50),
                        (255,200,0),(50,120,220),(240,240,240)],
        "captions": [
            "<streetwear-style> fashion, oversized black hoodie, "
            "bold graphic print, urban style, street photography",
            "<streetwear-style> outfit, cargo pants, "
            "layered look, sneakers, graffiti background",
            "<streetwear-style> look, puffer jacket, "
            "bright colorway, logo tape, city setting",
            "<streetwear-style> garment, oversized tee, "
            "dropped shoulders, distressed denim, urban editorial",
            "<streetwear-style> fashion, tracksuit set, "
            "bold branding, athletic silhouette, street style",
        ],
    },
    "luxury": {
        "token":       "<luxury-style>",
        "description": "Rich fabrics, ornate details, opulent aesthetic",
        "colors":      [(10,8,5),(40,30,20),(120,90,40),
                        (180,150,80),(210,180,120),(240,220,180)],
        "captions": [
            "<luxury-style> fashion, black velvet gown, "
            "gold embroidery, floor length, opulent detail",
            "<luxury-style> outfit, silk brocade jacket, "
            "intricate pattern, jewel tones, couture quality",
            "<luxury-style> look, fur-trimmed coat, "
            "rich texture, structured shoulders, editorial luxury",
            "<luxury-style> garment, embellished evening dress, "
            "crystal details, silk fabric, high fashion photography",
            "<luxury-style> fashion, gold brocade blazer, "
            "ornate buttons, rich fabric, luxury editorial",
        ],
    },
}

def create_brand_image(brand_name, config, idx):
    w, h = 512, 512
    img  = Image.new("RGB", (w, h), (250, 248, 245))
    draw = ImageDraw.Draw(img)
    colors = config["colors"]
    random.seed(idx * 42)

    if brand_name == "minimalist":
        bg = colors[0]
        img = Image.new("RGB", (w, h), bg)
        draw = ImageDraw.Draw(img)
        c1 = colors[random.randint(2, 4)]
        x1 = random.randint(120, 180)
        y1 = random.randint(60,  100)
        x2 = random.randint(330, 390)
        y2 = random.randint(420, 460)
        draw.rectangle([x1, y1, x2, y2], fill=c1)
        c2 = colors[1]
        draw.rectangle([x1+20, y1+20, x2-20, y2-100],
                       fill=c2, outline=colors[3], width=1)
        for i in range(3):
            lx = random.randint(x1+30, x2-30)
            draw.line([(lx, y2-80), (lx, y2-20)],
                      fill=colors[4], width=1)

    elif brand_name == "streetwear":
        bg = colors[random.randint(0, 1)]
        img = Image.new("RGB", (w, h), bg)
        draw = ImageDraw.Draw(img)
        ac = colors[random.randint(2, 5)]
        x1 = random.randint(100, 160)
        y1 = random.randint(60,  100)
        x2 = random.randint(350, 410)
        y2 = random.randint(420, 460)
        draw.rectangle([x1, y1, x2, y2], fill=colors[1])
        draw.rectangle([x1, y1, x2, y1+60], fill=ac)
        for _ in range(random.randint(3, 6)):
            rx1 = random.randint(x1+10, x2-60)
            ry1 = random.randint(y1+80, y2-100)
            rx2 = rx1 + random.randint(40, 120)
            ry2 = ry1 + random.randint(8, 20)
            draw.rectangle([rx1, ry1, rx2, ry2],
                           fill=colors[5], outline=ac, width=1)
        draw.rectangle([x1, y2-80, x2, y2],
                       fill=colors[random.randint(0,1)])

    elif brand_name == "luxury":
        bg = colors[0]
        img = Image.new("RGB", (w, h), bg)
        draw = ImageDraw.Draw(img)
        gold = colors[3]
        x1, y1 = 130, 70
        x2, y2 = 380, 450
        draw.rectangle([x1, y1, x2, y2],
                       fill=colors[1], outline=gold, width=2)
        for i in range(6):
            gy = y1 + 30 + i * 60
            draw.line([(x1+10, gy), (x2-10, gy)],
                      fill=gold, width=1)
        for i in range(5):
            gx = x1 + 30 + i * 48
            draw.line([(gx, y1+10), (gx, y2-10)],
                      fill=gold, width=1)
        cx = (x1 + x2) // 2
        for cy in range(y1+40, y2-40, 60):
            draw.ellipse([cx-8, cy-8, cx+8, cy+8],
                         fill=gold, outline=colors[4])

    img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
    return img

print("\nGenerating brand training datasets...\n")
all_captions = {}

for brand_name, config in BRANDS.items():
    img_dir = f"week4_lora/datasets/{brand_name}/images"
    captions_path = f"week4_lora/datasets/{brand_name}/captions.json"

    real_imgs = []
    src = f"datasets/processed/fashionpedia/train/images"
    if os.path.exists(src):
        all_files = [
            f for f in os.listdir(src)
            if f.endswith(".jpg")
        ]
        sample = random.sample(
            all_files, min(15, len(all_files))
        )
        for i, fname in enumerate(sample):
            dst = os.path.join(img_dir, f"real_{i:03d}.jpg")
            shutil.copy2(os.path.join(src, fname), dst)
            real_imgs.append(dst)

    synth_imgs = []
    for i in range(20):
        img  = create_brand_image(brand_name, config, i)
        path = os.path.join(img_dir, f"synth_{i:03d}.jpg")
        img.save(path, "JPEG", quality=95)
        synth_imgs.append(path)

    total = len(real_imgs) + len(synth_imgs)
    captions = {}
    cap_pool = config["captions"] * 10

    for j, fpath in enumerate(real_imgs + synth_imgs):
        fname = os.path.basename(fpath)
        captions[fname] = cap_pool[j % len(cap_pool)]

    with open(captions_path, "w") as f:
        json.dump(captions, f, indent=2)

    txt_dir = f"week4_lora/datasets/{brand_name}"
    for fname, cap in captions.items():
        stem = os.path.splitext(fname)[0]
        with open(
            os.path.join(txt_dir, f"{stem}.txt"), "w"
        ) as f:
            f.write(cap)

    all_captions[brand_name] = captions
    print(f"  {brand_name:<14}: {total} images | "
          f"token: {config['token']}")

meta = {
    "brands":       list(BRANDS.keys()),
    "tokens":       {k: v["token"] for k,v in BRANDS.items()},
    "descriptions": {k: v["description"] for k,v in BRANDS.items()},
    "image_counts": {
        k: len(os.listdir(
            f"week4_lora/datasets/{k}/images"
        )) for k in BRANDS
    },
}
with open("week4_lora/brand_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n{'='*54}")
print("   DATASET SUMMARY")
print(f"{'='*54}")
for brand in BRANDS:
    n = meta["image_counts"][brand]
    print(f"  {brand:<14}: {n} images")
print(f"\n  Metadata saved : week4_lora/brand_metadata.json")
print(f"  Caption format : brand_token + style description")
print(f"\nTask 1 complete!")

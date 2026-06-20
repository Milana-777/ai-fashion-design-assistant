import os, json, torch
import open_clip
from PIL import Image

print("=" * 54)
print("   CLIP CONSISTENCY EVALUATION")
print("=" * 54)

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.eval()

def clip_score(img_path, prompt):
    img  = preprocess(
        Image.open(img_path).convert("RGB")
    ).unsqueeze(0)
    text = tokenizer([prompt])
    with torch.no_grad():
        i = model.encode_image(img)
        t = model.encode_text(text)
        i = i / i.norm(dim=-1, keepdim=True)
        t = t / t.norm(dim=-1, keepdim=True)
    return round((i @ t.T).item() * 100, 2)

def img_similarity(path1, path2):
    i1 = preprocess(
        Image.open(path1).convert("RGB")
    ).unsqueeze(0)
    i2 = preprocess(
        Image.open(path2).convert("RGB")
    ).unsqueeze(0)
    with torch.no_grad():
        f1 = model.encode_image(i1)
        f2 = model.encode_image(i2)
        f1 = f1 / f1.norm(dim=-1, keepdim=True)
        f2 = f2 / f2.norm(dim=-1, keepdim=True)
    return round((f1 @ f2.T).item() * 100, 2)

BRANDS = {
    "minimalist": "clean minimal fashion white neutral palette",
    "streetwear": "urban streetwear oversized bold graphic",
    "luxury":     "luxury fashion opulent gold embroidery velvet",
}

print(f"\n{'─'*54}")
print(f"  {'Brand':<14} {'Design 1':>10} "
      f"{'Design 2':>10} {'Consistency':>12}")
print(f"{'─'*54}")

report = []
for brand, style_prompt in BRANDS.items():
    p1 = f"week4_lora/outputs/{brand}_design_1.png"
    p2 = f"week4_lora/outputs/{brand}_design_2.png"

    if not os.path.exists(p1) or not os.path.exists(p2):
        print(f"  {brand:<14} {'SKIP — run style_switching.py first':>34}")
        continue

    s1      = clip_score(p1, style_prompt)
    s2      = clip_score(p2, style_prompt)
    consist = img_similarity(p1, p2)

    bar = "█" * int(consist/5) + "░" * (20-int(consist/5))
    print(f"  {brand:<14} {s1:>10.1f} {s2:>10.1f} "
          f"  [{bar}] {consist:.1f}")

    report.append({
        "brand":       brand,
        "score_1":     s1,
        "score_2":     s2,
        "consistency": consist,
    })

if report:
    avg_c = sum(r["consistency"] for r in report)/len(report)
    print(f"{'─'*54}")
    print(f"  {'Average':<14} {'':>10} {'':>10} "
          f"  {'consistency':>12}: {avg_c:.1f}")

out = "week4_lora/clip_consistency_report.json"
with open(out, "w") as f:
    json.dump(report, f, indent=2)

print(f"\n{'='*54}")
print(f"   EVALUATION COMPLETE!")
print(f"{'='*54}")
print(f"\n  Report saved: {out}")
print(f"\nConsistency guide:")
print(f"  > 70 — Very consistent brand style")
print(f"  60-70 — Good consistency")
print(f"  < 60  — Style needs more training data")
print(f"\nWeek 4 Task 4 complete!")

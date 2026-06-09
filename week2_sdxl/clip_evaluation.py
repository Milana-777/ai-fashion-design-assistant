import torch
import open_clip
from PIL import Image
import os, json
import numpy as np

print("=" * 52)
print("   CLIP SCORE EVALUATION — FASHION IMAGES")
print("=" * 52)

print("\nLoading CLIP model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.eval()
print("CLIP model ready!\n")

def get_clip_score(image_path, prompt):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
    text  = tokenizer([prompt])
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features  = model.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)
        score = (image_features @ text_features.T).item()
    return round(score * 100, 2)

# ── Evaluate Task 1 outputs ──────────────────────────────────
TASK1_IMAGES = [
    {
        "path":   "week2_sdxl/outputs/elegant_dress.png",
        "prompt": "a high fashion elegant red evening dress silk fabric floor length fitted silhouette professional fashion photography"
    },
    {
        "path":   "week2_sdxl/outputs/streetwear_jacket.png",
        "prompt": "oversized streetwear jacket urban fashion black and white colorway professional clothing photography"
    },
    {
        "path":   "week2_sdxl/outputs/floral_summer_top.png",
        "prompt": "floral print summer blouse light cotton fabric pastel colors feminine style fashion product photography"
    },
]

# ── Evaluate Task 2 prompt experiments ───────────────────────
TASK2_IMAGES = [
    {
        "path":   "week2_sdxl/prompt_experiments/exp1_bare.png",
        "prompt": "a dress"
    },
    {
        "path":   "week2_sdxl/prompt_experiments/exp2_photo_style.png",
        "prompt": "a dress fashion photography white background studio lighting"
    },
    {
        "path":   "week2_sdxl/prompt_experiments/exp3_full_prompt.png",
        "prompt": "elegant navy blue midi dress satin fabric wrap silhouette v-neckline long sleeves high fashion photography"
    },
    {
        "path":   "week2_sdxl/prompt_experiments/exp4_high_cfg.png",
        "prompt": "elegant navy blue midi dress satin fabric wrap silhouette high fashion photography vogue editorial style"
    },
    {
        "path":   "week2_sdxl/prompt_experiments/exp5_more_steps.png",
        "prompt": "elegant navy blue midi dress satin fabric wrap silhouette high fashion photography vogue editorial"
    },
]

def evaluate_batch(images, label):
    print(f"\n{'─'*52}")
    print(f"  {label}")
    print(f"{'─'*52}")
    results = []
    for item in images:
        if not os.path.exists(item["path"]):
            print(f"  SKIP (not found): {item['path']}")
            continue
        score = get_clip_score(item["path"], item["prompt"])
        name  = os.path.basename(item["path"]).replace(".png","")
        bar   = "█" * int(score / 3) + "░" * (34 - int(score / 3))
        print(f"  {name:<28} [{bar}] {score:.1f}")
        results.append({"name": name, "score": score, "prompt": item["prompt"]})
    if results:
        avg = round(sum(r["score"] for r in results) / len(results), 2)
        print(f"\n  Average CLIP score: {avg}")
        best = max(results, key=lambda x: x["score"])
        print(f"  Best image        : {best['name']} ({best['score']})")
    return results

r1 = evaluate_batch(TASK1_IMAGES, "Task 1 — Initial fashion images")
r2 = evaluate_batch(TASK2_IMAGES, "Task 2 — Prompt experiments")

# ── Prompt quality comparison ─────────────────────────────────
if r2:
    print(f"\n{'─'*52}")
    print("  Prompt complexity vs CLIP score")
    print(f"{'─'*52}")
    labels = ["Bare", "Photo style", "Full prompt", "High CFG", "More steps"]
    for i, (r, lbl) in enumerate(zip(r2, labels)):
        print(f"  {lbl:<14}: {r['score']:.1f}")

# ── Save full report ──────────────────────────────────────────
report = {
    "task1_results": r1,
    "task2_results": r2,
    "summary": {
        "total_images_evaluated": len(r1) + len(r2),
        "best_task1": max(r1, key=lambda x: x["score"])["name"] if r1 else None,
        "best_task2": max(r2, key=lambda x: x["score"])["name"] if r2 else None,
    }
}
report_path = "week2_sdxl/clip_evaluation_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\n{'='*52}")
print("   CLIP EVALUATION COMPLETE!")
print(f"{'='*52}")
print(f"\nReport saved: {report_path}")
print("\nCLIP score guide:")
print("  < 20  — Poor match (prompt ignored)")
print("  20-25 — Moderate match")
print("  25-30 — Good match")
print("  > 30  — Excellent match")
print("\nTask 3 complete!")

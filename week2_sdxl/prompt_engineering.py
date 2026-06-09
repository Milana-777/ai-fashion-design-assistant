import torch
from diffusers import StableDiffusionPipeline
import os, time, json

MODEL_ID   = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "week2_sdxl/prompt_experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading model from cache (fast)...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    safety_checker=None,
)
pipe = pipe.to("cpu")
pipe.enable_attention_slicing()
print("Ready!\n")

# ── Prompt anatomy ──────────────────────────────────────────
# [SUBJECT] + [STYLE] + [FABRIC/MATERIAL] + [COLOR] +
# [PHOTOGRAPHY STYLE] + [QUALITY BOOSTERS]
# ────────────────────────────────────────────────────────────

EXPERIMENTS = [

    # Experiment 1 — bare minimum prompt (baseline)
    {
        "name":     "exp1_bare",
        "positive": "a dress",
        "negative": "",
        "cfg":      7.5,
        "steps":    20,
        "note":     "Baseline — no style guidance"
    },

    # Experiment 2 — add photography style
    {
        "name":     "exp2_photo_style",
        "positive": (
            "a dress, fashion photography, "
            "white background, studio lighting"
        ),
        "negative": "blurry, low quality",
        "cfg":      7.5,
        "steps":    20,
        "note":     "Added photo style keywords"
    },

    # Experiment 3 — full structured prompt
    {
        "name":     "exp3_full_prompt",
        "positive": (
            "elegant navy blue midi dress, "
            "satin fabric, wrap silhouette, "
            "v-neckline, long sleeves, "
            "high fashion photography, "
            "white background, studio lighting, "
            "vogue editorial style, ultra sharp, 4k"
        ),
        "negative": (
            "blurry, low quality, bad anatomy, "
            "watermark, text, ugly, deformed, "
            "extra limbs, bad proportions"
        ),
        "cfg":   7.5,
        "steps": 20,
        "note":  "Full structured fashion prompt"
    },

    # Experiment 4 — high CFG scale (more prompt adherence)
    {
        "name":     "exp4_high_cfg",
        "positive": (
            "elegant navy blue midi dress, "
            "satin fabric, wrap silhouette, "
            "v-neckline, long sleeves, "
            "high fashion photography, "
            "white background, studio lighting, "
            "vogue editorial style, ultra sharp, 4k"
        ),
        "negative": (
            "blurry, low quality, bad anatomy, "
            "watermark, text, ugly, deformed"
        ),
        "cfg":   12.0,
        "steps": 20,
        "note":  "High CFG=12 — stronger prompt adherence"
    },

    # Experiment 5 — more steps (better quality)
    {
        "name":     "exp5_more_steps",
        "positive": (
            "elegant navy blue midi dress, "
            "satin fabric, wrap silhouette, "
            "high fashion photography, "
            "white background, studio lighting, "
            "vogue editorial, ultra sharp, 4k"
        ),
        "negative": (
            "blurry, low quality, bad anatomy, "
            "watermark, text, ugly, deformed"
        ),
        "cfg":   7.5,
        "steps": 30,
        "note":  "More steps=30 — higher quality"
    },
]

results = []

for exp in EXPERIMENTS:
    print(f"Running {exp['name']}...")
    print(f"  Note   : {exp['note']}")
    print(f"  CFG    : {exp['cfg']}  Steps: {exp['steps']}")

    start = time.time()
    image = pipe(
        prompt              = exp["positive"],
        negative_prompt     = exp["negative"],
        guidance_scale      = exp["cfg"],
        num_inference_steps = exp["steps"],
        height=512, width=512,
    ).images[0]

    out_path = os.path.join(OUTPUT_DIR, f"{exp['name']}.png")
    image.save(out_path)
    elapsed = time.time() - start

    results.append({**exp, "time_sec": round(elapsed), "output": out_path})
    print(f"  Saved  : {out_path}  ({elapsed:.0f}s)\n")

# Save experiment log
log_path = os.path.join(OUTPUT_DIR, "experiment_log.json")
with open(log_path, "w") as f:
    json.dump(results, f, indent=2)

print("=" * 50)
print("   PROMPT EXPERIMENTS COMPLETE!")
print("=" * 50)
print(f"\nResults saved: {log_path}")
print("\nPrompt anatomy recap:")
print("  [SUBJECT] + [FABRIC] + [COLOR] + [SILHOUETTE]")
print("  + [PHOTO STYLE] + [BACKGROUND] + [QUALITY TAGS]")
print("\nKey parameters:")
print("  CFG scale  : 7-8 = balanced | 10-12 = strict prompt")
print("  Steps      : 20 = fast | 30 = quality | 50 = best")
print("\nTask 2 complete!")

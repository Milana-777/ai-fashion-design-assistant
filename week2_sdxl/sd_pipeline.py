import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os, time

MODEL_ID  = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "week2_sdxl/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 50)
print("   STABLE DIFFUSION — FASHION PIPELINE")
print("=" * 50)
print(f"\nLoading model: {MODEL_ID}")
print("This downloads ~4GB on first run — please wait...\n")

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    safety_checker=None,
)
pipe = pipe.to("cpu")

pipe.enable_attention_slicing()

print("Model loaded!\n")

FASHION_PROMPTS = [
    {
        "name": "elegant_dress",
        "positive": (
            "a high fashion elegant red evening dress, "
            "silk fabric, floor length, fitted silhouette, "
            "professional fashion photography, studio lighting, "
            "white background, vogue magazine style, "
            "ultra detailed, 4k"
        ),
        "negative": (
            "blurry, low quality, bad anatomy, "
            "watermark, text, ugly, deformed"
        ),
    },
    {
        "name": "streetwear_jacket",
        "positive": (
            "oversized streetwear jacket, urban fashion, "
            "black and white colorway, hoodie underneath, "
            "professional clothing photography, clean background, "
            "high resolution, editorial style"
        ),
        "negative": (
            "blurry, low quality, watermark, "
            "text, bad anatomy, deformed"
        ),
    },
    {
        "name": "floral_summer_top",
        "positive": (
            "floral print summer blouse, light cotton fabric, "
            "pastel colors, feminine style, "
            "fashion product photography, white background, "
            "sharp focus, professional lighting"
        ),
        "negative": (
            "blurry, low quality, watermark, "
            "text, bad proportions"
        ),
    },
]

for item in FASHION_PROMPTS:
    print(f"Generating: {item['name']}...")
    start = time.time()

    image = pipe(
        prompt          = item["positive"],
        negative_prompt = item["negative"],
        num_inference_steps = 20,
        guidance_scale      = 7.5,
        height = 512,
        width  = 512,
    ).images[0]

    out_path = os.path.join(OUTPUT_DIR, f"{item['name']}.png")
    image.save(out_path)
    elapsed = time.time() - start
    print(f"  Saved: {out_path}  ({elapsed:.0f}s)\n")

print("=" * 50)
print("   ALL IMAGES GENERATED!")
print("=" * 50)
print(f"\nOutputs saved to: {OUTPUT_DIR}/")
print("Task 1 complete!")

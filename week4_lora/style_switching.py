import os, json, time, torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from peft import PeftModel

print("=" * 54)
print("   STYLE SWITCHING — LORA ADAPTERS")
print("=" * 54)

MODEL_ID   = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "week4_lora/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BRANDS = {
    "minimalist": {
        "adapter": "week4_lora/adapters/minimalist",
        "token":   "<minimalist-style>",
        "prompts": [
            "<minimalist-style> elegant white dress, "
            "clean lines, neutral palette, "
            "fashion photography, white background",
            "<minimalist-style> beige structured blazer, "
            "minimal design, understated, editorial style",
        ],
    },
    "streetwear": {
        "adapter": "week4_lora/adapters/streetwear",
        "token":   "<streetwear-style>",
        "prompts": [
            "<streetwear-style> oversized black hoodie, "
            "bold graphic, urban fashion, "
            "street photography style",
            "<streetwear-style> cargo pants outfit, "
            "layered look, city background, editorial",
        ],
    },
    "luxury": {
        "adapter": "week4_lora/adapters/luxury",
        "token":   "<luxury-style>",
        "prompts": [
            "<luxury-style> black velvet evening gown, "
            "gold embroidery, opulent details, "
            "high fashion photography",
            "<luxury-style> silk brocade jacket, "
            "jewel tones, couture quality, editorial luxury",
        ],
    },
}

NEGATIVE = (
    "blurry, low quality, bad anatomy, "
    "watermark, text, ugly, deformed"
)

results = []

for brand_name, config in BRANDS.items():
    adapter_path = config["adapter"]
    if not os.path.exists(adapter_path):
        print(f"\n  SKIP {brand_name} — adapter not found")
        print(f"  Train it first with train_lora_lite.py")
        continue

    print(f"\nLoading {brand_name} adapter...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        safety_checker=None,
    )
    pipe.unet = PeftModel.from_pretrained(
        pipe.unet, adapter_path
    )
    pipe = pipe.to("cpu")
    pipe.enable_attention_slicing()
    print(f"  Adapter loaded!")

    for i, prompt in enumerate(config["prompts"]):
        print(f"  Generating prompt {i+1}/2...")
        start = time.time()

        image = pipe(
            prompt              = prompt,
            negative_prompt     = NEGATIVE,
            num_inference_steps = 20,
            guidance_scale      = 7.5,
            height=512, width=512,
        ).images[0]

        out_path = (
            f"{OUTPUT_DIR}/"
            f"{brand_name}_design_{i+1}.png"
        )
        image.save(out_path)
        elapsed = time.time() - start

        results.append({
            "brand":   brand_name,
            "prompt":  prompt,
            "output":  out_path,
            "time":    round(elapsed),
        })
        print(f"  Saved: {out_path} ({elapsed:.0f}s)")

    del pipe
    torch.cuda.empty_cache() if torch.cuda.is_available() \
        else None

log_path = "week4_lora/style_switching_log.json"
with open(log_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*54}")
print(f"   STYLE SWITCHING COMPLETE!")
print(f"{'='*54}")
print(f"\n  Generated : {len(results)} designs")
print(f"  Brands    : {list(BRANDS.keys())}")
print(f"  Log saved : {log_path}")
print(f"\nTask 3 complete!")

import torch, json, os
from diffusers import StableDiffusionPipeline

with open("week2_sdxl/prompt_library.json") as f:
    library = json.load(f)

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None,
)
pipe = pipe.to("cpu")
pipe.enable_attention_slicing()

os.makedirs("week2_sdxl/library_outputs", exist_ok=True)

TO_GENERATE = ["elegant_evening", "casual_summer", "streetwear_set"]

for name in TO_GENERATE:
    t = library["named_templates"][name]
    print(f"Generating: {name}...")
    image = pipe(
        prompt              = t["positive"],
        negative_prompt     = t["negative"],
        num_inference_steps = 20,
        guidance_scale      = 7.5,
        height=512, width=512,
    ).images[0]
    out = f"week2_sdxl/library_outputs/{name}.png"
    image.save(out)
    print(f"  Saved: {out}\n")

print("Library generation complete!")

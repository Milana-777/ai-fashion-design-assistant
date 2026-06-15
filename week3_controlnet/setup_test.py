import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
import os, time

print("=" * 52)
print("   CONTROLNET SETUP TEST")
print("=" * 52)

print("\n1. Checking imports...")
print(f"   OpenCV   : {cv2.__version__}")
print(f"   PyTorch  : {torch.__version__}")
print(f"   Diffusers: OK")

print("\n2. Loading ControlNet canny model...")
print("   (Downloads ~1.4GB on first run...)\n")

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float32,
)

print("3. Loading SD pipeline with ControlNet...")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float32,
    safety_checker=None,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config
)
pipe.enable_attention_slicing()
print("   Pipeline ready!\n")

print("4. Creating test sketch (synthetic edge map)...")
canvas = np.zeros((512, 512), dtype=np.uint8)
cv2.rectangle(canvas, (150, 80),  (360, 420), 255, 2)
cv2.ellipse(canvas,   (255, 60),  (60, 60),   0, 0, 360, 255, 2)
cv2.line(canvas, (150, 200), (100, 380), 255, 2)
cv2.line(canvas, (360, 200), (410, 380), 255, 2)
cv2.line(canvas, (150, 420), (200, 500), 255, 2)
cv2.line(canvas, (360, 420), (310, 500), 255, 2)
cv2.line(canvas, (150, 160), (80,  200), 255, 2)
cv2.line(canvas, (360, 160), (430, 200), 255, 2)

edge_img = Image.fromarray(canvas).convert("RGB")
edge_path = "week3_controlnet/edge_maps/test_sketch.png"
edge_img.save(edge_path)
print(f"   Saved: {edge_path}")

print("\n5. Generating fashion image from sketch...")
start = time.time()
image = pipe(
    prompt=(
        "fashion illustration, elegant dress, "
        "high fashion photography, white background, "
        "professional studio lighting, vogue style"
    ),
    negative_prompt="blurry, low quality, deformed, ugly",
    image=edge_img,
    num_inference_steps=20,
    guidance_scale=7.5,
    controlnet_conditioning_scale=1.0,
    height=512, width=512,
).images[0]

out_path = "week3_controlnet/outputs/test_output.png"
image.save(out_path)
elapsed = time.time() - start

print(f"   Saved: {out_path}  ({elapsed:.0f}s)")
print(f"\n{'='*52}")
print("   CONTROLNET SETUP COMPLETE!")
print(f"{'='*52}")
print("\nTask 1 complete!")

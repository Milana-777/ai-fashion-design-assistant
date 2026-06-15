import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
import os, time, json

print("=" * 54)
print("   SKETCH TO DESIGN — CONTROLNET PIPELINE")
print("=" * 54)

# ── Load pipeline ─────────────────────────────────────────────
print("\nLoading pipeline from cache...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float32,
)
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
print("Pipeline ready!\n")

# ── Edge map extractor ────────────────────────────────────────
def extract_canny_edges(image_path, low=100, high=200):
    img   = cv2.imread(image_path)
    img   = cv2.resize(img, (512, 512))
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, low, high)
    return Image.fromarray(edges).convert("RGB")

# ── Synthetic fashion sketch generator ───────────────────────
def make_fashion_sketch(style="dress", save_path=None):
    canvas = np.zeros((512, 512), dtype=np.uint8)

    if style == "dress":
        # Body
        cv2.rectangle(canvas, (180, 80), (330, 440), 255, 2)
        # Head
        cv2.ellipse(canvas, (255, 55), (45, 45), 0, 0, 360, 255, 2)
        # Waist
        cv2.line(canvas, (180, 220), (330, 220), 255, 1)
        # Flare skirt lines
        cv2.line(canvas, (180, 220), (110, 440), 255, 2)
        cv2.line(canvas, (330, 220), (400, 440), 255, 2)
        # Neckline V-shape
        cv2.line(canvas, (195, 100), (255, 160), 255, 2)
        cv2.line(canvas, (315, 100), (255, 160), 255, 2)
        # Arms
        cv2.line(canvas, (180, 130), (110, 260), 255, 2)
        cv2.line(canvas, (330, 130), (400, 260), 255, 2)

    elif style == "jacket":
        # Body
        cv2.rectangle(canvas, (160, 90), (350, 350), 255, 2)
        # Head
        cv2.ellipse(canvas, (255, 60), (45, 45), 0, 0, 360, 255, 2)
        # Lapels
        cv2.line(canvas, (200, 100), (255, 180), 255, 2)
        cv2.line(canvas, (310, 100), (255, 180), 255, 2)
        # Collar
        cv2.line(canvas, (200, 100), (190, 130), 255, 2)
        cv2.line(canvas, (310, 100), (320, 130), 255, 2)
        # Arms
        cv2.rectangle(canvas, (90, 110), (162, 300), 255, 2)
        cv2.rectangle(canvas, (348, 110), (420, 300), 255, 2)
        # Buttons
        for y in range(195, 345, 40):
            cv2.circle(canvas, (255, y), 5, 255, -1)
        # Pockets
        cv2.rectangle(canvas, (175, 270), (230, 310), 255, 1)
        cv2.rectangle(canvas, (280, 270), (335, 310), 255, 1)
        # Legs
        cv2.rectangle(canvas, (170, 350), (245, 500), 255, 2)
        cv2.rectangle(canvas, (265, 350), (340, 500), 255, 2)

    elif style == "top":
        # Body
        cv2.rectangle(canvas, (185, 100), (325, 300), 255, 2)
        # Head
        cv2.ellipse(canvas, (255, 65), (45, 45), 0, 0, 360, 255, 2)
        # Round neckline
        cv2.ellipse(canvas, (255, 100), (50, 25), 0, 0, 180, 255, 2)
        # Short sleeves
        cv2.line(canvas, (185, 130), (120, 190), 255, 2)
        cv2.line(canvas, (120, 190), (155, 200), 255, 2)
        cv2.line(canvas, (155, 200), (185, 160), 255, 2)
        cv2.line(canvas, (325, 130), (390, 190), 255, 2)
        cv2.line(canvas, (390, 190), (355, 200), 255, 2)
        cv2.line(canvas, (355, 200), (325, 160), 255, 2)
        # Pants
        cv2.rectangle(canvas, (175, 300), (248, 480), 255, 2)
        cv2.rectangle(canvas, (262, 300), (335, 480), 255, 2)
        cv2.line(canvas, (175, 300), (335, 300), 255, 2)

    img = Image.fromarray(canvas).convert("RGB")
    if save_path:
        img.save(save_path)
    return img

# ── Fashion generation prompts ────────────────────────────────
DESIGNS = [
    {
        "name":    "evening_dress",
        "sketch":  "dress",
        "prompt":  (
            "elegant evening dress, deep red silk fabric, "
            "flared A-line skirt, V-neckline, "
            "high fashion photography, studio lighting, "
            "white background, vogue editorial, ultra sharp"
        ),
        "negative": "blurry, ugly, deformed, low quality, watermark",
        "cfg":   8.0,
        "cnet":  1.0,
        "steps": 20,
    },
    {
        "name":    "power_suit",
        "sketch":  "jacket",
        "prompt":  (
            "tailored power suit jacket, navy blue wool fabric, "
            "structured shoulders, business professional, "
            "high fashion photography, white background, "
            "editorial style, sharp focus, 4k"
        ),
        "negative": "blurry, ugly, deformed, low quality, casual",
        "cfg":   8.0,
        "cnet":  0.9,
        "steps": 20,
    },
    {
        "name":    "casual_top",
        "sketch":  "top",
        "prompt":  (
            "casual cotton crop top, soft white fabric, "
            "minimalist design, streetwear style, "
            "fashion product photography, clean background, "
            "professional lighting, sharp details"
        ),
        "negative": "blurry, ugly, deformed, low quality, formal",
        "cfg":   7.5,
        "cnet":  1.0,
        "steps": 20,
    },
]

results = []

for design in DESIGNS:
    print(f"Processing: {design['name']}...")

    # Step A — generate sketch
    sketch_path = f"week3_controlnet/sketches/{design['name']}_sketch.png"
    edge_map    = make_fashion_sketch(design["sketch"], sketch_path)
    print(f"  Sketch saved : {sketch_path}")

    # Step B — extract clean edge map
    edge_path = f"week3_controlnet/edge_maps/{design['name']}_edges.png"
    edge_map.save(edge_path)
    print(f"  Edge map     : {edge_path}")

    # Step C — generate design
    print(f"  Generating...")
    start = time.time()
    image = pipe(
        prompt                     = design["prompt"],
        negative_prompt            = design["negative"],
        image                      = edge_map,
        num_inference_steps        = design["steps"],
        guidance_scale             = design["cfg"],
        controlnet_conditioning_scale = design["cnet"],
        height=512, width=512,
    ).images[0]

    out_path = f"week3_controlnet/outputs/{design['name']}.png"
    image.save(out_path)
    elapsed = time.time() - start

    results.append({
        "name":    design["name"],
        "sketch":  sketch_path,
        "edges":   edge_path,
        "output":  out_path,
        "time_sec": round(elapsed),
    })
    print(f"  Output saved : {out_path}  ({elapsed:.0f}s)\n")

# ── Also test with a real dataset image ───────────────────────
print("Testing with real dataset image...")
import glob
real_imgs = glob.glob(
    "datasets/processed/fashionpedia/train/images/*.jpg"
)
if real_imgs:
    src_path  = real_imgs[0]
    real_edge = extract_canny_edges(src_path, low=80, high=180)
    real_edge_path = "week3_controlnet/edge_maps/real_image_edges.png"
    real_edge.save(real_edge_path)

    real_out = pipe(
        prompt=(
            "high fashion garment, professional photography, "
            "white background, sharp focus, editorial style"
        ),
        negative_prompt="blurry, low quality, ugly",
        image=real_edge,
        num_inference_steps=20,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.8,
        height=512, width=512,
    ).images[0]
    real_out_path = "week3_controlnet/outputs/real_image_redesign.png"
    real_out.save(real_out_path)
    print(f"  Real image redesign saved: {real_out_path}\n")
else:
    print("  No dataset images found — skipping real image test\n")

# ── Save results log ──────────────────────────────────────────
log_path = "week3_controlnet/sketch_to_design_log.json"
with open(log_path, "w") as f:
    json.dump(results, f, indent=2)

print("=" * 54)
print("   SKETCH TO DESIGN COMPLETE!")
print("=" * 54)
print(f"\nGenerated {len(results)} fashion designs")
print(f"Log saved : {log_path}")
print(f"Outputs   : week3_controlnet/outputs/")
print("\nTask 2 complete!")

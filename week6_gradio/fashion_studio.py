import gradio as gr
import torch
import numpy as np
import cv2
import json
import os
import time
from PIL import Image, ImageDraw
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
import open_clip
import chromadb

print("Loading AI Fashion Creative Studio...")

MODEL_ID = "runwayml/stable-diffusion-v1-5"
NEGATIVE = (
    "blurry, low quality, bad anatomy, "
    "watermark, text, ugly, deformed"
)

print("  Loading CLIP model...")
clip_model, _, clip_preprocess = (
    open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
)
clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
clip_model.eval()

print("  Loading ChromaDB...")
try:
    chroma_client = chromadb.PersistentClient(
        path="week5_rag/chromadb"
    )
    img_col = chroma_client.get_collection("fashion_images")
    HAS_CHROMA = True
except Exception:
    HAS_CHROMA = False
    print("  ChromaDB not found — RAG disabled")

with open("week5_rag/knowledge_base.json") as f:
    KB = json.load(f)

sd_pipe       = None
controlnet_pipe = None

def load_sd():
    global sd_pipe
    if sd_pipe is None:
        print("  Loading SD pipeline...")
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            safety_checker=None,
        )
        sd_pipe.enable_attention_slicing()
    return sd_pipe

def load_controlnet():
    global controlnet_pipe
    if controlnet_pipe is None:
        print("  Loading ControlNet pipeline...")
        cn = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float32,
        )
        controlnet_pipe = (
            StableDiffusionControlNetPipeline.from_pretrained(
                MODEL_ID,
                controlnet=cn,
                torch_dtype=torch.float32,
                safety_checker=None,
            )
        )
        controlnet_pipe.scheduler = (
            UniPCMultistepScheduler.from_config(
                controlnet_pipe.scheduler.config
            )
        )
        controlnet_pipe.enable_attention_slicing()
    return controlnet_pipe

# ── Tab 1: Sketch2Design ──────────────────────────────────────
def sketch_to_design(sketch_img, prompt, cfg, steps, cnet_scale):
    if sketch_img is None:
        return None, "Please upload a sketch image."

    pipe  = load_controlnet()
    start = time.time()

    if isinstance(sketch_img, np.ndarray):
        pil_img = Image.fromarray(sketch_img)
    else:
        pil_img = sketch_img

    gray  = cv2.cvtColor(
        np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY
    )
    blur  = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 100, 200)
    edge_img = Image.fromarray(edges).convert("RGB")
    edge_img = edge_img.resize((512, 512))

    full_prompt = (
        f"{prompt}, high fashion photography, "
        f"white background, editorial style, sharp focus"
    )
    result = pipe(
        prompt=full_prompt,
        negative_prompt=NEGATIVE,
        image=edge_img,
        num_inference_steps=int(steps),
        guidance_scale=cfg,
        controlnet_conditioning_scale=cnet_scale,
        height=512, width=512,
    ).images[0]

    elapsed = time.time() - start
    return result, f"Generated in {elapsed:.0f}s"

# ── Tab 2: StyleMixer ─────────────────────────────────────────
STYLE_PROMPTS = {
    "Minimalist":  "clean minimal fashion, neutral palette, white background, editorial",
    "Streetwear":  "urban streetwear, oversized, bold graphic, city background",
    "Luxury":      "luxury fashion, opulent, gold detail, velvet, high fashion",
    "Bohemian":    "boho chic, floral, flowy, earthy tones, natural light",
    "Avant-garde": "avant-garde fashion, experimental, artistic, dramatic",
}

def style_mixer(garment, style, color, occasion, cfg, steps):
    pipe  = load_sd()
    start = time.time()

    style_desc = STYLE_PROMPTS.get(style, "fashion")
    prompt = (
        f"{color} {garment}, {style_desc}, "
        f"{occasion} occasion, fashion photography, "
        f"white background, sharp focus, 4k"
    )
    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE,
        num_inference_steps=int(steps),
        guidance_scale=cfg,
        height=512, width=512,
    ).images[0]

    elapsed = time.time() - start
    return result, f"Style: {style} | {elapsed:.0f}s"

# ── Tab 3: WardrobeGen ────────────────────────────────────────
def wardrobe_gen(occasion, season, persona, extra):
    pipe  = load_sd()
    start = time.time()

    persona_map = {
        "Classic":    "timeless classic style, structured",
        "Trendy":     "on-trend fashion forward look",
        "Minimalist": "clean minimal understated elegance",
        "Bold":       "bold statement maximalist look",
    }

    prompt = (
        f"complete {occasion} outfit, {season} season, "
        f"{persona_map.get(persona,'stylish')}, "
        f"{extra}, full body fashion photography, "
        f"white background, editorial quality"
    )
    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE,
        num_inference_steps=20,
        guidance_scale=7.5,
        height=512, width=512,
    ).images[0]

    elapsed = time.time() - start
    return result, f"Outfit generated in {elapsed:.0f}s"

# ── Tab 4: Trend explorer ─────────────────────────────────────
def trend_explorer(query):
    trend_matches = []
    for trend in KB["trends_2025"]:
        if any(
            kw in query.lower()
            for kw in trend["keywords"]
        ):
            trend_matches.append(trend["trend"])

    persona_matches = []
    for p in KB["style_personas"]:
        if p["persona"].lower() in query.lower():
            persona_matches.append({
                "persona":     p["persona"],
                "description": p["description"],
                "key_pieces":  p["key_pieces"],
            })

    palette_matches = []
    for pal in KB["color_palettes"]:
        if any(
            c in query.lower()
            for c in pal["colors"]
        ):
            palette_matches.append(pal)

    similar_imgs = []
    if HAS_CHROMA:
        tok = clip_tokenizer([query])
        with torch.no_grad():
            feat = clip_model.encode_text(tok)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        results = img_col.query(
            query_embeddings=feat.squeeze().numpy().tolist(),
            n_results=3,
        )
        similar_imgs = results["ids"][0]

    out = f"Query: {query}\n\n"
    out += f"Trend matches: {', '.join(trend_matches) if trend_matches else 'None found'}\n\n"

    if persona_matches:
        p = persona_matches[0]
        out += f"Style persona: {p['persona']}\n"
        out += f"Description  : {p['description']}\n"
        out += f"Key pieces   : {', '.join(p['key_pieces'])}\n\n"

    if palette_matches:
        out += f"Color palette: {palette_matches[0]['name']}\n"
        out += f"Colors       : {', '.join(palette_matches[0]['colors'][:4])}\n\n"

    if similar_imgs:
        out += f"Similar images found: {len(similar_imgs)}\n"
        out += "\n".join(f"  - {i}" for i in similar_imgs)

    return out

# ── Build Gradio app ──────────────────────────────────────────
with gr.Blocks(
    title="AI Fashion Creative Studio",
    theme=gr.themes.Soft(),
) as app:

    gr.Markdown("# AI Fashion Creative Studio")
    gr.Markdown(
        "Powered by Stable Diffusion, ControlNet, "
        "LoRA, CLIP and ChromaDB"
    )

    with gr.Tabs():

        # Tab 1 — Sketch2Design
        with gr.Tab("Sketch2Design"):
            gr.Markdown("### Upload a sketch and generate a fashion design")
            with gr.Row():
                with gr.Column():
                    sketch_input = gr.Image(
                        label="Upload sketch or draw",
                        type="numpy",
                        height=300,
                    )
                    s2d_prompt = gr.Textbox(
                        label="Fashion prompt",
                        value="elegant dress, silk fabric, evening wear",
                        lines=2,
                    )
                    with gr.Row():
                        s2d_cfg   = gr.Slider(5, 15, value=7.5, label="CFG scale")
                        s2d_steps = gr.Slider(10, 40, value=20, step=1, label="Steps")
                        s2d_cnet  = gr.Slider(0.5, 1.5, value=1.0, label="ControlNet strength")
                    s2d_btn = gr.Button("Generate design", variant="primary")
                with gr.Column():
                    s2d_output = gr.Image(label="Generated design", height=300)
                    s2d_info   = gr.Textbox(label="Info", lines=1)

            s2d_btn.click(
                sketch_to_design,
                inputs=[sketch_input, s2d_prompt, s2d_cfg, s2d_steps, s2d_cnet],
                outputs=[s2d_output, s2d_info],
            )

        # Tab 2 — StyleMixer
        with gr.Tab("StyleMixer"):
            gr.Markdown("### Mix brand styles to generate fashion designs")
            with gr.Row():
                with gr.Column():
                    sm_garment  = gr.Textbox(label="Garment type", value="dress")
                    sm_style    = gr.Dropdown(
                        list(STYLE_PROMPTS.keys()),
                        label="Brand style",
                        value="Minimalist",
                    )
                    sm_color    = gr.Textbox(label="Color", value="black")
                    sm_occasion = gr.Dropdown(
                        ["casual","formal","evening","streetwear","beach"],
                        label="Occasion",
                        value="evening",
                    )
                    with gr.Row():
                        sm_cfg   = gr.Slider(5, 15, value=8.0, label="CFG scale")
                        sm_steps = gr.Slider(10, 40, value=20, step=1, label="Steps")
                    sm_btn = gr.Button("Mix styles", variant="primary")
                with gr.Column():
                    sm_output = gr.Image(label="Generated design", height=350)
                    sm_info   = gr.Textbox(label="Info", lines=1)

            sm_btn.click(
                style_mixer,
                inputs=[sm_garment, sm_style, sm_color, sm_occasion, sm_cfg, sm_steps],
                outputs=[sm_output, sm_info],
            )

        # Tab 3 — WardrobeGen
        with gr.Tab("WardrobeGen"):
            gr.Markdown("### Generate a complete outfit for any occasion")
            with gr.Row():
                with gr.Column():
                    wg_occasion = gr.Dropdown(
                        ["business meeting","evening gala","casual weekend",
                         "beach holiday","date night","gym session"],
                        label="Occasion",
                        value="evening gala",
                    )
                    wg_season  = gr.Dropdown(
                        ["spring","summer","autumn","winter"],
                        label="Season",
                        value="summer",
                    )
                    wg_persona = gr.Dropdown(
                        ["Classic","Trendy","Minimalist","Bold"],
                        label="Style persona",
                        value="Classic",
                    )
                    wg_extra = gr.Textbox(
                        label="Extra details (optional)",
                        placeholder="e.g. floral print, pastel colors",
                        lines=2,
                    )
                    wg_btn = gr.Button("Generate outfit", variant="primary")
                with gr.Column():
                    wg_output = gr.Image(label="Generated outfit", height=350)
                    wg_info   = gr.Textbox(label="Info", lines=1)

            wg_btn.click(
                wardrobe_gen,
                inputs=[wg_occasion, wg_season, wg_persona, wg_extra],
                outputs=[wg_output, wg_info],
            )

        # Tab 4 — Trend explorer
        with gr.Tab("Trend explorer"):
            gr.Markdown("### Explore fashion trends and get style recommendations")
            with gr.Row():
                with gr.Column():
                    tr_query = gr.Textbox(
                        label="Describe your style or ask about a trend",
                        placeholder="e.g. minimalist summer outfit neutral tones",
                        lines=3,
                    )
                    tr_btn = gr.Button("Explore trends", variant="primary")
                with gr.Column():
                    tr_output = gr.Textbox(
                        label="Trend analysis",
                        lines=12,
                    )

            tr_btn.click(
                trend_explorer,
                inputs=[tr_query],
                outputs=[tr_output],
            )

print("\nAll systems ready!")
app.launch(share=False, server_port=7860)

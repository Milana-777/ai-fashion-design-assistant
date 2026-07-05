import gradio as gr
import torch
import numpy as np
import cv2
import json
import os
import time
from PIL import Image
from diffusers import StableDiffusionPipeline
import open_clip
import chromadb

print("Loading AI Fashion Creative Studio (lite mode)...")

MODEL_ID = "runwayml/stable-diffusion-v1-5"
NEGATIVE = "blurry, low quality, bad anatomy, watermark, text, ugly, deformed"

print("  Loading CLIP...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
clip_model.eval()

print("  Loading ChromaDB...")
try:
    chroma_client = chromadb.PersistentClient(path="week5_rag/chromadb")
    img_col = chroma_client.get_collection("fashion_images")
    HAS_CHROMA = True
except Exception:
    HAS_CHROMA = False

with open("week5_rag/knowledge_base.json") as f:
    KB = json.load(f)

sd_pipe = None

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

def sketch_to_design(sketch_img, prompt, cfg, steps, cnet_scale):
    if sketch_img is None:
        return None, "Please upload a sketch image."
    pipe  = load_sd()
    start = time.time()
    full_prompt = (
        f"{prompt}, inspired by fashion sketch, "
        f"high fashion photography, white background, "
        f"editorial style, sharp focus, vogue"
    )
    result = pipe(
        prompt=full_prompt,
        negative_prompt=NEGATIVE,
        num_inference_steps=int(steps),
        guidance_scale=cfg,
        height=512, width=512,
    ).images[0]
    elapsed = time.time() - start
    return result, f"Generated in {elapsed:.0f}s"

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

def trend_explorer(query):
    trend_matches = []
    for trend in KB["trends_2025"]:
        if any(kw in query.lower() for kw in trend["keywords"]):
            trend_matches.append(trend["trend"])
    persona_matches = []
    for p in KB["style_personas"]:
        if p["persona"].lower() in query.lower():
            persona_matches.append(p)
    palette_matches = []
    for pal in KB["color_palettes"]:
        if any(c in query.lower() for c in pal["colors"]):
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
        out += f"Style persona : {p['persona']}\n"
        out += f"Description   : {p['description']}\n"
        out += f"Key pieces    : {', '.join(p['key_pieces'])}\n\n"
    if palette_matches:
        out += f"Color palette : {palette_matches[0]['name']}\n"
        out += f"Colors        : {', '.join(palette_matches[0]['colors'][:4])}\n\n"
    if similar_imgs:
        out += f"Similar images: {len(similar_imgs)} found\n"
        out += "\n".join(f"  - {i}" for i in similar_imgs)
    return out

with gr.Blocks(title="AI Fashion Creative Studio") as app:
    gr.Markdown("# AI Fashion Creative Studio")
    gr.Markdown("Powered by Stable Diffusion, CLIP and ChromaDB")

    with gr.Tabs():
        with gr.Tab("Sketch2Design"):
            gr.Markdown("### Upload a sketch and generate a fashion design")
            with gr.Row():
                with gr.Column():
                    sketch_input = gr.Image(label="Upload sketch", type="numpy", height=300)
                    s2d_prompt   = gr.Textbox(label="Fashion prompt", value="elegant dress, silk fabric, evening wear", lines=2)
                    with gr.Row():
                        s2d_cfg   = gr.Slider(5, 15, value=7.5, label="CFG scale")
                        s2d_steps = gr.Slider(10, 40, value=20, step=1, label="Steps")
                        s2d_cnet  = gr.Slider(0.5, 1.5, value=1.0, label="Strength")
                    s2d_btn = gr.Button("Generate design", variant="primary")
                with gr.Column():
                    s2d_output = gr.Image(label="Generated design", height=300)
                    s2d_info   = gr.Textbox(label="Info", lines=1)
            s2d_btn.click(
                sketch_to_design,
                inputs=[sketch_input, s2d_prompt, s2d_cfg, s2d_steps, s2d_cnet],
                outputs=[s2d_output, s2d_info],
            )

        with gr.Tab("StyleMixer"):
            gr.Markdown("### Mix brand styles to generate fashion designs")
            with gr.Row():
                with gr.Column():
                    sm_garment  = gr.Textbox(label="Garment", value="dress")
                    sm_style    = gr.Dropdown(list(STYLE_PROMPTS.keys()), label="Style", value="Minimalist")
                    sm_color    = gr.Textbox(label="Color", value="black")
                    sm_occasion = gr.Dropdown(["casual","formal","evening","streetwear","beach"], label="Occasion", value="evening")
                    with gr.Row():
                        sm_cfg   = gr.Slider(5, 15, value=8.0, label="CFG")
                        sm_steps = gr.Slider(10, 40, value=20, step=1, label="Steps")
                    sm_btn = gr.Button("Mix styles", variant="primary")
                with gr.Column():
                    sm_output = gr.Image(label="Generated design", height=350)
                    sm_info   = gr.Textbox(label="Info", lines=1)
            sm_btn.click(style_mixer, inputs=[sm_garment, sm_style, sm_color, sm_occasion, sm_cfg, sm_steps], outputs=[sm_output, sm_info])

        with gr.Tab("WardrobeGen"):
            gr.Markdown("### Generate a complete outfit for any occasion")
            with gr.Row():
                with gr.Column():
                    wg_occasion = gr.Dropdown(["business meeting","evening gala","casual weekend","beach holiday","date night"], label="Occasion", value="evening gala")
                    wg_season   = gr.Dropdown(["spring","summer","autumn","winter"], label="Season", value="summer")
                    wg_persona  = gr.Dropdown(["Classic","Trendy","Minimalist","Bold"], label="Persona", value="Classic")
                    wg_extra    = gr.Textbox(label="Extra details", placeholder="e.g. floral print", lines=2)
                    wg_btn      = gr.Button("Generate outfit", variant="primary")
                with gr.Column():
                    wg_output = gr.Image(label="Generated outfit", height=350)
                    wg_info   = gr.Textbox(label="Info", lines=1)
            wg_btn.click(wardrobe_gen, inputs=[wg_occasion, wg_season, wg_persona, wg_extra], outputs=[wg_output, wg_info])

        with gr.Tab("Trend explorer"):
            gr.Markdown("### Explore fashion trends and get style recommendations")
            with gr.Row():
                with gr.Column():
                    tr_query = gr.Textbox(label="Describe your style", placeholder="e.g. minimalist summer neutral tones", lines=3)
                    tr_btn   = gr.Button("Explore trends", variant="primary")
                with gr.Column():
                    tr_output = gr.Textbox(label="Trend analysis", lines=12)
            tr_btn.click(trend_explorer, inputs=[tr_query], outputs=[tr_output])

print("\nAll systems ready!")
app.launch(server_port=7860)

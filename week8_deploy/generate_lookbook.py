import torch, os, json, time
import open_clip
from PIL import Image
from diffusers import StableDiffusionPipeline

print("=" * 54)
print("   AI FASHION LOOKBOOK GENERATOR")
print("=" * 54)

OUTPUT_DIR = "week8_deploy/lookbook"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOOKBOOK_DESIGNS = [
    {"name":"minimalist_white_dress",   "prompt":"minimalist white linen dress, clean lines, neutral palette, studio photography, white background"},
    {"name":"luxury_black_gown",        "prompt":"luxury black velvet evening gown, gold embroidery, floor length, editorial fashion photography"},
    {"name":"streetwear_hoodie",        "prompt":"oversized streetwear hoodie, bold graphic print, urban style, city background, editorial"},
    {"name":"floral_summer_dress",      "prompt":"floral print summer midi dress, pastel colors, feminine silhouette, natural light photography"},
    {"name":"power_blazer_suit",        "prompt":"tailored navy power suit blazer, structured shoulders, business professional, white background"},
    {"name":"bohemian_maxi_dress",      "prompt":"bohemian chiffon maxi dress, earthy tones, flowy silhouette, golden hour photography"},
    {"name":"leather_jacket_look",      "prompt":"black leather jacket outfit, edgy style, silver hardware, dark editorial photography"},
    {"name":"pastel_coord_set",         "prompt":"pastel lavender coord set, crop top and wide leg pants, minimalist style, clean background"},
    {"name":"trench_coat_classic",      "prompt":"classic beige trench coat, belted waist, timeless style, autumn street photography"},
    {"name":"silk_slip_dress",          "prompt":"silk bias-cut slip dress, champagne color, elegant drape, luxury fashion photography"},
    {"name":"oversized_denim",          "prompt":"oversized denim jacket outfit, casual streetwear, vintage wash, urban editorial"},
    {"name":"red_power_dress",          "prompt":"red fitted power dress, structured silhouette, bold statement, professional photography"},
    {"name":"athleisure_set",           "prompt":"matching athleisure set, performance fabric, sporty chic, clean studio photography"},
    {"name":"vintage_plaid_coat",       "prompt":"vintage plaid wool coat, rich colors, classic pattern, autumn fashion editorial"},
    {"name":"white_linen_suit",         "prompt":"white linen summer suit, relaxed fit, minimal style, bright studio photography"},
    {"name":"emerald_evening_dress",    "prompt":"emerald green satin evening dress, off-shoulder neckline, luxury editorial photography"},
    {"name":"cargo_street_style",       "prompt":"cargo pants street style outfit, utilitarian fashion, urban editorial, city background"},
    {"name":"wrap_dress_floral",        "prompt":"wrap dress floral print, feminine silhouette, resort wear, tropical background"},
    {"name":"monochrome_all_black",     "prompt":"all black monochrome outfit, minimalist edgy style, sleek silhouette, dark editorial"},
    {"name":"couture_ballgown",         "prompt":"couture ballgown, ivory silk taffeta, dramatic volume, high fashion studio photography"},
]

print(f"\nLoading SD pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None,
)
pipe.enable_attention_slicing()
print("Pipeline ready!\n")

print("Loading CLIP for style analysis...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
clip_model.eval()

def get_clip_score(img_path, prompt):
    img  = clip_preprocess(
        Image.open(img_path).convert("RGB")
    ).unsqueeze(0)
    tok  = clip_tokenizer([prompt])
    with torch.no_grad():
        i = clip_model.encode_image(img)
        t = clip_model.encode_text(tok)
        i = i / i.norm(dim=-1, keepdim=True)
        t = t / t.norm(dim=-1, keepdim=True)
    return round((i @ t.T).item() * 100, 2)

NEGATIVE = "blurry, low quality, bad anatomy, watermark, text, ugly, deformed"
lookbook = []
total    = len(LOOKBOOK_DESIGNS)

for idx, design in enumerate(LOOKBOOK_DESIGNS):
    print(f"[{idx+1:02d}/{total}] Generating: {design['name']}...")
    start = time.time()

    image = pipe(
        prompt              = design["prompt"] + ", high fashion photography, sharp focus, 4k",
        negative_prompt     = NEGATIVE,
        num_inference_steps = 20,
        guidance_scale      = 7.5,
        height=512, width=512,
    ).images[0]

    out_path = os.path.join(OUTPUT_DIR, f"{design['name']}.png")
    image.save(out_path)
    elapsed = time.time() - start

    score = get_clip_score(out_path, design["prompt"])

    lookbook.append({
        "rank":       idx + 1,
        "name":       design["name"],
        "prompt":     design["prompt"],
        "output":     out_path,
        "clip_score": score,
        "time_sec":   round(elapsed),
    })
    print(f"       Saved | CLIP: {score:.1f} | {elapsed:.0f}s\n")

lookbook_sorted = sorted(lookbook, key=lambda x: x["clip_score"], reverse=True)

report = {
    "total_designs": len(lookbook),
    "avg_clip_score": round(sum(d["clip_score"] for d in lookbook) / len(lookbook), 2),
    "best_design":    lookbook_sorted[0]["name"],
    "best_score":     lookbook_sorted[0]["clip_score"],
    "designs":        lookbook_sorted,
}

report_path = "week8_deploy/lookbook_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print("=" * 54)
print("   LOOKBOOK COMPLETE!")
print("=" * 54)
print(f"\n  Total designs   : {report['total_designs']}")
print(f"  Avg CLIP score  : {report['avg_clip_score']}")
print(f"  Best design     : {report['best_design']} ({report['best_score']})")
print(f"\n  Top 5 designs by CLIP score:")
for d in lookbook_sorted[:5]:
    print(f"    {d['clip_score']:5.1f}  {d['name']}")
print(f"\n  Report saved: {report_path}")
print("\nTask 3 complete!")

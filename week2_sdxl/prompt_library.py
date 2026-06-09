import json, os, random
from itertools import product

# ── Master template schema ────────────────────────────────────
TEMPLATE_SCHEMA = {
    "categories": {
        "dress":      "dress",
        "top":        "blouse top shirt",
        "outerwear":  "jacket coat blazer",
        "pants":      "trousers pants",
        "skirt":      "skirt",
        "jumpsuit":   "jumpsuit overall",
        "activewear": "sports outfit activewear set",
    },
    "silhouettes": {
        "fitted":    "fitted bodycon silhouette",
        "oversized": "oversized relaxed fit",
        "a-line":    "A-line flared silhouette",
        "straight":  "straight cut minimal silhouette",
        "wrap":      "wrap style silhouette",
    },
    "fabrics": {
        "silk":     "silk satin fabric, lustrous sheen",
        "cotton":   "soft cotton fabric, casual texture",
        "linen":    "linen fabric, breathable natural texture",
        "velvet":   "rich velvet fabric, luxurious texture",
        "denim":    "denim fabric, structured texture",
        "chiffon":  "sheer chiffon fabric, flowy drape",
        "leather":  "genuine leather fabric, structured",
    },
    "colors": {
        "black":     "all black colorway",
        "white":     "crisp white colorway",
        "navy":      "deep navy blue colorway",
        "red":       "vibrant red colorway",
        "beige":     "neutral beige colorway",
        "emerald":   "emerald green colorway",
        "blush":     "soft blush pink colorway",
        "multicolor":"multicolor print colorway",
    },
    "patterns": {
        "solid":    "solid color, no pattern",
        "floral":   "floral print pattern",
        "stripe":   "classic stripe pattern",
        "check":    "check plaid pattern",
        "abstract": "abstract artistic pattern",
        "geometric":"geometric pattern",
    },
    "occasions": {
        "casual":     "casual everyday wear",
        "formal":     "formal occasion wear",
        "evening":    "evening gala wear",
        "streetwear": "urban streetwear style",
        "business":   "business professional attire",
        "beach":      "beach resort wear",
    },
    "photo_styles": {
        "editorial": (
            "vogue editorial fashion photography, "
            "professional studio lighting, white background"
        ),
        "lookbook": (
            "fashion lookbook style, "
            "natural lighting, clean background"
        ),
        "product": (
            "fashion product photography, "
            "flat lay style, white background, "
            "commercial quality"
        ),
        "runway": (
            "fashion runway show photography, "
            "dramatic lighting, high contrast"
        ),
    },
    "quality_tags": [
        "ultra detailed",
        "sharp focus",
        "high resolution",
        "4k",
        "professional quality",
    ],
    "negative_base": (
        "blurry, low quality, bad anatomy, "
        "watermark, text, ugly, deformed, "
        "extra limbs, bad proportions, "
        "duplicate, cropped, worst quality"
    ),
}

# ── Template builder ──────────────────────────────────────────
def build_prompt(
    category   = "dress",
    silhouette = "fitted",
    fabric     = "silk",
    color      = "black",
    pattern    = "solid",
    occasion   = "formal",
    photo_style= "editorial",
    extra_tags = None,
):
    S = TEMPLATE_SCHEMA
    parts = [
        S["colors"].get(color, color),
        S["patterns"].get(pattern, pattern),
        S["silhouettes"].get(silhouette, silhouette),
        S["categories"].get(category, category),
        S["fabrics"].get(fabric, fabric),
        S["occasions"].get(occasion, occasion),
        S["photo_styles"].get(photo_style, photo_style),
        ", ".join(S["quality_tags"][:3]),
    ]
    if extra_tags:
        parts.append(", ".join(extra_tags))

    prompt = ", ".join(p for p in parts if p)
    return {
        "positive": prompt,
        "negative": S["negative_base"],
        "params": {
            "category":    category,
            "silhouette":  silhouette,
            "fabric":      fabric,
            "color":       color,
            "pattern":     pattern,
            "occasion":    occasion,
            "photo_style": photo_style,
        }
    }

# ── Pre-built named templates ─────────────────────────────────
NAMED_TEMPLATES = {
    "elegant_evening": build_prompt(
        category="dress", silhouette="fitted",
        fabric="silk", color="black",
        pattern="solid", occasion="evening",
        photo_style="editorial"
    ),
    "casual_summer": build_prompt(
        category="dress", silhouette="a-line",
        fabric="cotton", color="blush",
        pattern="floral", occasion="casual",
        photo_style="lookbook"
    ),
    "power_blazer": build_prompt(
        category="outerwear", silhouette="straight",
        fabric="linen", color="navy",
        pattern="solid", occasion="business",
        photo_style="editorial"
    ),
    "streetwear_set": build_prompt(
        category="activewear", silhouette="oversized",
        fabric="cotton", color="black",
        pattern="solid", occasion="streetwear",
        photo_style="lookbook"
    ),
    "boho_chic": build_prompt(
        category="dress", silhouette="a-line",
        fabric="chiffon", color="multicolor",
        pattern="floral", occasion="casual",
        photo_style="lookbook"
    ),
    "luxury_evening": build_prompt(
        category="dress", silhouette="fitted",
        fabric="velvet", color="emerald",
        pattern="solid", occasion="evening",
        photo_style="runway"
    ),
    "summer_beach": build_prompt(
        category="dress", silhouette="a-line",
        fabric="chiffon", color="white",
        pattern="solid", occasion="beach",
        photo_style="product"
    ),
    "classic_trench": build_prompt(
        category="outerwear", silhouette="straight",
        fabric="cotton", color="beige",
        pattern="solid", occasion="casual",
        photo_style="editorial"
    ),
}

# ── Random prompt generator ───────────────────────────────────
def generate_random_prompt():
    S = TEMPLATE_SCHEMA
    return build_prompt(
        category    = random.choice(list(S["categories"].keys())),
        silhouette  = random.choice(list(S["silhouettes"].keys())),
        fabric      = random.choice(list(S["fabrics"].keys())),
        color       = random.choice(list(S["colors"].keys())),
        pattern     = random.choice(list(S["patterns"].keys())),
        occasion    = random.choice(list(S["occasions"].keys())),
        photo_style = random.choice(list(S["photo_styles"].keys())),
    )

# ── Batch variant generator ───────────────────────────────────
def generate_color_variants(base_template_name, colors):
    base = NAMED_TEMPLATES[base_template_name]
    variants = []
    for color in colors:
        p = base["params"].copy()
        p["color"] = color
        t = build_prompt(**p)
        t["variant_name"] = f"{base_template_name}_{color}"
        variants.append(t)
    return variants

# ── Run and display ───────────────────────────────────────────
print("=" * 54)
print("   FASHION PROMPT TEMPLATE LIBRARY")
print("=" * 54)

print(f"\n{len(NAMED_TEMPLATES)} named templates built:")
for name, t in NAMED_TEMPLATES.items():
    print(f"\n  [{name}]")
    print(f"  Prompt : {t['positive'][:80]}...")

print("\n\nGenerating 3 random prompts:")
for i in range(3):
    rp = generate_random_prompt()
    print(f"\n  Random {i+1}: {rp['positive'][:80]}...")

print("\n\nColor variants for elegant_evening:")
variants = generate_color_variants(
    "elegant_evening",
    ["red", "white", "emerald", "blush"]
)
for v in variants:
    print(f"  {v['variant_name']:<35} | {v['positive'][:55]}...")

os.makedirs("week2_sdxl", exist_ok=True)
library_data = {
    "schema":           TEMPLATE_SCHEMA,
    "named_templates":  NAMED_TEMPLATES,
    "random_samples":   [generate_random_prompt() for _ in range(10)],
    "color_variants":   variants,
}
out_path = "week2_sdxl/prompt_library.json"
with open(out_path, "w") as f:
    json.dump(library_data, f, indent=2)

print(f"\n\nLibrary saved: {out_path}")
print(f"Total templates: {len(NAMED_TEMPLATES)} named + unlimited random")
print(f"\nSchema coverage:")
for key, vals in TEMPLATE_SCHEMA.items():
    if isinstance(vals, dict):
        print(f"  {key:<15}: {len(vals)} options")

print("\n" + "=" * 54)
print("   PROMPT LIBRARY COMPLETE!")
print("=" * 54)
print("\nWeek 2 Task 4 complete!")

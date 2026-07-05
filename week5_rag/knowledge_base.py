import json, os

print("=" * 54)
print("   FASHION KNOWLEDGE BASE — TASK 1")
print("=" * 54)

KNOWLEDGE_BASE = {
    "trends_2025": [
        {"trend": "quiet luxury",     "keywords": ["minimalist","neutral","understated","clean lines","beige","silk"], "season": "all"},
        {"trend": "dopamine dressing","keywords": ["bold colors","vibrant","maximalist","playful","bright"], "season": "spring"},
        {"trend": "gorpcore",         "keywords": ["outdoor","technical","utility","functional","waterproof"], "season": "autumn"},
        {"trend": "sheer layers",     "keywords": ["transparent","layered","delicate","feminine","gossamer"], "season": "summer"},
        {"trend": "power suiting",    "keywords": ["tailored","structured","blazer","professional","sharp"], "season": "all"},
        {"trend": "cottagecore",      "keywords": ["floral","prairie","romantic","vintage","linen","whimsical"], "season": "spring"},
        {"trend": "techwear",         "keywords": ["technical","futuristic","urban","functional","modular"], "season": "all"},
        {"trend": "old money",        "keywords": ["preppy","classic","polo","loafers","heritage","refined"], "season": "all"},
    ],
    "style_personas": [
        {"persona": "minimalist",  "description": "Clean lines, neutral palette, quality over quantity",  "key_pieces": ["white shirt","straight trousers","trench coat","leather loafers"]},
        {"persona": "maximalist",  "description": "Bold prints, layering, statement accessories",          "key_pieces": ["printed dress","fur coat","platform boots","statement bag"]},
        {"persona": "streetwear",  "description": "Urban, comfortable, brand-conscious, sporty",          "key_pieces": ["hoodie","cargo pants","sneakers","puffer jacket"]},
        {"persona": "bohemian",    "description": "Flowy fabrics, earthy tones, vintage-inspired",        "key_pieces": ["maxi dress","fringe bag","ankle boots","wide brim hat"]},
        {"persona": "classic",     "description": "Timeless pieces, quality fabrics, understated",        "key_pieces": ["blazer","tailored trousers","silk blouse","ballet flats"]},
        {"persona": "edgy",        "description": "Dark palette, leather, unconventional silhouettes",    "key_pieces": ["leather jacket","combat boots","band tee","ripped jeans"]},
    ],
    "color_palettes": [
        {"name": "quiet luxury",    "colors": ["ivory","camel","taupe","cream","slate gray","navy"]},
        {"name": "earth tones",     "colors": ["terracotta","rust","olive","mustard","burnt orange","chocolate"]},
        {"name": "monochromes",     "colors": ["all black","all white","all beige","all gray","all navy"]},
        {"name": "pastels",         "colors": ["blush","lavender","mint","baby blue","buttercream","peach"]},
        {"name": "bold brights",    "colors": ["cobalt blue","fuchsia","lime green","bright red","electric purple"]},
        {"name": "jewel tones",     "colors": ["emerald","sapphire","ruby","amethyst","topaz","garnet"]},
    ],
    "occasion_guides": [
        {"occasion": "business formal",   "recommended": ["tailored suit","silk blouse","heels","minimal jewelry"]},
        {"occasion": "casual friday",     "recommended": ["smart jeans","blazer","loafers","simple tee"]},
        {"occasion": "evening event",     "recommended": ["midi dress","heels","clutch","statement earrings"]},
        {"occasion": "weekend brunch",    "recommended": ["linen trousers","fitted top","sandals","tote bag"]},
        {"occasion": "date night",        "recommended": ["wrap dress","heeled boots","small bag","delicate jewelry"]},
        {"occasion": "gym to street",     "recommended": ["leggings","oversized hoodie","trainers","sports bra"]},
    ],
}

os.makedirs("week5_rag", exist_ok=True)
kb_path = "week5_rag/knowledge_base.json"
with open(kb_path, "w") as f:
    json.dump(KNOWLEDGE_BASE, f, indent=2)

print(f"\nKnowledge base built:")
for key, items in KNOWLEDGE_BASE.items():
    print(f"  {key:<20}: {len(items)} entries")

print(f"\nSaved: {kb_path}")
print("\nTask 1 complete!")

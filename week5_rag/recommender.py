import os, json, torch
import numpy as np
import open_clip
import chromadb
from PIL import Image

print("=" * 54)
print("   FASHION RECOMMENDATION ENGINE — TASK 4")
print("=" * 54)

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.eval()

client   = chromadb.PersistentClient(path="week5_rag/chromadb")
img_col  = client.get_collection("fashion_images")
text_col = client.get_collection("fashion_styles")

with open("week5_rag/knowledge_base.json") as f:
    kb = json.load(f)

def embed_text(query):
    tok = tokenizer([query])
    with torch.no_grad():
        feat = model.encode_text(tok)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze().numpy().tolist()

def embed_image(img_path):
    img = preprocess(
        Image.open(img_path).convert("RGB")
    ).unsqueeze(0)
    with torch.no_grad():
        feat = model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze().numpy().tolist()

def get_trend_advice(query):
    for trend in kb["trends_2025"]:
        if any(kw in query.lower()
               for kw in trend["keywords"]):
            return trend["trend"]
    return "classic style"

def recommend(query, mode="text", n=5):
    if mode == "text":
        emb = embed_text(query)
    else:
        emb = embed_image(query)

    results = img_col.query(
        query_embeddings=[emb], n_results=n
    )
    style_results = text_col.query(
        query_embeddings=[emb], n_results=3
    )

    trend    = get_trend_advice(query)
    sim_imgs = results["ids"][0]
    styles   = [
        m["query"] for m in
        style_results["metadatas"][0]
    ]
    distances = results["distances"][0]

    return {
        "query":           query,
        "trend_match":     trend,
        "similar_images":  sim_imgs,
        "distances":       [round(d, 4) for d in distances],
        "style_matches":   styles,
    }

TEST_QUERIES = [
    "minimalist white dress clean lines neutral",
    "bold streetwear oversized hoodie urban",
    "elegant evening gown luxury silk velvet",
    "casual summer floral bohemian dress",
    "power suit professional tailored blazer",
]

os.makedirs("week5_rag/outputs", exist_ok=True)
all_results = []

print("\nRunning recommendations...\n")
for query in TEST_QUERIES:
    rec = recommend(query, mode="text", n=5)
    all_results.append(rec)

    print(f"Query    : {query[:50]}")
    print(f"Trend    : {rec['trend_match']}")
    print(f"Top imgs : {rec['similar_images'][:3]}")
    print(f"Styles   : {rec['style_matches'][0][:50]}...")
    print()

out_path = "week5_rag/outputs/recommendations.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"{'='*54}")
print("   RECOMMENDATION ENGINE COMPLETE!")
print(f"{'='*54}")
print(f"\n  Queries processed : {len(all_results)}")
print(f"  Results saved     : {out_path}")
print(f"\nHow it works:")
print(f"  1. Text/image query → CLIP embedding")
print(f"  2. ChromaDB nearest-neighbor search")
print(f"  3. Knowledge base trend matching")
print(f"  4. Return similar images + style advice")
print(f"\nWeek 5 complete!")

import os, json, torch
import numpy as np
import open_clip
from PIL import Image
from tqdm import tqdm

print("=" * 54)
print("   CLIP EMBEDDINGS — TASK 2")
print("=" * 54)

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.eval()
print("CLIP model loaded!\n")

IMG_DIR  = "datasets/processed/fashionpedia/train/images"
OUT_PATH = "week5_rag/image_embeddings.npz"

images, embeddings, ids = [], [], []

img_files = [
    f for f in os.listdir(IMG_DIR)
    if f.endswith(".jpg")
][:200]

print(f"Embedding {len(img_files)} images...")
for fname in tqdm(img_files):
    path = os.path.join(IMG_DIR, fname)
    try:
        img = preprocess(
            Image.open(path).convert("RGB")
        ).unsqueeze(0)
        with torch.no_grad():
            feat = model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        embeddings.append(feat.squeeze().numpy())
        ids.append(fname)
    except Exception as e:
        print(f"  Skip {fname}: {e}")

embeddings_arr = np.array(embeddings)
np.savez(OUT_PATH, embeddings=embeddings_arr,
         ids=np.array(ids))

print(f"\nImage embeddings:")
print(f"  Count : {len(ids)}")
print(f"  Shape : {embeddings_arr.shape}")
print(f"  Saved : {OUT_PATH}")

with open("week5_rag/knowledge_base.json") as f:
    kb = json.load(f)

print(f"\nEmbedding style text queries...")
TEXT_QUERIES = []
for trend in kb["trends_2025"]:
    TEXT_QUERIES.append(
        f"{trend['trend']} fashion style, "
        f"{', '.join(trend['keywords'][:3])}"
    )
for persona in kb["style_personas"]:
    TEXT_QUERIES.append(
        f"{persona['persona']} fashion style, "
        f"{persona['description']}"
    )

text_embeddings, text_ids = [], []
for query in TEXT_QUERIES:
    tok = tokenizer([query])
    with torch.no_grad():
        feat = model.encode_text(tok)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    text_embeddings.append(feat.squeeze().numpy())
    text_ids.append(query[:50])

text_arr = np.array(text_embeddings)
np.savez(
    "week5_rag/text_embeddings.npz",
    embeddings=text_arr,
    ids=np.array(text_ids)
)

print(f"  Text queries embedded: {len(text_ids)}")
print(f"  Saved: week5_rag/text_embeddings.npz")
print("\nTask 2 complete!")

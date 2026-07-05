import os, json
import numpy as np
import chromadb

print("=" * 54)
print("   CHROMADB VECTOR STORE — TASK 3")
print("=" * 54)

client = chromadb.PersistentClient(path="week5_rag/chromadb")

try:
    client.delete_collection("fashion_images")
    client.delete_collection("fashion_styles")
except: pass

img_col  = client.create_collection("fashion_images")
text_col = client.create_collection("fashion_styles")

print("\nLoading image embeddings...")
img_data = np.load("week5_rag/image_embeddings.npz",
                   allow_pickle=True)
img_embs = img_data["embeddings"].tolist()
img_ids  = img_data["ids"].tolist()

BATCH = 50
for i in range(0, len(img_ids), BATCH):
    batch_ids  = img_ids[i:i+BATCH]
    batch_embs = img_embs[i:i+BATCH]
    img_col.add(
        ids        = [str(id) for id in batch_ids],
        embeddings = batch_embs,
        metadatas  = [{"source": "fashionpedia",
                       "filename": str(id)}
                      for id in batch_ids],
    )

print(f"  Stored {len(img_ids)} image vectors")

print("Loading text embeddings...")
txt_data = np.load("week5_rag/text_embeddings.npz",
                   allow_pickle=True)
txt_embs = txt_data["embeddings"].tolist()
txt_ids  = txt_data["ids"].tolist()

text_col.add(
    ids        = [f"style_{i}" for i in range(len(txt_ids))],
    embeddings = txt_embs,
    metadatas  = [{"query": str(q)} for q in txt_ids],
    documents  = [str(q) for q in txt_ids],
)
print(f"  Stored {len(txt_ids)} style text vectors")

print("\nTesting retrieval...")
import open_clip, torch
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.eval()

test_query = "elegant black evening dress fashion photography"
tok  = tokenizer([test_query])
with torch.no_grad():
    feat = model.encode_text(tok)
    feat = feat / feat.norm(dim=-1, keepdim=True)

results = img_col.query(
    query_embeddings = feat.squeeze().numpy().tolist(),
    n_results        = 5,
)
print(f"\n  Query: '{test_query}'")
print(f"  Top 5 similar images:")
for i, (rid, dist) in enumerate(zip(
    results["ids"][0],
    results["distances"][0]
)):
    print(f"    {i+1}. {rid}  (dist: {dist:.4f})")

print(f"\n{'='*54}")
print("   CHROMADB READY!")
print(f"{'='*54}")
print(f"\n  Collections: fashion_images, fashion_styles")
print(f"  DB path    : week5_rag/chromadb/")
print("\nTask 3 complete!")

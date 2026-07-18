import json, pathlib
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = pathlib.Path("data/processed/chunks.jsonl")
OUT_DIR = pathlib.Path("deploy")
OUT_DIR.mkdir(exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

with open(CHUNKS_PATH, encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

texts = [c["text"] for c in chunks]
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
embeddings = np.array(embeddings, dtype="float32")

index = faiss.IndexFlatIP(embeddings.shape[1])  # normalized vne similarity
index.add(embeddings)
faiss.write_index(index, str(OUT_DIR / "faiss.index"))

with open(OUT_DIR / "chunks_meta.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False)

print(f"Built FAISS index with {len(chunks)} chunks -> {OUT_DIR}")
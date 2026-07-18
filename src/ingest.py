import json, pathlib
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = pathlib.Path("data/processed/chunks.jsonl")
INDEX_NAME = "fssai_regs"
EMBED_DIM = 384

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}], use_ssl=False)
model = SentenceTransformer("all-MiniLM-L6-v2")

def create_index():
    if client.indices.exists(INDEX_NAME):
        client.indices.delete(INDEX_NAME)
    client.indices.create(INDEX_NAME, body={
        "settings": {"index": {"knn": True}},
        "mappings": {"properties": {
            "text": {"type": "text"},
            "doc": {"type": "keyword"},
            "page": {"type": "integer"},
            "embedding": {"type": "knn_vector", "dimension": EMBED_DIM,
                          "method": {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"}}
        }}
    })

def bulk_index(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    actions = [{"_index": INDEX_NAME, "_id": c["id"],
                "_source": {"text": c["text"], "doc": c["doc"], "page": c["page"], "embedding": emb.tolist()}}
               for c, emb in zip(chunks, embeddings)]
    helpers.bulk(client, actions)

if __name__ == "__main__":
    create_index()
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    print(f"Indexing {len(chunks)} chunks...")
    bulk_index(chunks)
    client.indices.refresh(INDEX_NAME)
    print("Done.")
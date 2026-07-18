# FSSAI Regulation RAG

A retrieval-augmented generation (RAG) system that answers questions about Indian food safety regulations (FSSAI) using hybrid search (BM25 + vector) over the source PDFs, with Gemini generating grounded, cited answers.

## Why this project

Built as a portfolio project combining a backend/search background (OpenSearch) with hands-on food-tech domain knowledge — the corpus (FSSAI additive, contaminant, and product-standard regulations) is one I can independently verify answers against, which is unusually valuable for evaluating a first RAG system honestly.

## Architecture

```
PDF (data/raw/)
   |  extract.py   -- pdfplumber, per-page text extraction
   v
pages.jsonl
   |  chunk.py     -- overlapping chunks across full document (not per-page)
   v
chunks.jsonl
   |  ingest.py    -- sentence-transformers embeddings (local, free) -> OpenSearch bulk index
   v
OpenSearch index (BM25 text + knn_vector embedding, single node, Docker)
   |  search.py    -- hybrid query: BM25 + kNN combined via normalization search pipeline
   v
top-k relevant chunks
   |  generate.py  -- Gemini (gemini-3.5-flash) answers using ONLY retrieved context
   v
grounded answer with source page citation
```

## Stack

- **OpenSearch** (Docker, single-node, security disabled for local dev) — hybrid BM25 + kNN vector search
- **sentence-transformers** (`all-MiniLM-L6-v2`) — local embeddings, no API cost or rate limits
- **Gemini API** (`gemini-3.5-flash`, free tier) — final answer generation only
- **pdfplumber** — PDF text extraction

## Setup

```powershell
docker-compose up -d

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

$env:GEMINI_API_KEY = "your-key"   # from aistudio.google.com
```

## Running the pipeline

Run once, in order, to build the index:
```powershell
python src/extract.py
python src/chunk.py
python src/ingest.py
```

Query it:
```powershell
python src/search.py      # raw hybrid retrieval, no generation
python src/generate.py    # full RAG: retrieval + Gemini-generated answer
```

Evaluate retrieval quality:
```powershell
python src/eval.py
```

## Current results

- **Corpus:** 2 FSSAI regulation PDFs (Food Products Standards & Food Additives Compendium 2022; a narrower Contaminants/pesticide-residue notification)
- **Retrieval eval:** Recall@5 = 80% (16/20) on a hand-written eval set grounded in the actual ingested text, using a strict all-keywords-present check
- **Known gap:** `Contaminants_Regulations.pdf` turned out to be a narrow 19-page amendment (pesticide residues + banned antibiotics), not the full consolidated contaminants regulation — some contaminant-limit questions (e.g. heavy metals in milk) have no answer in the current corpus

## Known limitations / future work

- Chunking is still fixed-size character sliding windows; some facts sit far enough from their product-name context (long descriptive preambles before a numbers table) that they land in separate chunks despite whole-document chunking
- No reranker yet — a cross-encoder second-stage re-score over the top ~20 hybrid results would likely fix several of the remaining retrieval misses
- Embedding model is a small, fast one (`all-MiniLM-L6-v2`); a larger local model (`all-mpnet-base-v2`) may improve recall at the cost of ingest speed
- Hybrid BM25/vector weight ratio (currently 0.4/0.6) has not been tuned against the eval set
- No demo UI or deployment yet

## License

MIT

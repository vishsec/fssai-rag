# FSSAI Regulation RAG

A retrieval-augmented generation (RAG) system that answers questions about Indian food safety regulations (FSSAI) using hybrid search (BM25 + vector) over the source PDFs, with Gemini generating grounded, cited answers.

**Live demo:** https://fssai-rag-vishsec.streamlit.app/

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

## Deployment (`deploy/`)

The live demo intentionally uses a **different retrieval backend** than local development: OpenSearch isn't practical to host free alongside a public app, so the deployed version swaps it for an embedded stack that reproduces the same hybrid search logic without a server:

- **FAISS** (`IndexFlatIP` on normalized embeddings) replaces OpenSearch's kNN vector search
- **rank_bm25** (pure Python, in-process) replaces OpenSearch's BM25 text search
- Both scores are min-max normalized and combined in Python, same weighting (0.4 BM25 / 0.6 vector) as the OpenSearch pipeline

`src/build_faiss_index.py` pre-builds the FAISS index + chunk metadata from `data/processed/chunks.jsonl`; those artifacts (`deploy/faiss.index`, `deploy/chunks_meta.json`) are committed and loaded directly by `deploy/app.py` (Streamlit), so the deployed app never needs OpenSearch, Docker, or a re-run of the ingestion pipeline. Hosted free on **Streamlit Community Cloud**.

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
- **Retrieval eval:** Recall@5 = 80% on a hand-written eval set grounded in the actual ingested text, using a strict all-keywords-present check
- **Generation eval:** ~78% of generated answers correctly stated the retrieved fact (checked manually — a naive keyword-presence check turned out to be unreliable here too, since an answer stating a fact is *absent* still contains that fact's keywords)
- **Known gap:** `Contaminants_Regulations.pdf` turned out to be a narrow 19-page amendment (pesticide residues + banned antibiotics), not the full consolidated contaminants regulation — some contaminant-limit questions (e.g. heavy metals in milk) have no answer in the current corpus

## v2 roadmap

- **Reranker** — a cross-encoder second-stage re-score over the top ~20 hybrid results, to fix cases where the right chunk is retrieved but ranked too low
- **Larger embedding model** — try `all-mpnet-base-v2` instead of `all-MiniLM-L6-v2`, trading ingest speed for recall
- **Hybrid weight tuning** — the 0.4/0.6 BM25/vector split hasn't been tuned against the eval set yet
- **Smarter chunking** — fixed-size sliding windows still occasionally separate a fact from its product-name context when the gap exceeds the chunk size; structure-aware chunking (splitting on regulation clause numbers) would be more robust than character counts
- **Generation-level evaluation** — current eval only measures retrieval recall; measuring whether Gemini's final answers are actually correct (not just whether the right chunk was retrieved) is a meaningful next layer
- **Fuller contaminants corpus** — the current `Contaminants_Regulations.pdf` is a narrow amendment notification; sourcing the full consolidated regulation would close known answer gaps (e.g. milk-specific heavy metal limits)

## License

MIT

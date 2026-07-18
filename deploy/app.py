import os, json, pathlib
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from google import genai

DEPLOY_DIR = pathlib.Path(__file__).parent
BM25_WEIGHT, VECTOR_WEIGHT, TOP_K = 0.4, 0.6, 5

@st.cache_resource
def load_index():
    index = faiss.read_index(str(DEPLOY_DIR / "faiss.index"))
    with open(DEPLOY_DIR / "chunks_meta.json", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

@st.cache_resource
def load_bm25(_chunks):
    tokenized = [c["text"].lower().split() for c in _chunks]
    return BM25Okapi(tokenized)

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def normalize(scores):
    scores = np.array(scores, dtype="float32")
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())

def hybrid_search(query, index, chunks, bm25, embedder, k=TOP_K):
    query_vec = embedder.encode([query], normalize_embeddings=True).astype("float32")
    vec_scores, vec_ids = index.search(query_vec, len(chunks))
    vec_scores, vec_ids = vec_scores[0], vec_ids[0]

    bm25_scores = bm25.get_scores(query.lower().split())

    vec_norm = normalize(vec_scores)
    bm25_norm = normalize(bm25_scores[vec_ids])  # reorder to match vec_ids

    combined = BM25_WEIGHT * bm25_norm + VECTOR_WEIGHT * vec_norm
    top_local = np.argsort(-combined)[:k]
    return [chunks[i] for i in vec_ids[top_local]]

def generate_answer(question, results):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    context = "\n\n".join(f"[Source: {r['doc']}, page {r['page']}]\n{r['text']}" for r in results)
    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}

Answer:"""
    response = client.models.generate_content(model="gemini-3.5-flash", contents=prompt)
    return response.text

st.title("FSSAI Food Regulation Assistant")
st.caption("Ask about food additive limits, contaminant threshrom FSSAI regulations.")

index, chunks = load_index()
bm25 = load_bm25(chunks)
embedder = load_embedder()

question = st.text_input("Ask a question about FSSAI food regulations:")

if question:
    with st.spinner("Searching regulations..."):
        results = hybrid_search(question, index, chunks, bm25, embedder)
    with st.spinner("Generating answer..."):
        answer = generate_answer(question, results)
    st.markdown("### Answer")
    st.write(answer)
    with st.expander("Sources used"):
        for r in results:
            st.markdown(f"**{r['doc']}, page {r['page']}**")
            st.write(r["text"][:300] + "...")
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

INDEX_NAME = "fssai_regs"
PIPELINE_NAME = "hybrid_norm_pipeline"

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}], use_ssl=False)
model = SentenceTransformer("all-MiniLM-L6-v2")

def setup_pipeline():
    client.transport.perform_request("PUT", f"/_search/pipeline/{PIPELINE_NAME}", body={
        "description": "hybrid bm25 + knn normalization",
        "phase_results_processors": [{
            "normalization-processor": {
                "normalization": {"technique": "min_max"},
                "combination": {"technique": "arithmetic_mean", "parameters": {"weights": [0.4, 0.6]}}
            }
        }]
    })

def hybrid_search(query, k=5):
    query_vector = model.encode(query).tolist()
    body = {"size": k, "query": {"hybrid": {"queries": [
        {"match": {"text": {"query": query}}},
        {"knn": {"embedding": {"vector": query_vector, "k": k}}}
    ]}}}
    res = client.search(index=INDEX_NAME, body=body, params={"search_pipeline": PIPELINE_NAME})
    return [{"score": h["_score"], "text": h["_source"]["text"],
              "doc": h["_source"]["doc"], "page": h["_source"]["page"]} for h in res["hits"]["hits"]]

# if __name__ == "__main__":
#     setup_pipeline()
#     for r in hybrid_search("maximum permissible limit of lead in milk"):
#         print(r["score"], r["doc"], r["page"], r["text"][:120])

if __name__ == "__main__":
    setup_pipeline()
    for r in hybrid_search("What is the maximum permissible limit of lead in beeswax?"):
        print("---")
        print(r["score"], r["doc"], r["page"])
        print(r["text"])
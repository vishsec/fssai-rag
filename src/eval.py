import json
from search import hybrid_search

with open("data/eval_set.json", encoding="utf-8") as f:
    eval_set = json.load(f)

if __name__ == "__main__":
    hits = 0
    for item in eval_set:
        results = hybrid_search(item["question"], k=5)
        combined = " ".join(r["text"].lower() for r in results)
        hit = all(kw.lower() in combined for kw in item["expected_keywords"])
        hits += hit
        print(f"[{'HIT' if hit else 'MISS'}] {item['question']}")
    print(f"\nRecall@5: {hits}/{len(eval_set)} = {hits/len(eval_set):.0%}")
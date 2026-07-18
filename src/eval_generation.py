import json, time
from generate import answer

with open("data/eval_set.json", encoding="utf-8") as f:
    eval_set = json.load(f)

if __name__ == "__main__":
    hits = 0
    for item in eval_set:
        ans, _ = answer(item["question"])
        hit = all(kw.lower() in ans.lower() for kw in item["expected_keywords"])
        hits += hit
        print(f"[{'HIT' if hit else 'MISS'}] {item['question']}\n  -> {ans}\n")
        time.sleep(15)  # stay under Gemini free tier's 10 RPM
    print(f"\nGeneration accuracy: {hits}/{len(eval_set)} = {hits/len(eval_set):.0%}")
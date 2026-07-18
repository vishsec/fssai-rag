import json, pathlib, re
from collections import defaultdict

IN_PATH = pathlib.Path("data/processed/pages.jsonl")
OUT_PATH = pathlib.Path("data/processed/chunks.jsonl")
CHUNK_SIZE = 1500
OVERLAP = 300

def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        chunks.append((start, text[start:start + size]))
        start += size - overlap
    return chunks

if __name__ == "__main__":
    pages_by_doc = defaultdict(list)
    with open(IN_PATH, encoding="utf-8") as f:
        for line in f:
            page = json.loads(line)
            pages_by_doc[page["doc"]].append(page)

    chunks = []
    for doc, pages in pages_by_doc.items():
        pages.sort(key=lambda p: p["page"])
        full_text, offsets = "", []
        for p in pages:
            text = re.sub(r"\s+", " ", p["text"]).strip()
            offsets.append((len(full_text), p["page"]))
            full_text += text + " "

        def page_for_offset(offset):
            page_num = offsets[0][1]
            for start, pnum in offsets:
                if start <= offset:
                    page_num = pnum
                else:
                    break
            return page_num

        for i, (start, c) in enumerate(chunk_text(full_text)):
            if c.strip():
                chunks.append({"id": f"{doc}_c{i}", "doc": doc, "page": page_for_offset(start), "text": c})

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote {len(chunks)} chunks to {OUT_PATH}")
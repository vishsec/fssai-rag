import pdfplumber, json, pathlib

RAW_DIR = pathlib.Path("data/raw")
OUT_DIR = pathlib.Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_pdf(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({"doc": pdf_path.name, "page": i + 1, "text": text})
    return pages

if __name__ == "__main__":
    all_pages = []
    for pdf_file in RAW_DIR.glob("*.pdf"):
        print(f"Extracting {pdf_file.name}...")
        all_pages.extend(extract_pdf(pdf_file))
    out_path = OUT_DIR / "pages.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for p in all_pages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_pages)} pages to {out_path}")
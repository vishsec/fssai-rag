import os
from google import genai
from search import hybrid_search
from dotenv import load_dotenv

load_dotenv()

client_gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.5-flash"

def answer(question, k=5):
    results = hybrid_search(question, k=k)
    context = "\n\n".join(f"[Source: {r['doc']}, page {r['page']}]\n{r['text']}" for r in results)
    prompt = f"""Answer the question using ONLY the context beontain the answer, say so.

Context:
{context}

Question: {question}

Answer:"""
    response = client_gemini.models.generate_content(model=MODEL, contents=prompt)
    return response.text, results

if __name__ == "__main__":
    ans, sources = answer("What is the maximum permissible limit of lead in beeswax?")
    print(ans)
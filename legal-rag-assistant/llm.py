import requests
from config import OLLAMA_MODEL, OLLAMA_URL, LLM_TIMEOUT

def build_compact_context(chunks, max_chars=3000):
    context_parts = []
    total = 0

    for c in chunks:
        text = c["text"].strip()
        if not text:
            continue

        if total + len(text) > max_chars:
            break

        context_parts.append(text)
        total += len(text)

    return "\n\n".join(context_parts)




def generate_answer(query, chunks):
    if not chunks:
        return "No directly relevant statutory provision was found for this query."

    context = build_compact_context(chunks)

    prompt = f"""
You are a legal assistant.

Answer ONLY using the statutory text below.
If the answer is not present, say: "Not mentioned in the provided statutory text."

Statutory text:
{context}

Question:
{query}

Answer:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=LLM_TIMEOUT
    )

    return response.json().get("response", "").strip()
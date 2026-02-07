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
You are a domain-specific LEGAL ASSISTANT.

STRICT RULES (DO NOT VIOLATE):
1. You MUST answer using ONLY the statutory text provided below.
2. You MUST NOT use prior knowledge, assumptions, or general legal principles.
3. If the answer is NOT explicitly stated in the text, reply EXACTLY with:
   "Not mentioned in the provided statutory text."
4. Do NOT interpret beyond the text.
5. Do NOT merge multiple sections unless explicitly connected in the text.
6. Do NOT provide legal advice or opinions.
7. Use formal legal language.
8. Quote the relevant Act name and Section number where applicable.

STATUTORY TEXT (SOURCE OF TRUTH):
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
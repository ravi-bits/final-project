import os
from groq import Groq
import streamlit as st
from config import GROQ_MODEL

# -----------------------------
# BUILD CONTEXT
# -----------------------------
def build_compact_context(chunks, max_chars=1800):
    """
    We aggressively compress context because legal sections are long.
    Prioritize first sections (most relevant FAISS matches).
    """

    context_parts = []
    total = 0

    for c in chunks:
        text = c["text"].strip()

        # remove excessive whitespace
        text = " ".join(text.split())

        # keep only first 600 chars of each section
        text = text[:600]

        if total + len(text) > max_chars:
            break

        context_parts.append(
            f"{c['act_name']} - Section {c['section_number']}:\n{text}"
        )
        total += len(text)

    return "\n\n".join(context_parts)



# -----------------------------
# GENERATE ANSWER (GROQ LLM)
# -----------------------------
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

STATUTORY TEXT:
{context}

Question:
{query}

Answer:
"""

    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a strict statutory legal assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0
    )

    return response.choices[0].message.content.strip()

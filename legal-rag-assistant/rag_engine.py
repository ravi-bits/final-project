import json
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import DATASET_PATH, EMBEDDING_MODEL, TOP_K


class LegalRAGEngine:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.chunks = []
        self.embeddings = None
        self.index = None

        self._load_dataset()
        self._build_index()

    # -------------------------------
    # DATASET LOADING (FIXED)
    # -------------------------------
    def _load_dataset(self):
        import json
        from config import DATASET_PATH

        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.chunks = []

        for act in data.get("acts", []):
            act_name = act.get("dataset_name", "Unknown Act").strip()
            # Remove dataset suffix after "-"
            base_name = act_name.split("-")[0]
            base_name = re.sub(r"\b\d{4}\b", "", base_name)
            base_name = re.sub(r"[^\w\s]", "", base_name)  # ⬅ removes commas
            base_name = re.sub(r"\s+", " ", base_name)
            act_key = base_name.lower().strip()

            year = act.get("year", "")

            for section in act.get("sections", []):
                section_number = str(section.get("section_number", "")).strip()
                section_title = section.get("section_title", "")
                section_text = section.get("section_description", "")

                if not section_text:
                    continue

                self.chunks.append({
                    "text": section_text,
                    "act_name": act_name,
                    "act_key": act_key,
                    "year": year,
                    "section_number": section_number,
                    "section_title": section_title
                })


    # -------------------------------
    # FAISS INDEX
    # -------------------------------
    def _build_index(self):
        self.index_chunks = []

        for c in self.chunks:
            if c.get("text"):
                self.index_chunks.append(c)

        if not self.index_chunks:
            raise ValueError("No valid text chunks available for indexing")

        texts = [c["text"] for c in self.index_chunks]

        self.embeddings = self.model.encode(texts, convert_to_numpy=True)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)






    # -------------------------------
    # SECTION EXISTENCE CHECK (KEY FIX)
    # -------------------------------

    def section_exists(self, query: str) -> bool:
        import re
        match = re.search(r'section\s+(\d+)', query.lower())
        if not match:
            return True

        sec = match.group(1)
        return any(str(c.get("section_number")) == sec for c in self.chunks)




    # -------------------------------
    # ACT INFERENCE FROM QUERY
    # -------------------------------
    def infer_acts_from_query(self, query: str):
        q = query.lower()
        matched = set()

        for c in self.chunks:
            if c["act_key"] and c["act_key"] in q:
                matched.add(c["act_key"])

        return matched




    # -------------------------------
    # RETRIEVAL
    # -------------------------------
    def retrieve(self, query: str):
        explicit_section = self.extract_section_from_query(query)
        explicit_act = self.extract_explicit_act(query)
        query_lower = query.lower()

        retrieved = []

        # -------------------------------
        # 1️⃣ SECTION-FIRST (HARD LAW RULE)
        # -------------------------------
        if explicit_section:
            for c in self.chunks:
                if c["section_number"] == explicit_section:
                    if explicit_act and c["act_key"] != explicit_act:
                        continue
                    retrieved.append(c)

            if retrieved:
                return {
                    "chunks": retrieved,
                    "sources": [
                        f"{c['act_name']}, {c['year']} – Section {c['section_number']}"
                        for c in retrieved
                    ],
                    "section_missing": False
                }

            return {
                "chunks": [],
                "sources": [],
                "section_missing": True
            }

        # -------------------------------
        # 2️⃣ SEMANTIC RETRIEVAL
        # -------------------------------
        query_vec = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        D, I = self.index.search(query_vec, TOP_K)

        for score, idx in zip(D[0], I[0]):
            if idx >= len(self.chunks):
                continue

            c = self.chunks[idx]

            # 🔒 HARD ACT FILTER
            if explicit_act:
                if c["act_key"] != explicit_act:
                    continue

            # 🔒 FALLBACK ACT FILTER (textual)
            elif "companies act" in query_lower:
                if "companies act" not in c["act_key"]:
                    continue

            retrieved.append(c)

        return {
            "chunks": retrieved,
            "sources": [
                f"{c['act_name']}, {c['year']} – Section {c['section_number']}"
                for c in retrieved
            ],
            "section_missing": False
        }





    def extract_explicit_act(self, query: str):
        q = re.sub(r"[^\w\s]", "", query.lower())
        q = re.sub(r"\s+", " ", q)

        for c in self.chunks:
            if c["act_key"] and c["act_key"] in q:
                return c["act_key"]

        return None


    def extract_section_from_query(self, query: str):
        match = re.search(r'section\s+(\d+)', query.lower())
        if match:
            return match.group(1)
        return None
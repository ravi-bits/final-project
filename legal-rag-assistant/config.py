import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "INDIAN_LAWS_ALL_ACTS_FINAL_CLEAN_MASTER_DATASET_v2_1.json"
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- LLM Provider ---
LLM_PROVIDER = "groq"
GROQ_MODEL = "llama-3.3-70b-versatile"
LLM_TIMEOUT = 60

TOP_K = 3
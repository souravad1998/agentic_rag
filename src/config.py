"""Central config — env vars and constants live here."""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "dev.properties"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Paths (relative to this file's parent dir)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")
CSV_FILE_PATH = os.path.join(DATA_DIR, "bhagavad_gita_verses.csv")

# ChromaDB
COLLECTION_NAME = "bhagavad_verses"

# Models — change these when swapping providers or upgrading
EMBEDDING_MODEL = "gemini-embedding-001"
RERANKER_MODEL = "gemini-2.5-flash"      # cheap & fast, used for re-ranking only
LLM_MODEL = "gemini-2.5-pro"       # latest reasoning model

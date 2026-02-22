"""
Configuration — loads environment variables and defines constants.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "dev.properties"))

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")
CSV_FILE_PATH = os.path.join(DATA_DIR, "bhagavad_gita_verses.csv")

# ChromaDB
COLLECTION_NAME = "bhagavad_verses"

# Model Names
EMBEDDING_MODEL = "gemini-embedding-001"
RERANKER_MODEL = "gemini-2.5-flash"
LLM_MODEL = "gemini-2.5-pro"

"""Wrapper around Gemini's embedding model for document + query embeddings."""

import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import GEMINI_API_KEY, EMBEDDING_MODEL


class EmbeddingManager:

    def __init__(self, api_key: str = GEMINI_API_KEY, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.api_key = api_key
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading the Embedding Model: {self.model_name}")
            self.model = GoogleGenerativeAIEmbeddings(
                model=self.model_name,
                google_api_key=self.api_key
            )
            print("Model Loaded Successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Batch-embed a list of texts. Returns shape (n, 3072)."""
        if not self.model:
            raise ValueError("Model not loaded.")
        print(f"Generating embeddings for {len(texts)} documents...")
        embeddings = np.array(self.model.embed_documents(texts))
        print(f"Done — shape: {embeddings.shape}")
        return embeddings

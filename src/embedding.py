"""
Embedding Manager — handles loading the Gemini embedding model
and generating embeddings for documents and queries.
"""

import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import GEMINI_API_KEY, EMBEDDING_MODEL


class EmbeddingManager:
    """Manages the Gemini embedding model for document and query embeddings."""

    def __init__(self, api_key: str = GEMINI_API_KEY, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.api_key = api_key
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the embedding model."""
        try:
            print(f"Loading the Embedding Model: {self.model_name}")
            self.model = GoogleGenerativeAIEmbeddings(
                model=self.model_name,
                google_api_key=self.api_key
            )
            print("Model Loaded Successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            np.ndarray with shape (len(texts), embedding_dim)
        """
        if not self.model:
            raise ValueError("Model not loaded. Please load the model first.")
        print(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.model.embed_documents(texts)
        embeddings = np.array(embeddings)
        print(f"Embeddings generated successfully! Shape: {embeddings.shape}")
        return embeddings

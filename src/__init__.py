"""Agentic RAG — Vedic AI Life Coach package."""

from src.config import GEMINI_API_KEY, VECTOR_STORE_DIR, COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL
from src.embedding import EmbeddingManager
from src.vectorstore import VectorStore
from src.data_loader import load_documents, split_documents
from src.search import RAGRetriever
from src.prompt_builder import PromptBuilder
from src.guardrails import Guardrails
from src.rag_pipeline import AdvancedRAGPipeline

__all__ = [
    "GEMINI_API_KEY", "VECTOR_STORE_DIR", "COLLECTION_NAME", "EMBEDDING_MODEL", "LLM_MODEL",
    "EmbeddingManager",
    "VectorStore",
    "load_documents", "split_documents",
    "RAGRetriever",
    "PromptBuilder",
    "Guardrails",
    "AdvancedRAGPipeline",
]

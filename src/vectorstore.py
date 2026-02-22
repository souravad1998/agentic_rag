"""
Vector Store — handles ChromaDB persistent storage for document embeddings.
"""

import hashlib
import numpy as np
import chromadb
from typing import List, Any
from src.config import VECTOR_STORE_DIR, COLLECTION_NAME


class VectorStore:
    """Manages ChromaDB persistent vector store for document embeddings."""

    def __init__(self, collection_name: str = COLLECTION_NAME, persistent_directory: str = VECTOR_STORE_DIR):
        self.collection_name = collection_name
        self.persistent_directory = persistent_directory
        self.client = None
        self.collection = None
        self._load_vector_store()

    def _load_vector_store(self):
        """Initialize ChromaDB client and collection."""
        try:
            print(f"Loading the Vector Store: {self.collection_name}")
            self.client = chromadb.PersistentClient(path=self.persistent_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Bhagavad Gita Verses embeddings for RAG"}
            )
            print(f"Vector Store Loaded Successfully! Collection: {self.collection.name}")
            print(f"Existing Collection Count: {self.collection.count()}")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            raise e

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store.

        Args:
            documents: List of LangChain Document objects
            embeddings: Numpy array of embeddings
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must be equal.")

        print(f"Adding {len(documents)} documents to the vector store...")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{hashlib.md5(doc.page_content.encode()).hexdigest()[:12]}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.upsert(
                documents=documents_text,
                embeddings=embeddings_list,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Added {len(documents)} documents successfully! Total: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents: {e}")
            raise e

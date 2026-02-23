"""ChromaDB persistent vector store for document embeddings."""

import hashlib
import numpy as np
import chromadb
from typing import List, Any
from src.config import VECTOR_STORE_DIR, COLLECTION_NAME


class VectorStore:

    def __init__(self, collection_name: str = COLLECTION_NAME, persistent_directory: str = VECTOR_STORE_DIR):
        self.collection_name = collection_name
        self.persistent_directory = persistent_directory
        self.client = None
        self.collection = None
        self._load_vector_store()

    def _load_vector_store(self):
        try:
            print(f"Loading the Vector Store: {self.collection_name}")
            self.client = chromadb.PersistentClient(path=self.persistent_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Bhagavad Gita Verses embeddings for RAG"}
            )
            print(f"Vector Store Loaded! Collection: {self.collection.name}")
            print(f"Existing docs: {self.collection.count()}")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """Upsert docs + embeddings into ChromaDB. Deduplicates via content hash."""
        if len(documents) != len(embeddings):
            raise ValueError("Mismatch: documents and embeddings must be same length.")

        print(f"Adding {len(documents)} documents...")

        ids, metadatas, texts, emb_list = [], [], [], []

        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            # deterministic ID from content hash — avoids duplicates on re-run
            doc_id = f"doc_{hashlib.md5(doc.page_content.encode()).hexdigest()[:12]}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            texts.append(doc.page_content)
            emb_list.append(emb.tolist())

        try:
            self.collection.upsert(
                documents=texts,
                embeddings=emb_list,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Done! Total docs in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents: {e}")
            raise

"""RAG retriever — semantic search + LLM-based re-ranking."""

import re
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from src.config import GEMINI_API_KEY, RERANKER_MODEL
from src.embedding import EmbeddingManager
from src.vectorstore import VectorStore


class RAGRetriever:
    """Handles query → embed → search → rerank pipeline."""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager, api_key: str = GEMINI_API_KEY):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        # Flash model for re-ranking — fast and cheap
        self.reranker_llm = ChatGoogleGenerativeAI(
            model=RERANKER_MODEL,
            google_api_key=api_key,
            temperature=0.0,
        )
        print(f"📚 RAG Retriever ready — {self.vector_store.collection.count()} docs")

    def retrieve(self, query: str, top_k: int = 5, where_filter: dict = None, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Embed query → search ChromaDB → return top_k docs above threshold."""
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        query_embedding = self.embedding_manager.model.embed_query(query)

        try:
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if where_filter:
                query_params["where"] = where_filter

            results = self.vector_store.collection.query(**query_params)

            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                for i, (doc_id, doc, meta, dist) in enumerate(zip(
                    results['ids'][0], results['documents'][0],
                    results['metadatas'][0], results['distances'][0]
                )):
                    # ChromaDB returns cosine distance, we want similarity
                    similarity = round(1 - dist, 4)
                    if similarity >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': doc,
                            'metadata': meta,
                            'similarity_score': similarity,
                            'distance': dist,
                            'rank': i + 1
                        })
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")
            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

    def llm_rerank(self, query: str, results: List[Dict[str, Any]], final_k: int = 5) -> List[Dict[str, Any]]:
        """Ask Gemini Flash to re-order results by relevance. Cheap second opinion."""
        if len(results) <= final_k:
            return results

        candidates = "\n\n".join([
            f"[{i+1}] (Verse: {r['metadata'].get('source', '?')} | {r['metadata'].get('chapter_title', '?')})\n{r['content'][:300]}"
            for i, r in enumerate(results)
        ])

        prompt = f"""You are a scripture relevance judge. Given the user's question, rank the following scripture passages from MOST to LEAST relevant.
        USER'S QUESTION: "{query}"
        PASSAGES:
        {candidates}
        Return ONLY the passage numbers in order of relevance, separated by commas.
        Example: 3, 1, 5, 2, 4
        Do not explain, just return the numbers."""

        try:
            response = self.reranker_llm.invoke([HumanMessage(content=prompt)])
            ranking_text = response.content.strip()
            ranked_indices = [int(x.strip()) - 1 for x in re.findall(r'\d+', ranking_text)]

            reranked = []
            for idx in ranked_indices:
                if 0 <= idx < len(results):
                    results[idx]['rank'] = len(reranked) + 1
                    reranked.append(results[idx])

            print(f"📊 LLM re-ranked {len(results)} → top {final_k}")
            return reranked[:final_k]

        except Exception as e:
            # graceful fallback — better to return unranked than nothing
            print(f"⚠️ Re-ranking failed ({e}), returning original order")
            return results[:final_k]

    def retrieve_and_rerank(self, query: str, initial_k: int = 10, final_k: int = 5,
                            where_filter: dict = None, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Full pipeline: cast a wide net (initial_k) → LLM rerank → keep top final_k."""
        results = self.retrieve(query, top_k=initial_k, where_filter=where_filter, score_threshold=score_threshold)
        return self.llm_rerank(query, results, final_k=final_k)

    def pretty_print(self, results: List[Dict[str, Any]]) -> None:
        """Debug helper — prints results in a readable format."""
        print("=" * 60)
        print("📖  RETRIEVED VERSES")
        print("=" * 60)
        for r in results:
            meta = r["metadata"]
            print(f"\n--- Rank {r['rank']} (similarity: {r['similarity_score']}) ---")
            print(f"  📌 {meta.get('chapter_title', '?')} | Verse: {meta.get('source', '?')}")
            print(f"  📝 {r['content'][:300]}")
        print("=" * 60)

"""
RAG Pipeline — end-to-end Advanced RAG Pipeline for Vedic Life Coaching.

Flow: User Question → Retrieve Verses → Build Prompt → LLM → Cited Answer
"""

from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from src.config import GEMINI_API_KEY, LLM_MODEL
from src.search import RAGRetriever
from src.prompt_builder import PromptBuilder


class AdvancedRAGPipeline:
    """
    End-to-end RAG pipeline for Vedic Life Coaching.

    Components:
        - RAGRetriever:    Semantic search + LLM re-ranking
        - PromptBuilder:   System prompt + context formatting
        - LLM (Gemini):    Generates scripture-grounded responses
    """

    def __init__(self, retriever: RAGRetriever, api_key: str = GEMINI_API_KEY,
                 model_name: str = LLM_MODEL, temperature: float = 0.6):
        """
        Initialize the Advanced RAG Pipeline.

        Args:
            retriever:    RAGRetriever instance
            api_key:      Gemini API key
            model_name:   LLM model for response generation
            temperature:  Creativity control (0.0=focused, 1.0=creative)
        """
        self.retriever = retriever
        self.prompt_builder = PromptBuilder()
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
        )
        self.history: List[Dict[str, Any]] = []
        print(f"🙏 Vedic Life Coach ready (LLM: {model_name})")

    def query(self, question: str, top_k: int = 5, use_reranking: bool = True,
              where_filter: dict = None, min_score: float = 0.0,
              summarize: bool = False) -> Dict[str, Any]:
        """
        Run the full RAG pipeline on a user question.

        Args:
            question:       The user's life question or situation
            top_k:          Number of verses to use in context
            use_reranking:  Whether to use LLM re-ranking
            where_filter:   Filter by metadata, e.g. {"chapter_title": "Karm Yog"}
            min_score:      Minimum similarity score threshold
            summarize:      Whether to generate a 2-sentence summary

        Returns:
            Dict with keys: question, answer, sources, citations, summary
        """
        # Step 1: Retrieve relevant verses
        print(f"\n🔍 Step 1: Retrieving verses for: '{question}'")
        if use_reranking:
            results = self.retriever.retrieve_and_rerank(
                question, initial_k=top_k * 2, final_k=top_k,
                where_filter=where_filter, score_threshold=min_score
            )
        else:
            results = self.retriever.retrieve(
                question, top_k=top_k,
                where_filter=where_filter, score_threshold=min_score
            )

        if not results:
            return {
                "question": question,
                "answer": "🙏 I couldn't find relevant verses. Please try rephrasing.",
                "sources": [], "citations": [], "summary": None,
            }

        # Step 2: Build prompt with context
        print(f"📝 Step 2: Building prompt with {len(results)} verses")
        context = self.prompt_builder.build_context(results)
        messages = self.prompt_builder.build_messages(question, context)

        # Step 3: Generate LLM response
        print("🙏 Step 3: Generating response...")
        response = self.llm.invoke(messages)
        answer = response.content

        # Handle different response formats
        if isinstance(answer, list):
            parts = []
            for part in answer:
                if isinstance(part, dict) and 'text' in part:
                    parts.append(part['text'])
                elif isinstance(part, str):
                    parts.append(part)
                else:
                    parts.append(str(part))
            answer = "\n".join(parts)
        elif isinstance(answer, dict) and 'text' in answer:
            answer = answer['text']

        # Step 4: Build citations
        citations = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            citations.append(
                f"[{i}] {meta.get('chapter_title', '?')} — Verse {meta.get('source', '?')} "
                f"(relevance: {r['similarity_score']})"
            )

        answer_with_citations = answer + "\n\n📚 Scripture References:\n" + "\n".join(citations)

        # Step 5: Optional summary
        summary = None
        if summarize:
            print("📊 Step 5: Generating summary...")
            summary_resp = self.llm.invoke([
                HumanMessage(content=f"Summarize this guidance in exactly 2 sentences:\n\n{answer}")
            ])
            summary = summary_resp.content

        # Step 6: Store in history
        result = {
            "question": question,
            "answer": answer_with_citations,
            "sources": results,
            "citations": citations,
            "summary": summary,
        }
        self.history.append(result)
        print(f"✅ Done! ({len(self.history)} queries in history)\n")

        return result

    def display(self, result: Dict[str, Any]) -> None:
        """Pretty-print a query result."""
        print("=" * 60)
        print("🙏  VEDIC LIFE COACH")
        print("=" * 60)
        print(f"\n Your Question: {result['question']}\n")
        print("─" * 60)
        print(f"\n{result['answer']}")
        if result.get("summary"):
            print(f"\n📝 Summary: {result['summary']}")
        print("\n" + "=" * 60)

    def get_history(self) -> List[Dict[str, Any]]:
        """Return full conversation history."""
        return self.history

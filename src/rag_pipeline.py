"""
End-to-end RAG pipeline for Vedic Life Coaching.

Question → Guardrails → Retrieve → Rerank → Prompt → LLM → Cited Answer
"""

from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from src.config import GEMINI_API_KEY, LLM_MODEL
from src.search import RAGRetriever
from src.prompt_builder import PromptBuilder
from src.guardrails import Guardrails


class AdvancedRAGPipeline:

    def __init__(self, retriever: RAGRetriever, api_key: str = GEMINI_API_KEY,
                 model_name: str = LLM_MODEL, temperature: float = 0.6):
        self.retriever = retriever
        self.prompt_builder = PromptBuilder()
        self.guardrails = Guardrails(api_key=api_key)
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
        )
        self.history: List[Dict[str, Any]] = []
        print(f"🙏 Vedic Life Coach ready (LLM: {model_name})")

    def _extract_text(self, content) -> str:
        """Gemini sometimes returns list[dict] instead of str. Normalize it."""
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and 'text' in part:
                    parts.append(part['text'])
                elif isinstance(part, str):
                    parts.append(part)
                else:
                    parts.append(str(part))
            return "\n".join(parts)
        elif isinstance(content, dict) and 'text' in content:
            return content['text']
        return content

    def query(self, question: str, top_k: int = 5, use_reranking: bool = True,
              where_filter: dict = None, min_score: float = 0.0,
              summarize: bool = False) -> Dict[str, Any]:
        """Run the full pipeline. Returns dict with answer, sources, citations."""

        # guardrail — reject harmful/off-topic before burning API calls
        is_allowed, guard_message = self.guardrails.validate(question)
        if not is_allowed:
            return {
                "question": question,
                "answer": guard_message,
                "sources": [], "citations": [], "summary": None,
            }

        # retrieve
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

        # build prompt + generate
        print(f"📝 Step 2: Building prompt with {len(results)} verses")
        context = self.prompt_builder.build_context(results)
        messages = self.prompt_builder.build_messages(question, context)

        print("🙏 Step 3: Generating response...")
        response = self.llm.invoke(messages)
        answer = self._extract_text(response.content)

        # citations
        citations = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            citations.append(
                f"[{i}] {meta.get('chapter_title', '?')} — Verse {meta.get('source', '?')} "
                f"(relevance: {r['similarity_score']})"
            )
        answer_with_citations = answer + "\n\n📚 Scripture References:\n" + "\n".join(citations)

        # optional summary
        summary = None
        if summarize:
            print("📊 Generating summary...")
            summary_resp = self.llm.invoke([
                HumanMessage(content=f"Summarize this guidance in exactly 2 sentences:\n\n{answer}")
            ])
            summary = self._extract_text(summary_resp.content)

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
        return self.history

"""Prompt construction for the Vedic Life Coach LLM."""

from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage


SYSTEM_PROMPT = """You are a warm, wise Vedic Life Coach — like a kind elder who speaks simply and deeply.

YOUR STYLE:
- Speak like a caring friend, not a textbook
- Short sentences. Powerful words. No filler.
- Use **bold** for key scripture quotes
- Use emojis sparingly (🙏, 💡, ✨) to add warmth

RESPONSE FORMAT (strict):
1. 🤗 **Empathy** (1-2 sentences) — acknowledge their pain, make them feel heard
2. 📖 **Wisdom** (2-3 sentences) — ONE key teaching with the exact verse citation in bold
3. 💡 **Your Takeaway** — ONE specific, actionable thing they can do TODAY

RULES:
- MAX 100 words. That's it. Respect their time.
- ONLY use wisdom from the RETRIEVED VERSES below
- NEVER invent quotes. If nothing fits, say so honestly.
- DO NOT mention verse numbers, chapter names, or scripture references in your response — citations are added separately.
- NO long paragraphs. NO bullet lists. NO lecturing.
- Write like a WhatsApp message from a wise friend, not an essay."""


class PromptBuilder:
    """Builds the system + user message pair that gets sent to the LLM."""

    @staticmethod
    def build_context(results: List[Dict[str, Any]]) -> str:
        """Turn retriever results into a formatted context block."""
        if not results:
            return "No relevant verses were found."

        verses = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            verse_ref = meta.get("source", "unknown")
            chapter = meta.get("chapter_title", "unknown")
            verses.append(
                f"[Verse {i}] {chapter} — Verse {verse_ref}\n"
                f"{r['content']}"
            )
        return "\n\n".join(verses)

    @staticmethod
    def build_messages(question: str, context: str) -> List:
        """Assemble the final [SystemMessage, HumanMessage] pair for the LLM."""
        user_prompt = (
            f"RETRIEVED SCRIPTURE VERSES:\n"
            f"{'━' * 50}\n"
            f"{context}\n"
            f"{'━' * 50}\n\n"
            f"USER'S LIFE SITUATION:\n"
            f"{question}\n\n"
            f"Provide compassionate guidance grounded in the above verses."
        )
        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

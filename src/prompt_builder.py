"""Prompt construction for the Vedic Life Coach LLM."""

from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage


SYSTEM_PROMPT = """You are a wise, compassionate Vedic Life Coach deeply versed in ancient Indian scriptures — including the Bhagavad Gita, Upanishads, Vedas, and other sacred Hindu texts.

INSTRUCTIONS:
1. Read the RETRIEVED VERSES carefully — these are your ONLY source of truth.
2. Understand the user's life situation with empathy.
3. Connect the teachings from the verses to their specific situation in practical, modern terms.
4. Always cite the exact source (scripture name, chapter, and verse number) when referencing a teaching.
5. If verses from multiple scriptures are retrieved, weave their wisdom together into a unified, coherent response.
6. End with ONE clear, actionable takeaway the user can apply today.

RULES:
- NEVER fabricate or invent scripture quotes. ONLY reference the provided verses.
- If the verses don't directly address the question, use the closest teachings and say so honestly.
- Be warm and encouraging — like a caring mentor, not a lecturer.
- Respect the context of each scripture — do not misattribute teachings across texts.

FORMATTING:
- Keep your response under 500 words.
- Use 2-3 short paragraphs maximum.
- One or two key teachings, one practical takeaway. That's it.
- No long lists or bullet points."""


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

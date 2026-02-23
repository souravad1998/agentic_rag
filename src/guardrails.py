"""
Guardrails — two-layer input filter.

Layer 1: Regex keyword check (instant, zero cost) — catches harmful content
Layer 2: LLM topic check (Gemini Flash) — catches off-topic questions
"""

import re
from typing import Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from src.config import GEMINI_API_KEY, RERANKER_MODEL


# ─── Harmful content patterns ───────────────────────────────────

HARMFUL_PATTERNS = [
    # self-harm
    r"\b(suicide|suicidal|kill\s*(myself|yourself|herself|himself|themselves))\b",
    r"\b(murder|homicide|assassinate|slaughter)\b",
    r"\b(poison|poisoning|overdose|hang\s*(myself|yourself))\b",
    r"\b(die|dying|death|dead)\b.*\b(want|wish|how\s*to|make\s*(someone|him|her|them))\b",
    r"\b(want|wish|going)\s*(to)?\s*(die|kill|end\s*(it|my\s*life|everything))\b",
    r"\b(self[\s-]?harm|cut\s*(myself|yourself)|slit|bleed)\b",
    r"\b(end\s*(my|your|his|her)\s*life)\b",
    # violence
    r"\b(bomb|explosive|weapon|gun|shoot|stab|attack|beat|thrash)\b",
    # illegal
    r"\b(how\s*to\s*(hack|steal|rob|kidnap|traffick))\b",
    # abuse
    r"\b(abuse|molest|rape|assault|sex|fuck)\b",
]

HARMFUL_REGEX = re.compile("|".join(HARMFUL_PATTERNS), re.IGNORECASE)

# fast exact-match set — checked before regex for speed
BLOCKED_WORDS = {
    "suicide", "murder", "kill", "poison", "homicide",
    "assassinate", "bomb", "explosive", "weapon", "rape",
    "molest", "hack", "kidnap", "traffick", "terrorist", "sex", "fuck",
}


# ─── Canned responses ───────────────────────────────────────────

SAFETY_RESPONSE = (
    "🙏 I sense you may be going through something very difficult. "
    "I'm a spiritual guidance tool and not equipped to help with this type of concern.\n\n"
    "If you or someone you know is in crisis, please reach out:\n"
    "📞 **iCall**: 9152987821\n"
    "📞 **Vandrevala Foundation**: 1860-2662-345 (24/7)\n"
    "📞 **AASRA**: 9820466726\n\n"
    "You are not alone. Please talk to someone who can help. 🙏 \n"
    "Thank You."
)

IRRELEVANT_RESPONSE = (
    "🙏 This question doesn't seem related to life guidance or spiritual wisdom. "
    "I'm here to help with life challenges using teachings from the Gita and Vedic scriptures.\n\n"
    "Try asking about: relationships, purpose, emotions, duty, personal growth, "
    "grief, motivation, anger, peace, or any life situation you're facing."
)


# ─── Relevance filter prompt (few-shot, strict) ─────────────────

RELEVANCE_PROMPT = (
    'You are an EXTREMELY strict topic filter for a Vedic spiritual guidance chatbot. '
    'Your job is to REJECT anything that is NOT a genuine life/emotional/spiritual question.\n\n'

    'ACCEPT ONLY if the question is genuinely about:\n'
    '- Emotional pain: grief, sadness, loneliness, heartbreak, depression, anxiety, fear\n'
    '- Life dilemmas: career confusion, relationship problems, family conflicts, life purpose\n'
    '- Personal growth: self-improvement, discipline, motivation, overcoming laziness\n'
    '- Spirituality: dharma, karma, meditation, detachment, inner peace, soul, moksha\n'
    '- Philosophy: meaning of life, death, suffering, duty, right vs wrong\n'
    '- Vedic concepts: yoga, scriptures, Gita teachings, Vedantic philosophy\n\n'

    'REJECT everything else, including but not limited to:\n'
    '- How-to/technical: driving, cooking, coding, building, fixing, operating\n'
    '- Products/brands: cars, bikes, phones, gadgets, Mercedes, Royal Enfield\n'
    '- Academic: math, science, history facts, geography, exams, homework\n'
    '- Entertainment: movies, games, music, sports, celebrities, TV shows\n'
    '- Food/recipes: cooking, baking, restaurants, ingredients\n'
    '- Technology: programming, AI, software, apps, websites\n'
    '- Politics/news: elections, government, current events\n'
    '- Shopping/money: buying, selling, prices, investments, stocks, crypto\n'
    '- Travel/places: hotels, flights, tourism, destinations\n'
    '- Random/trivia: jokes, riddles, fun facts, weather, time\n'
    '- Health/medical: symptoms, medicines, diseases, doctors (suggest a doctor instead)\n\n'

    'EXAMPLES:\n'
    '"I feel lost in life" → yes\n'
    '"How to ride a Royal Enfield?" → no\n'
    '"I lost my job and feel hopeless" → yes\n'
    '"How to make pasta?" → no\n'
    '"What is the meaning of karma?" → yes\n'
    '"Best smartphone under 20000?" → no\n'
    '"My parents don\'t understand me" → yes\n'
    '"How to drive a Mercedes?" → no\n'
    '"I feel angry all the time" → yes\n'
    '"Who won the cricket match?" → no\n'
    '"How to find inner peace?" → yes\n'
    '"Capital of France?" → no\n\n'

    'DEFAULT TO "no" if you are even slightly unsure.\n'
    'Answer ONLY "yes" or "no". Nothing else.\n\n'
)


class Guardrails:
    """Two-layer input filter: keyword check → LLM relevance check."""

    def __init__(self, api_key: str = GEMINI_API_KEY, enable_llm_check: bool = True):
        self.enable_llm_check = enable_llm_check
        if enable_llm_check:
            self.llm = ChatGoogleGenerativeAI(
                model=RERANKER_MODEL,
                google_api_key=api_key,
                temperature=0.0,
            )
        print("🛡️  Guardrails active")

    def check_keywords(self, question: str) -> Tuple[bool, str]:
        """Layer 1: instant regex/keyword scan. Returns (is_safe, message)."""
        q = question.lower().strip()

        # exact word match first (O(1) set lookup)
        words = set(re.findall(r'\b\w+\b', q))
        if words & BLOCKED_WORDS:
            print("🛡️  BLOCKED: harmful keyword detected")
            return False, SAFETY_RESPONSE

        # then regex for multi-word patterns
        if HARMFUL_REGEX.search(q):
            print("🛡️  BLOCKED: harmful pattern detected")
            return False, SAFETY_RESPONSE

        return True, ""

    def check_relevance(self, question: str) -> Tuple[bool, str]:
        """Layer 2: LLM-based topic check. Returns (is_relevant, message)."""
        if not self.enable_llm_check:
            return True, ""

        try:
            response = self.llm.invoke([
                HumanMessage(content=RELEVANCE_PROMPT + f'Question: "{question}"')
            ])

            answer = response.content
            if isinstance(answer, list):
                answer = str(answer[0]) if answer else "no"
            if isinstance(answer, dict):
                answer = answer.get('text', 'no')

            if "yes" not in answer.lower():
                print("🛡️  BLOCKED: off-topic question")
                return False, IRRELEVANT_RESPONSE

            return True, ""

        except Exception as e:
            # fail open — better to answer than crash
            print(f"⚠️ Relevance check failed ({e}), allowing through")
            return True, ""

    def validate(self, question: str) -> Tuple[bool, str]:
        """Run both layers. Returns (is_allowed, message)."""
        is_safe, msg = self.check_keywords(question)
        if not is_safe:
            return False, msg

        is_relevant, msg = self.check_relevance(question)
        if not is_relevant:
            return False, msg

        return True, ""

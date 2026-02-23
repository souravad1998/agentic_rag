"""Streamlit UI for the Vedic AI Life Coach."""

import streamlit as st
from src import EmbeddingManager, VectorStore, RAGRetriever, AdvancedRAGPipeline


# ─── Page config ─────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Based Vedic Life Coach",
    page_icon="🙏",
    layout="centered",
)


# ─── Custom CSS ──────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }

    /* header */
    .hero {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #f5af19, #f12711);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #a0a0b8;
        font-size: 1rem;
        font-weight: 300;
    }

    /* chat bubbles */
    .user-msg {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
    }
    .bot-msg {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #ffffff;
        padding: 1.2rem 1.4rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        max-width: 90%;
        font-size: 0.95rem;
        line-height: 1.6;
        backdrop-filter: blur(10px);
    }

    /* source cards */
    .source-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin: 0.3rem 0;
        font-size: 0.82rem;
        color: #e8e8f0;
    }
    .source-card strong { color: #f5af19; }

    /* guardrail warning */
    .guard-msg {
        background: rgba(255, 87, 87, 0.12);
        border: 1px solid rgba(255, 87, 87, 0.3);
        color: #ffffff;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        font-size: 0.92rem;
    }

    /* sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    /* sidebar buttons — transparent with white border */
    section[data-testid="stSidebar"] button {
        background: transparent !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        color: white !important;
    }
    section[data-testid="stSidebar"] button:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(255, 255, 255, 0.5) !important;
    }
    section[data-testid="stSidebar"] button p {
        color: white !important;
    }
    #MainMenu, footer, header { visibility: hidden; }

    div[data-testid="stChatInput"] textarea {
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Init pipeline (cached so it loads once) ─────────────────

@st.cache_resource
def init_pipeline():
    em = EmbeddingManager()
    vs = VectorStore()
    retriever = RAGRetriever(vs, em)
    return AdvancedRAGPipeline(retriever)


coach = init_pipeline()


# ─── Sidebar ─────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Verses to retrieve", 3, 10, 5)
    use_reranking = st.toggle("LLM Re-ranking", value=True)
    summarize = st.toggle("Show summary", value=False)

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown(
        "This AI Life Coach draws wisdom from **14K+ Hindu scripture verses** "
        "(Bhagavad Gita, Vedas, Upanishads) to guide you through life's challenges."
    )
    st.markdown(
        "Built with LangChain, Gemini, ChromaDB, "
        "and dual-layer guardrails."
    )

    st.markdown("---")
    st.markdown("### 💡 Try asking")
    examples = [
        "I feel lost in life, what should I do?",
        "How to overcome fear and self-doubt?",
        "I lost my job and feel hopeless.",
        "What is the meaning of karma?",
        "How do I control my anger?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["prefill"] = ex

    # history viewer
    st.markdown("---")
    st.markdown("### 🕐 Query History")
    history = coach.get_history()
    if history:
        st.caption(f"{len(history)} queries so far")
        for i, h in enumerate(reversed(history), 1):
            q = h["question"][:50] + "..." if len(h["question"]) > 50 else h["question"]
            with st.expander(f"{i}. {q}"):
                st.markdown(f"**Q:** {h['question']}")
                st.markdown(f"**A:** {h['answer'][:300]}...")
                if h.get("citations"):
                    st.markdown("**Sources:**")
                    for c in h["citations"]:
                        st.caption(c)
        if st.button("🗑️ Clear History", use_container_width=True):
            coach.history.clear()
            st.session_state.messages = []
            st.rerun()
    else:
        st.caption("No queries yet — ask something!")


# ─── Header ──────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <h1>🙏 RAG Based Vedic Life Coach</h1>
    <p>Wisdom from ancient scriptures, applied to modern life</p>
</div>
""", unsafe_allow_html=True)


# ─── Chat history ────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# render past messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    elif msg.get("blocked"):
        st.markdown(f'<div class="guard-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        # show sources in expander
        if msg.get("sources"):
            with st.expander("📚 View scripture sources"):
                for s in msg["sources"]:
                    meta = s["metadata"]
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>{meta.get("chapter_title", "?")}</strong> — '
                        f'Verse {meta.get("source", "?")} '
                        f'(similarity: {s["similarity_score"]})<br>'
                        f'{s["content"][:200]}...</div>',
                        unsafe_allow_html=True
                    )


# ─── Chat input ──────────────────────────────────────────────

# handle example button clicks
prefill = st.session_state.pop("prefill", None)

# disable input while a query is running
is_processing = st.session_state.get("processing", False)

if prompt := (prefill or st.chat_input(
    "⏳ Processing... please wait" if is_processing else "Ask me anything about life...",
    disabled=is_processing
)):
    # add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="user-msg">{prompt}</div>', unsafe_allow_html=True)

    # lock input while processing
    st.session_state["processing"] = True

    # generate response
    with st.spinner("🙏 Seeking wisdom from the scriptures..."):
        result = coach.query(
            prompt,
            top_k=top_k,
            use_reranking=use_reranking,
            summarize=summarize,
        )

    # unlock input
    st.session_state["processing"] = False

    # check if guardrail blocked it
    is_blocked = len(result["sources"]) == 0 and (
        "not equipped" in result["answer"] or
        "doesn't seem related" in result["answer"]
    )

    if is_blocked:
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "blocked": True,
        })
        st.markdown(f'<div class="guard-msg">{result["answer"]}</div>', unsafe_allow_html=True)
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "summary": result.get("summary"),
        })
        st.markdown(f'<div class="bot-msg">{result["answer"]}</div>', unsafe_allow_html=True)

        # sources expander
        if result["sources"]:
            with st.expander("📚 View scripture sources"):
                for s in result["sources"]:
                    meta = s["metadata"]
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>{meta.get("chapter_title", "?")}</strong> — '
                        f'Verse {meta.get("source", "?")} '
                        f'(similarity: {s["similarity_score"]})<br>'
                        f'{s["content"][:200]}...</div>',
                        unsafe_allow_html=True
                    )

        if result.get("summary"):
            st.info(f"📝 **Summary:** {result['summary']}")

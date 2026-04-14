"""
app.py — PaperMind Streamlit UI
Chat with research papers using RAG + LLaMA 3
"""

import streamlit as st
from groq import Groq

from rag import (
    load_model,
    extract_text_and_metadata,
    chunk_text,
    embed_chunks,
    retrieve_top_chunks,
    generate_answer,
    generate_summary,
)

# ──────────────────────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PaperMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Hide Streamlit default header */
    #MainMenu, footer, header { visibility: hidden; }

    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #2a2f3e;
    }

    /* Cards */
    .stat-card {
        background: #1e2433;
        border: 1px solid #2a2f3e;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #7c83fd;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #8b95a8;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #1e2433 !important;
        border: 1px solid #2a2f3e !important;
        border-radius: 12px !important;
        margin-bottom: 8px !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7c83fd, #a78bfa);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        transform: translateY(-1px);
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #1e2433;
        border: 2px dashed #2a2f3e;
        border-radius: 12px;
        padding: 10px;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background: #1e2433;
        border: 1px solid #2a2f3e;
        border-radius: 10px;
    }

    /* Success / info boxes */
    .stSuccess, .stInfo, .stWarning {
        border-radius: 10px;
    }

    /* Title styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #7c83fd, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .main-subtitle {
        color: #8b95a8;
        font-size: 1rem;
        margin-top: 4px;
    }

    /* Chunk card */
    .chunk-card {
        background: #161b27;
        border-left: 3px solid #7c83fd;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.85rem;
        color: #b0bac8;
        line-height: 1.6;
    }
    .chunk-score {
        font-size: 0.75rem;
        color: #7c83fd;
        font-weight: 600;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Cached Resources
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🧠 PaperMind")
    st.markdown("<small style='color:#8b95a8'>RAG-powered paper assistant</small>", unsafe_allow_html=True)
    st.divider()

    groq_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com",
    )

    if groq_key:
        st.success("API key loaded ✓")

    st.divider()

    st.markdown("**⚙️ Retrieval Settings**")
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=6, value=3,
                      help="More chunks = more context, but slower")
    chunk_size = st.slider("Chunk size (words)", min_value=200, max_value=600, value=400, step=50)

    st.divider()

    st.markdown("""
**How it works**
1. 📄 Upload your PDF
2. ✂️ Text split into chunks
3. 🔢 Chunks embedded (MiniLM)
4. 🔍 Query retrieves top chunks
5. 🤖 LLaMA 3 generates answer
""")

    st.divider()
    st.markdown("<small style='color:#8b95a8'>Built with sentence-transformers · Groq · Streamlit</small>",
                unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">🧠 PaperMind</p>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Upload a research paper and chat with it using AI</p>',
            unsafe_allow_html=True)
st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Guard
# ──────────────────────────────────────────────────────────────────────────────

if not groq_key:
    st.info("👈 Enter your **free Groq API key** in the sidebar to get started.\n\n"
            "Get one at [console.groq.com](https://console.groq.com) — takes 1 minute.")
    st.stop()

model = get_model()

try:
    client = Groq(api_key=groq_key)
except Exception:
    st.error("Invalid Groq API key. Please check and try again.")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# File Upload
# ──────────────────────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "📄 Upload a research paper (PDF)",
    type="pdf",
    help="Upload any research paper or document in PDF format",
)

if uploaded:
    # Re-process only if new file
    if "last_file" not in st.session_state or st.session_state.last_file != uploaded.name:
        with st.spinner("⚙️ Processing your PDF — extracting, chunking & embedding..."):
            try:
                meta = extract_text_and_metadata(uploaded)
                chunks = chunk_text(meta["full_text"], chunk_size=chunk_size)
                embeddings = embed_chunks(model, chunks)

                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.session_state.meta = meta
                st.session_state.last_file = uploaded.name
                st.session_state.messages = []
                st.session_state.summary = None

            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                st.stop()

    meta = st.session_state.meta

    # ── Stats Row ─────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{meta['page_count']}</div>
            <div class="stat-label">Pages</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{meta['word_count']:,}</div>
            <div class="stat-label">Words</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(st.session_state.chunks)}</div>
            <div class="stat-label">Chunks</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">✓</div>
            <div class="stat-label">Ready</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["💬 Chat", "📋 Auto Summary"])

    # ── Tab 1: Chat ───────────────────────────────────────────────────────────
    with tab1:
        # Suggested questions
        st.markdown("**💡 Try asking:**")
        q_col1, q_col2, q_col3 = st.columns(3)
        suggestions = [
            "What is the main contribution?",
            "What methods were used?",
            "What are the key findings?",
        ]
        for col, suggestion in zip([q_col1, q_col2, q_col3], suggestions):
            with col:
                if st.button(suggestion, key=suggestion):
                    st.session_state.prefill = suggestion

        st.divider()

        # Chat history
        for msg in st.session_state.get("messages", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Handle prefill from suggestion buttons
        prefill = st.session_state.pop("prefill", "")

        query = st.chat_input("Ask anything about the paper...", key="chat_input")
        query = query or prefill

        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Searching paper and generating answer..."):
                    try:
                        top_chunks, scores = retrieve_top_chunks(
                            query,
                            st.session_state.chunks,
                            st.session_state.embeddings,
                            model,
                            top_k=top_k,
                        )
                        answer = generate_answer(client, query, top_chunks)
                    except Exception as e:
                        answer = f"⚠️ Error generating answer: {e}"
                        top_chunks, scores = [], []

                st.markdown(answer)

                if top_chunks:
                    with st.expander("📚 View source chunks used"):
                        for i, (chunk, score) in enumerate(zip(top_chunks, scores)):
                            st.markdown(
                                f'<div class="chunk-card">'
                                f'<div class="chunk-score">📍 Chunk {i+1} — Similarity: {score:.3f}</div>'
                                f'{chunk[:450]}{"..." if len(chunk) > 450 else ""}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

            st.session_state.messages.append({"role": "assistant", "content": answer})

        # Clear chat button
        if st.session_state.get("messages"):
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Clear Chat", key="clear"):
                st.session_state.messages = []
                st.rerun()

    # ── Tab 2: Summary ────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### 📋 Auto-Generated Paper Summary")
        st.caption("PaperMind reads the paper and extracts the key information for you.")

        if st.session_state.get("summary"):
            st.markdown(st.session_state.summary)
        else:
            if st.button("✨ Generate Summary", key="summarize"):
                with st.spinner("Reading and summarizing the paper..."):
                    try:
                        summary = generate_summary(client, st.session_state.meta["full_text"])
                        st.session_state.summary = summary
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")

else:
    # ── Empty state ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #8b95a8;">
        <div style="font-size: 4rem;">📄</div>
        <h3 style="color: #c0c8d8; margin-top: 16px;">Upload a research paper to get started</h3>
        <p>Supports any PDF — research papers, textbooks, reports</p>
    </div>
    """, unsafe_allow_html=True)
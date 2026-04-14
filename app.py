import streamlit as st
from groq import Groq
from rag import load_model, extract_chunks, get_embeddings, retrieve, ask_llm

# ── Page Config ───────────────────────────────
st.set_page_config(page_title="PaperMind", page_icon="🧠", layout="wide")
st.title("🧠 PaperMind")
st.caption("Upload any research paper PDF and ask questions — powered by RAG + LLaMA 3")

# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Free key at console.groq.com",
    )
    st.markdown("---")
    st.markdown("""
**How it works**
1. 📄 Upload your PDF
2. ✂️ Text is split into chunks
3. 🔢 Chunks are embedded (MiniLM)
4. 🔍 Your question retrieves top chunks
5. 🤖 LLaMA 3 generates an answer
""")
    st.markdown("---")
    st.markdown("Built with `sentence-transformers` · `Groq` · `Streamlit`")

# ── Guard ─────────────────────────────────────
if not groq_key:
    st.info("👈 Enter your free Groq API key in the sidebar to get started.")
    st.stop()

# ── Load model & client ───────────────────────
@st.cache_resource
def get_model():
    return load_model()

model = get_model()
client = Groq(api_key=groq_key)

# ── File Upload ───────────────────────────────
uploaded = st.file_uploader("Upload a research paper (PDF)", type="pdf")

if uploaded:
    if "last_file" not in st.session_state or st.session_state.last_file != uploaded.name:
        with st.spinner("🔄 Processing PDF..."):
            chunks = extract_chunks(uploaded)
            embeddings = get_embeddings(model, chunks)
            st.session_state.chunks = chunks
            st.session_state.embeddings = embeddings
            st.session_state.last_file = uploaded.name
            st.session_state.messages = []
        st.success(f"✅ Ready! Indexed **{len(chunks)} chunks** from `{uploaded.name}`")

    st.markdown("---")

    # ── Chat Interface ────────────────────────
    st.subheader("💬 Ask anything about the paper")

    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("e.g. What is the main contribution of this paper?")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                top_chunks, scores = retrieve(
                    query,
                    st.session_state.chunks,
                    st.session_state.embeddings,
                    model,
                )
                answer = ask_llm(client, query, top_chunks)

            st.write(answer)

            with st.expander("📚 View retrieved source chunks"):
                for i, (chunk, score) in enumerate(zip(top_chunks, scores)):
                    st.markdown(f"**Chunk {i+1}** — similarity: `{score:.3f}`")
                    st.caption(chunk[:400] + ("..." if len(chunk) > 400 else ""))
                    st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})
"""
rag.py — Core RAG pipeline for PaperMind
Handles PDF parsing, chunking, embedding, retrieval, and LLM generation.
"""

import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def load_model() -> SentenceTransformer:
    """Load the sentence transformer model for embeddings."""
    return SentenceTransformer("all-MiniLM-L6-v2")


# ──────────────────────────────────────────────────────────────────────────────
# PDF Processing
# ──────────────────────────────────────────────────────────────────────────────

def extract_text_and_metadata(pdf_file) -> dict:
    """
    Extract full text and metadata from a PDF file.
    Returns a dict with text, page_count, and word_count.
    """
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    pages_text = []
    for page in doc:
        pages_text.append(page.get_text())

    full_text = " ".join(pages_text)
    word_count = len(full_text.split())

    return {
        "full_text": full_text,
        "page_count": len(doc),
        "word_count": word_count,
    }


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    Overlap ensures context is not lost at chunk boundaries.
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Embeddings
# ──────────────────────────────────────────────────────────────────────────────

def embed_chunks(model: SentenceTransformer, chunks: list[str]) -> np.ndarray:
    """Generate embeddings for a list of text chunks."""
    return model.encode(chunks, show_progress_bar=False)


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    """Generate embedding for a single query string."""
    return model.encode([query])[0]


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def retrieve_top_chunks(
    query: str,
    chunks: list[str],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 3,
) -> tuple[list[str], list[float]]:
    """
    Retrieve the top-k most relevant chunks for a given query
    using cosine similarity between query and chunk embeddings.
    """
    q_emb = embed_query(model, query)
    scores = [cosine_similarity(q_emb, emb) for emb in embeddings]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices], [scores[i] for i in top_indices]


# ──────────────────────────────────────────────────────────────────────────────
# LLM Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_answer(client, query: str, context_chunks: list[str]) -> str:
    """
    Generate a grounded answer using LLaMA 3 via Groq API.
    The LLM is strictly instructed to answer only from the provided context.
    """
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are PaperMind, an expert AI research assistant.
Your job is to answer questions about research papers accurately and clearly.

RULES:
- Answer ONLY using the context provided below.
- If the answer is not in the context, say: "I couldn't find this in the document."
- Be concise but thorough.
- Use bullet points when listing multiple items.
- Always sound confident and professional.

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=700,
    )
    return response.choices[0].message.content


def generate_summary(client, text: str) -> str:
    """
    Generate a concise summary of the research paper.
    Uses the first ~2000 words to stay within token limits.
    """
    excerpt = " ".join(text.split()[:2000])

    prompt = f"""You are PaperMind, an AI research assistant.
Summarize the following research paper excerpt in a structured way.

Provide:
1. **Title / Topic** (inferred if not explicit)
2. **Main Objective** (1-2 sentences)
3. **Key Contributions** (3-4 bullet points)
4. **Methods Used** (brief)
5. **Main Findings** (2-3 bullet points)

Be concise and use simple language.

Paper Excerpt:
{excerpt}

Summary:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=600,
    )
    return response.choices[0].message.content
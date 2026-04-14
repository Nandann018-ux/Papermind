import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer


def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_chunks(pdf_file, chunk_size=400, overlap=50):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    full_text = " ".join(page.get_text() for page in doc)
    words = full_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def get_embeddings(model, chunks):
    return model.encode(chunks, show_progress_bar=False)


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def retrieve(query, chunks, embeddings, model, top_k=3):
    q_emb = model.encode([query])[0]
    scores = [cosine_sim(q_emb, e) for e in embeddings]
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_idx], [scores[i] for i in top_idx]


def ask_llm(client, query, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a helpful research assistant.
Answer the user's question using ONLY the context below.
If the answer isn't in the context, say "I couldn't find this in the document."

Context:
{context}

Question: {query}
Answer:"""
    resp = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=600,
    )
    return resp.choices[0].message.content
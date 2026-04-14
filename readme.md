# PaperMind — Chat with Research Papers

> Upload any research paper PDF and get instant, accurate answers — powered by RAG + LLaMA 3.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red?style=flat&logo=streamlit)
![LLaMA](https://img.shields.io/badge/LLM-LLaMA%203-purple?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## Live Demo
[**Try PaperMind here**](https://papermindgit.streamlit.app/)

---

## What is PaperMind?

PaperMind is a **Retrieval-Augmented Generation (RAG)** app that lets you:

- Upload any research paper PDF
- Ask natural language questions and get grounded answers
- Auto-generate a structured paper summary
- See exactly which parts of the paper the answer came from

---

## Tech Stack

| Layer | Tool |
|---|---|
| **PDF Parsing** | PyMuPDF |
| **Embeddings** | `sentence-transformers` — `all-MiniLM-L6-v2` |
| **Vector Search** | NumPy cosine similarity |
| **LLM** | LLaMA 3 (8B) via Groq API (free) |
| **UI** | Streamlit |

---

## RAG Architecture

```
PDF Upload
    │
    ▼
Text Extraction  (PyMuPDF)
    │
    ▼
Chunking  (400 words, 60 word overlap)
    │
    ▼
Embedding  (all-MiniLM-L6-v2)
    │
    ▼
Cosine Similarity Search  ◄── User Query (embedded)
    │
    ▼
Top-K Relevant Chunks
    │
    ▼
LLaMA 3 via Groq  →  Grounded Answer
```

---

## Features

- PDF upload with instant chunking & indexing
- Semantic search using sentence embeddings
- LLaMA 3 answers grounded strictly in the document
- Auto paper summary (objective, contributions, findings)
- Source chunk viewer for every answer
- Suggested question buttons for quick start
- Adjustable retrieval settings (chunk size, top-k)
- Conversation memory within session
- Clean dark UI

---

## Project Structure

```
papermind/
├── app.py              # Streamlit UI
├── rag.py              # RAG pipeline (chunking, embedding, retrieval, LLM)
├── requirements.txt    # Dependencies
├── .gitignore
└── README.md
```

## API Key

Get a **free** Groq API key at [console.groq.com](https://console.groq.com) — takes under 2 minutes. Enter it in the app sidebar.

---

## License

[MIT](LICENSE)

---

> Star this repo if you found it useful!

# PaperMind — Chat with Research Papers

> Upload any research paper PDF and get instant, accurate answers — powered by RAG + LLaMA 3.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red?style=flat&logo=streamlit)
![LLaMA](https://img.shields.io/badge/LLM-LLaMA%203-purple?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## Live Demo
[**Try PaperMind here**](https://your-app-link.streamlit.app) ← *(update after deploying)*

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

---

## Run Locally

```bash
# Clone
git clone https://github.com/your-username/papermind.git
cd papermind

# Install
pip3 install -r requirements.txt

# Run
python3 -m streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set main file as `app.py`
4. Click **Deploy**

---

## API Key

Get a **free** Groq API key at [console.groq.com](https://console.groq.com) — takes under 2 minutes. Enter it in the app sidebar.

---

## Author

**Your Name**
- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [linkedin.com/in/your-profile](https://linkedin.com/in/your-profile)

---

## License

[MIT](LICENSE)

---

> Star this repo if you found it useful!
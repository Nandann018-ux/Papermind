# PaperMind — Chat with Research Papers

## Live Demo
[**Try PaperMind here**](https://your-app-link.streamlit.app) ← *(replace after deploying)*

---

## What is PaperMind?

PaperMind is a **Retrieval-Augmented Generation (RAG)** application that allows you to:

- Upload any research paper in PDF format
- Ask natural language questions about it
- Get accurate, grounded answers with source references
- Understand complex papers faster without reading every word

---

## Tech Stack

| Layer | Tool |
|---|---|
| **PDF Parsing** | PyMuPDF |
| **Embeddings** | `sentence-transformers` (all-MiniLM-L6-v2) |
| **Vector Search** | NumPy cosine similarity |
| **LLM** | LLaMA 3 (8B) via Groq API |
| **UI** | Streamlit |

---

## Architecture

```
PDF Upload
    │
    ▼
Text Extraction (PyMuPDF)
    │
    ▼
Chunking (400 words, 50 word overlap)
    │
    ▼
Embedding (all-MiniLM-L6-v2)
    │
    ▼
Cosine Similarity Search  ◄── User Query (embedded)
    │
    ▼
Top-K Relevant Chunks
    │
    ▼
LLaMA 3 via Groq API
    │
    ▼
Grounded Answer + Source Chunks
```

---

## Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/your-username/papermind.git
cd papermind
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Add your Groq API key
Get a **free** key at [console.groq.com](https://console.groq.com) and paste it in the app sidebar.

---

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo → select `app.py`
4. Click **Deploy**

---

## Features

- Upload and parse any PDF research paper
- Intelligent text chunking with overlap
- Semantic search using sentence embeddings
- LLaMA 3 powered answers grounded in the document
- Source chunk viewer for full transparency
- Conversation memory within a session
- Clean, minimal UI

---

## Project Structure

```
papermind/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # You are here
```

---

## Environment

No `.env` file needed — API key is entered directly in the app sidebar.

---

> Star this Repo If you found this useful, consider starring the repo!
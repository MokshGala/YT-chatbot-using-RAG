# 🎬 AI Video Learning Assistant

> Turn any YouTube video into an interactive AI knowledge base — ask unlimited questions, get grounded answers, see your sources.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green?style=flat-square)](https://python.langchain.com)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-purple?style=flat-square&logo=google)](https://aistudio.google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## 📌 Overview

The **AI Video Learning Assistant** is a production-grade Retrieval-Augmented Generation (RAG) application that lets you have a natural language conversation with any YouTube video.

Paste a URL → the app extracts the transcript, splits it into overlapping chunks, generates semantic embeddings via Google's `text-embedding-004`, stores them in an in-memory FAISS index, and serves all future questions through a Gemini 2.5 Flash RAG chain — all in one session with **no repeated embedding calls**.

This project was built as a portfolio showcase for AI/GenAI engineering roles, demonstrating clean architecture, modular design, and production-level coding practices.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🔗 Universal URL support | `watch?v=`, `youtu.be/`, `/shorts/`, `/embed/` |
| 📥 Automatic transcript extraction | Via `youtube-transcript-api`, multilingual fallback |
| 🧠 Semantic embeddings | Google `text-embedding-004` |
| ⚡ FAISS vector store | In-memory, session-cached — built once per video |
| 💬 Unlimited Q&A | No re-indexing between questions |
| 📄 Source transparency | Each answer shows the transcript excerpts used |
| 🎨 Modern dark UI | Glassmorphism, Inter font, gradient accents |
| 🛡️ Graceful error handling | Invalid URLs, missing transcripts, API failures |
| 🔐 Secure key management | `.env` only, never exposed |

---

## 🏗️ Architecture

```
YouTube URL
    │
    ▼
youtube_loader.py ──── extract_video_id()
    │                  fetch_transcript()
    ▼
vector_store.py ─────  RecursiveCharacterTextSplitter (1000/200)
    │                  GoogleGenerativeAIEmbeddings (text-embedding-004)
    │                  FAISS.from_documents()
    ▼
rag_pipeline.py ─────  RunnableParallel
    │                    ├── retriever (FAISS, k=4) → format_docs
    │                    └── RunnablePassthrough (question)
    │                  PromptTemplate
    │                  ChatGoogleGenerativeAI (gemini-2.5-flash)
    │                  StrOutputParser
    ▼
app.py ─────────────── Streamlit UI + Session State
```

**Key Design Choices**

- **Session State caching** — The FAISS store and LangChain chain are built once and stored in `st.session_state`. Subsequent questions reuse them, keeping latency low.
- **Typed exceptions** — `InvalidYouTubeURLError`, `TranscriptUnavailableError` etc. give the UI precise, user-friendly messages.
- **Stateless modules** — `vector_store.py` and `rag_pipeline.py` have no global state; all persistence is in `app.py`.
- **`config.py` as single source of truth** — All tuneable parameters (chunk size, overlap, top-k, model names) in one file.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language Model | Gemini 2.5 Flash |
| Embeddings | Google `text-embedding-004` |
| Vector Store | FAISS (CPU) |
| RAG Orchestration | LangChain 0.2 (Runnables) |
| Transcript | `youtube-transcript-api` |
| UI Framework | Streamlit 1.35 |
| Environment | `python-dotenv` |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- A Google Gemini API key → [Get one here](https://aistudio.google.com/app/apikey)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ai-video-learning-assistant.git
cd ai-video-learning-assistant

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Open .env and add your GOOGLE_API_KEY
```

### Running the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | ✅ Yes | Your Google AI Studio API key |

Create a `.env` file (see `.env.example`) — **never commit it**.

---

## 📁 Project Structure

```
YT-chatbot-using-RAG/
├── app.py                  # Streamlit entry point — UI wiring only
├── config.py               # All constants & tuneable parameters
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .env.example            # API key template
├── .gitignore              # Excludes .env, .venv, __pycache__
└── src/
    ├── __init__.py
    ├── youtube_loader.py   # URL validation + transcript extraction
    ├── vector_store.py     # Text splitting + embedding + FAISS index
    ├── rag_pipeline.py     # LangChain chain construction + invocation
    ├── prompts.py          # PromptTemplate definitions
    └── utils.py            # Shared helpers (metadata, formatters, sanitiser)
```

---

## 📸 Screenshots

> _Add screenshots here after first run_

| Video Processing | Chat Interface |
|---|---|
| _(screenshot)_ | _(screenshot)_ |

---

## 🗺️ Future Roadmap

- [ ] **Multi-video support** — Load multiple videos and query across all of them
- [ ] **MMR retrieval** — Maximal Marginal Relevance for more diverse context chunks
- [ ] **Hybrid retrieval** — BM25 + FAISS with reranking (Cohere / Jina)
- [ ] **Query rewriting** — LLM-powered query expansion before retrieval
- [ ] **Conversation memory** — Multi-turn context with LangChain ConversationBufferMemory
- [ ] **Answer citations** — Timestamp-level citations linking back to the video
- [ ] **LangSmith tracing** — Full RAG pipeline observability
- [ ] **Deployment** — Streamlit Cloud / Docker / GCP Cloud Run

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Built with ❤️ · Powered by Google Gemini · Orchestrated by LangChain</p>
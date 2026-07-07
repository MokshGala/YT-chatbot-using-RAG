"""
config.py
---------
Central configuration for the AI Video Learning Assistant.
All tuneable parameters live here — no hardcoded values elsewhere.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Credentials ──────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# ── Model Configuration ──────────────────────────────────────────────────────
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL: str = "gemini-2.5-flash"
LLM_TEMPERATURE: float = 0.0

# ── Text Splitting ────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVER_K: int = 4
RETRIEVER_SEARCH_TYPE: str = "similarity"

# ── Transcript Languages (priority order) ─────────────────────────────────────
TRANSCRIPT_LANGUAGES: list[str] = ["en", "en-US", "en-GB"]

# ── Application Metadata ──────────────────────────────────────────────────────
APP_NAME: str = "AI Video Learning Assistant"
APP_DESCRIPTION: str = (
    "Ask any question about a YouTube video. "
    "Powered by Google Gemini and FAISS retrieval."
)
APP_VERSION: str = "1.0.0"

TECH_STACK: list[dict] = [
    {"name": "Gemini 2.5 Flash", "role": "Language Model"},
    {"name": "text-embedding-004", "role": "Embeddings"},
    {"name": "FAISS", "role": "Vector Store"},
    {"name": "LangChain", "role": "RAG Orchestration"},
    {"name": "Streamlit", "role": "UI Framework"},
]

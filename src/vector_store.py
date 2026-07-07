"""
src/vector_store.py
-------------------
Handles text splitting, embedding generation, and FAISS index construction.

This module is intentionally stateless — it takes raw text in and returns
a ready-to-query FAISS store. Session state management is the app's concern.
"""

from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    RETRIEVER_K,
    RETRIEVER_SEARCH_TYPE,
)


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VectorStoreResult:
    """Holds the FAISS store and build statistics."""
    store: FAISS
    chunk_count: int


# ── Core Functions ────────────────────────────────────────────────────────────

def build_vector_store(transcript_text: str) -> VectorStoreResult:
    """
    Split the transcript into chunks, embed each chunk, and build a FAISS index.

    This is the most expensive operation in the pipeline and should be called
    only once per video. Store the result in Streamlit session state.

    Args:
        transcript_text: The raw transcript string from youtube_loader.

    Returns:
        A VectorStoreResult with the FAISS store and chunk count.

    Raises:
        Exception: Propagated from LangChain or Google API on failure.
    """
    chunks = _split_text(transcript_text)
    embeddings_model = _create_embeddings_model()
    store = FAISS.from_documents(chunks, embeddings_model)

    return VectorStoreResult(store=store, chunk_count=len(chunks))


def get_retriever(store: FAISS) -> VectorStoreRetriever:
    """
    Wrap a FAISS store in a configured LangChain retriever.

    Args:
        store: A built FAISS vector store.

    Returns:
        A VectorStoreRetriever ready to accept queries.
    """
    return store.as_retriever(
        search_type=RETRIEVER_SEARCH_TYPE,
        search_kwargs={"k": RETRIEVER_K},
    )


# ── Private Helpers ───────────────────────────────────────────────────────────

def _split_text(text: str):
    """Split raw text into overlapping chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.create_documents([text])


def _create_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    """Instantiate the Google Generative AI embeddings model from config."""
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

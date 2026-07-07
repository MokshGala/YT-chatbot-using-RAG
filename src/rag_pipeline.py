"""
src/rag_pipeline.py
--------------------
Constructs and invokes the LangChain RAG chain.

The chain follows the original tutorial architecture:
  RunnableParallel (retriever → context, question passthrough)
  → PromptTemplate
  → Gemini LLM
  → StrOutputParser

Source documents are retrieved separately so the UI can display them
without modifying the main chain's output shape.
"""

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

from config import LLM_MODEL, LLM_TEMPERATURE
from src.prompts import get_rag_prompt


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """Holds the generated answer and the source documents used."""
    answer: str
    source_documents: list[Document]


# ── Chain Construction ────────────────────────────────────────────────────────

def build_chain(retriever: VectorStoreRetriever):
    """
    Build the full LangChain RAG chain.

    Architecture:
        question ──► RunnableParallel ──► PromptTemplate ──► LLM ──► StrOutputParser
                          │
                    ┌─────┴───────┐
                    │             │
               retriever     passthrough
               (context)     (question)

    Args:
        retriever: A configured FAISS-backed LangChain retriever.

    Returns:
        An invocable LangChain Runnable that accepts a question string
        and returns the answer string.
    """
    prompt = get_rag_prompt()
    llm = _create_llm()
    parser = StrOutputParser()

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(_format_docs),
        "question": RunnablePassthrough(),
    })

    return parallel_chain | prompt | llm | parser


# ── Invocation ────────────────────────────────────────────────────────────────

def ask(chain, retriever: VectorStoreRetriever, question: str) -> RAGResponse:
    """
    Run the RAG chain for a given question and return the answer with sources.

    The retriever is called separately to capture source documents for UI display.
    FAISS similarity search is a fast in-memory operation so the overhead is
    negligible compared to the LLM call.

    Args:
        chain: The compiled LangChain Runnable from build_chain().
        retriever: The same retriever used to build the chain.
        question: The user's question string.

    Returns:
        A RAGResponse with the answer and the list of source Document objects.

    Raises:
        ValueError: If the question is empty after stripping whitespace.
        Exception: Propagated from LangChain or Gemini API on failure.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    source_documents = retriever.invoke(question)
    answer = chain.invoke(question)

    return RAGResponse(answer=answer, source_documents=source_documents)


# ── Private Helpers ───────────────────────────────────────────────────────────

def _create_llm() -> ChatGoogleGenerativeAI:
    """Instantiate the Gemini LLM from config."""
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )


def _format_docs(documents: list[Document]) -> str:
    """Join retrieved document page content into a single context string."""
    return "\n\n".join(doc.page_content for doc in documents)

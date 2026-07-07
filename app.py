"""
app.py
------
Streamlit entry point for the AI Video Learning Assistant.

This file is responsible ONLY for UI wiring:
- Session state initialisation
- Sidebar rendering
- URL input and video processing
- Chat interface

All business logic lives in src/.
"""

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

from config import (
    APP_DESCRIPTION,
    APP_NAME,
    APP_VERSION,
    GOOGLE_API_KEY,
    TECH_STACK,
)
from src.youtube_loader import (
    InvalidYouTubeURLError,
    TranscriptUnavailableError,
    VideoLoadError,
    extract_video_id,
    fetch_transcript,
)
from src.vector_store import build_vector_store, get_retriever
from src.rag_pipeline import build_chain, ask
from src.utils import fetch_video_metadata, format_source_chunks, sanitise_question


# ── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  /* ── Global ── */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* ── Main background ── */
  .stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.04);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(20px);
  }

  /* ── Header ── */
  .app-header {
    text-align: center;
    padding: 2rem 0 1rem;
  }
  .app-header h1 {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a855f7, #6366f1, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
  }
  .app-header p {
    color: rgba(255,255,255,0.55);
    font-size: 1rem;
    margin-top: 0.5rem;
    font-weight: 400;
  }

  /* ── Cards ── */
  .glass-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
    transition: border-color 0.2s ease;
  }
  .glass-card:hover {
    border-color: rgba(168, 85, 247, 0.35);
  }

  /* ── Status badge ── */
  .status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
  }
  .status-idle {
    background: rgba(99,102,241,0.15);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.3);
  }
  .status-processing {
    background: rgba(245,158,11,0.15);
    color: #fcd34d;
    border: 1px solid rgba(245,158,11,0.3);
  }
  .status-ready {
    background: rgba(16,185,129,0.15);
    color: #6ee7b7;
    border: 1px solid rgba(16,185,129,0.3);
  }
  .status-error {
    background: rgba(239,68,68,0.15);
    color: #fca5a5;
    border: 1px solid rgba(239,68,68,0.3);
  }

  /* ── Stat chips ── */
  .stat-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 0.5rem;
  }
  .stat-chip {
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 0.75rem;
    color: #c7d2fe;
  }

  /* ── Tech stack badges ── */
  .tech-badge {
    display: inline-block;
    background: rgba(168,85,247,0.12);
    border: 1px solid rgba(168,85,247,0.25);
    border-radius: 6px;
    padding: 3px 9px;
    font-size: 0.72rem;
    color: #d8b4fe;
    margin: 2px;
    font-weight: 500;
  }

  /* ── Chat messages ── */
  [data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    margin-bottom: 0.75rem !important;
  }

  /* ── Source expander ── */
  .source-expander {
    margin-top: 0.5rem;
  }

  /* ── URL input override ── */
  .stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: white !important;
    font-family: 'Inter', sans-serif !important;
  }
  .stTextInput > div > div > input:focus {
    border-color: rgba(168,85,247,0.6) !important;
    box-shadow: 0 0 0 3px rgba(168,85,247,0.15) !important;
  }

  /* ── Process button ── */
  .stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.02em !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important;
  }
  .stButton > button:active {
    transform: translateY(0) !important;
  }

  /* ── Divider ── */
  hr {
    border-color: rgba(255,255,255,0.08) !important;
  }

  /* ── Sidebar labels ── */
  .sidebar-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(255,255,255,0.35);
    font-weight: 600;
    margin-bottom: 4px;
  }

  /* ── Video thumbnail ── */
  .video-thumb {
    border-radius: 10px;
    width: 100%;
    border: 1px solid rgba(255,255,255,0.1);
  }

  /* ── Section heading ── */
  .section-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgba(255,255,255,0.35);
    font-weight: 600;
    margin-bottom: 8px;
  }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialisation ──────────────────────────────────────────────

def _init_session_state() -> None:
    """Initialise all session state keys with safe defaults."""
    defaults = {
        "vector_store": None,
        "retriever": None,
        "chain": None,
        "chat_history": [],
        "video_metadata": None,
        "transcript_stats": None,   # TranscriptResult
        "chunk_count": 0,
        "processing_status": "idle",  # idle | processing | ready | error
        "current_video_id": None,
        "last_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    """Render the full application sidebar."""
    with st.sidebar:
        # App branding
        st.markdown(f"## 🎬 {APP_NAME}")
        st.markdown(f"<span style='color:rgba(255,255,255,0.4);font-size:0.8rem'>v{APP_VERSION}</span>", unsafe_allow_html=True)
        st.markdown("---")

        # Processing status
        st.markdown('<p class="sidebar-label">Status</p>', unsafe_allow_html=True)
        _render_status_badge(st.session_state.processing_status)
        st.markdown("---")

        # Video information (shown after successful processing)
        if st.session_state.processing_status == "ready" and st.session_state.video_metadata:
            meta = st.session_state.video_metadata
            st.markdown('<p class="sidebar-label">Current Video</p>', unsafe_allow_html=True)
            if meta.thumbnail_url:
                st.markdown(
                    f'<img src="{meta.thumbnail_url}" class="video-thumb"/>',
                    unsafe_allow_html=True,
                )
                st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown(f"**{meta.title}**")
            st.markdown(f"<span style='color:rgba(255,255,255,0.5);font-size:0.85rem'>by {meta.author}</span>", unsafe_allow_html=True)
            st.markdown(
                f'<a href="{meta.watch_url}" target="_blank" style="color:#a78bfa;font-size:0.8rem;">▶ Open on YouTube</a>',
                unsafe_allow_html=True,
            )
            st.markdown("---")

            # Stats
            st.markdown('<p class="sidebar-label">Transcript Stats</p>', unsafe_allow_html=True)
            stats = st.session_state.transcript_stats
            if stats:
                st.markdown(
                    f"""<div class="stat-row">
                        <div class="stat-chip">📝 {stats.word_count:,} words</div>
                        <div class="stat-chip">🔤 {stats.character_count:,} chars</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            st.markdown('<p class="sidebar-label" style="margin-top:0.75rem">Vector Store Stats</p>', unsafe_allow_html=True)
            st.markdown(
                f"""<div class="stat-row">
                    <div class="stat-chip">🧩 {st.session_state.chunk_count} chunks</div>
                </div>""",
                unsafe_allow_html=True,
            )
            st.markdown("---")

        # Tech stack
        st.markdown('<p class="sidebar-label">Tech Stack</p>', unsafe_allow_html=True)
        for item in TECH_STACK:
            st.markdown(
                f'<span class="tech-badge">{item["name"]}</span>',
                unsafe_allow_html=True,
            )
        st.markdown("---")
        st.markdown(
            '<p style="color:rgba(255,255,255,0.2);font-size:0.7rem;text-align:center">Built with ❤ for AI Portfolio</p>',
            unsafe_allow_html=True,
        )


def _render_status_badge(status: str) -> None:
    """Render a coloured status badge based on current processing state."""
    config_map = {
        "idle":       ("status-idle",       "○ Idle",       "Waiting for a YouTube URL"),
        "processing": ("status-processing", "◌ Processing", "Building your knowledge base…"),
        "ready":      ("status-ready",      "● Ready",      "Knowledge base loaded. Ask away!"),
        "error":      ("status-error",      "✕ Error",      st.session_state.last_error or "An error occurred"),
    }
    css_class, label, subtitle = config_map.get(status, config_map["idle"])
    st.markdown(
        f'<div class="status-badge {css_class}">{label}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:rgba(255,255,255,0.4);font-size:0.8rem;margin-top:6px">{subtitle}</p>',
        unsafe_allow_html=True,
    )


# ── Video Processing ──────────────────────────────────────────────────────────

def _process_video(url: str) -> None:
    """
    Orchestrate the full indexing pipeline for a YouTube URL.

    Steps:
        1. Extract and validate video ID
        2. Check if this video is already loaded (skip if same ID)
        3. Fetch transcript
        4. Fetch video metadata
        5. Build FAISS vector store
        6. Build the RAG chain
        7. Persist everything to session state

    Args:
        url: Raw YouTube URL string from the text input.
    """
    try:
        video_id = extract_video_id(url)
    except InvalidYouTubeURLError as exc:
        st.error(f"**Invalid URL** — {exc}")
        return

    # Avoid rebuilding if the same video is already loaded
    if (
        st.session_state.current_video_id == video_id
        and st.session_state.processing_status == "ready"
    ):
        st.info("✅ This video is already loaded. You can start asking questions!")
        return

    st.session_state.processing_status = "processing"
    st.session_state.last_error = None
    st.session_state.chat_history = []

    progress_placeholder = st.empty()

    try:
        with progress_placeholder.container():
            # Step 1: Transcript
            with st.spinner("📥 Fetching transcript…"):
                transcript_result = fetch_transcript(video_id)

            # Step 2: Metadata (non-blocking — failure is acceptable)
            with st.spinner("🎬 Fetching video metadata…"):
                video_metadata = fetch_video_metadata(video_id)

            # Step 3: Embeddings + FAISS
            with st.spinner(f"🧠 Building embeddings for {transcript_result.word_count:,} words…"):
                vs_result = build_vector_store(transcript_result.text)

            # Step 4: Chain
            with st.spinner("⚡ Assembling RAG chain…"):
                retriever = get_retriever(vs_result.store)
                chain = build_chain(retriever)

        # Persist to session state
        st.session_state.vector_store = vs_result.store
        st.session_state.retriever = retriever
        st.session_state.chain = chain
        st.session_state.transcript_stats = transcript_result
        st.session_state.chunk_count = vs_result.chunk_count
        st.session_state.video_metadata = video_metadata
        st.session_state.current_video_id = video_id
        st.session_state.processing_status = "ready"

        progress_placeholder.empty()
        st.success(
            f"✅ **{video_metadata.title}** is ready! "
            f"Indexed {vs_result.chunk_count} chunks from {transcript_result.word_count:,} words."
        )
        st.rerun()

    except TranscriptUnavailableError as exc:
        _set_error(str(exc))
        st.error(f"**Transcript Unavailable** — {exc}")
    except VideoLoadError as exc:
        _set_error(str(exc))
        st.error(f"**Video Load Error** — {exc}")
    except Exception as exc:
        _set_error(str(exc))
        st.error(f"**Unexpected Error** — {exc}")


def _set_error(message: str) -> None:
    """Transition to the error state with a descriptive message."""
    st.session_state.processing_status = "error"
    st.session_state.last_error = message


# ── Chat Interface ─────────────────────────────────────────────────────────────

def _render_chat_interface() -> None:
    """Render the scrollable chat history and the input box."""
    # Header row with Clear Chat button
    col_title, col_btn = st.columns([5, 1])
    with col_title:
        st.markdown("### 💬 Chat")
    with col_btn:
        if st.session_state.chat_history:
            if st.button("🗑 Clear", key="clear_chat", help="Clear conversation history"):
                st.session_state.chat_history = []
                st.rerun()

    # Render existing history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📄 View Source Excerpts", expanded=False):
                    for excerpt in message["sources"]:
                        st.markdown(excerpt)
                        st.markdown("---")

    # Chat input (only active when a video is ready)
    is_ready = st.session_state.processing_status == "ready"
    placeholder_text = (
        "Ask anything about the video…"
        if is_ready
        else "Process a video above to start chatting"
    )

    if question := st.chat_input(placeholder_text, disabled=not is_ready, key="chat_input"):
        _handle_question(question)


def _handle_question(raw_question: str) -> None:
    """
    Process a user question through the RAG chain and update history.

    Args:
        raw_question: The raw string from st.chat_input.
    """
    question = sanitise_question(raw_question)
    if not question:
        st.warning("Please enter a question before submitting.")
        return

    # Append user message immediately
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            try:
                response = ask(
                    chain=st.session_state.chain,
                    retriever=st.session_state.retriever,
                    question=question,
                )
                formatted_sources = format_source_chunks(response.source_documents)

                st.markdown(response.answer)

                if formatted_sources:
                    with st.expander("📄 View Source Excerpts", expanded=False):
                        for excerpt in formatted_sources:
                            st.markdown(excerpt)
                            st.markdown("---")

                # Persist assistant turn to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": formatted_sources,
                })

            except ValueError as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"**Generation Error** — {exc}")


# ── Main Layout ───────────────────────────────────────────────────────────────

def main() -> None:
    """Application entry point. Renders all UI sections."""
    _init_session_state()

    # Guard: missing API key
    if not GOOGLE_API_KEY:
        st.error(
            "**Missing GOOGLE_API_KEY** — Create a `.env` file with your key. "
            "See `.env.example` for reference."
        )
        st.stop()

    _render_sidebar()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="app-header">
            <h1>🎬 AI Video Learning Assistant</h1>
            <p>{APP_DESCRIPTION}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── URL Input Section ─────────────────────────────────────────────────────
    st.markdown("#### 🔗 YouTube Video URL")
    st.markdown(
        '<p style="color:rgba(255,255,255,0.45);font-size:0.85rem;margin-top:-0.5rem">'
        "Paste any YouTube URL — regular videos, Shorts, and embed links are all supported."
        "</p>",
        unsafe_allow_html=True,
    )

    col_input, col_btn = st.columns([5, 1], vertical_alignment="bottom")
    with col_input:
        url_input = st.text_input(
            label="YouTube URL",
            label_visibility="collapsed",
            placeholder="https://www.youtube.com/watch?v=...",
            key="url_input",
        )
    with col_btn:
        process_clicked = st.button(
            "⚡ Process",
            key="process_btn",
            use_container_width=True,
            help="Extract transcript and build the knowledge base",
        )

    if process_clicked:
        if not url_input.strip():
            st.warning("Please enter a YouTube URL first.")
        else:
            _process_video(url_input.strip())

    st.markdown("---")

    # ── Chat Section ──────────────────────────────────────────────────────────
    if st.session_state.processing_status == "idle":
        st.markdown(
            """
            <div style="text-align:center;padding:3rem 0;color:rgba(255,255,255,0.3)">
                <div style="font-size:3rem">🎬</div>
                <p style="font-size:1rem;margin-top:0.75rem">
                    Paste a YouTube URL above and click <strong>Process</strong><br/>
                    to create your AI-powered knowledge base.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif st.session_state.processing_status == "error":
        st.markdown(
            """
            <div style="text-align:center;padding:2rem 0;color:rgba(255,255,255,0.3)">
                <div style="font-size:2.5rem">⚠️</div>
                <p style="font-size:0.9rem;margin-top:0.5rem">
                    Fix the error above and try a different URL.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        _render_chat_interface()


if __name__ == "__main__":
    main()

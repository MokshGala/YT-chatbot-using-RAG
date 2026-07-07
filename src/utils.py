"""
src/utils.py
------------
Shared utility functions used across the application.

Responsibilities:
- Fetch video metadata from the YouTube oEmbed API (no API key required)
- Format source document chunks for display
- Sanitise user input
"""

import re
from dataclasses import dataclass, field

import requests
from langchain_core.documents import Document


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class VideoMetadata:
    """Metadata retrieved from the YouTube oEmbed API."""
    video_id: str
    title: str = "Unknown Title"
    author: str = "Unknown Author"
    thumbnail_url: str = ""

    @property
    def watch_url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"

    @property
    def embed_url(self) -> str:
        return f"https://www.youtube.com/embed/{self.video_id}"


# ── Video Metadata ─────────────────────────────────────────────────────────────

def fetch_video_metadata(video_id: str) -> VideoMetadata:
    """
    Fetch video title and author using the YouTube oEmbed API.

    This endpoint requires no API key and is suitable for public videos.
    Falls back to a default VideoMetadata object on any failure.

    Args:
        video_id: The 11-character YouTube video ID.

    Returns:
        A VideoMetadata dataclass populated with available info.
    """
    oembed_url = (
        f"https://www.youtube.com/oembed"
        f"?url=https://www.youtube.com/watch?v={video_id}&format=json"
    )
    try:
        response = requests.get(oembed_url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return VideoMetadata(
            video_id=video_id,
            title=data.get("title", "Unknown Title"),
            author=data.get("author_name", "Unknown Author"),
            thumbnail_url=data.get(
                "thumbnail_url",
                f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
            ),
        )
    except Exception:
        # Silently fall back — metadata is nice to have, not required
        return VideoMetadata(
            video_id=video_id,
            thumbnail_url=f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
        )


# ── Source Document Formatting ────────────────────────────────────────────────

def format_source_chunks(documents: list[Document]) -> list[str]:
    """
    Extract and lightly format page content from retrieved Documents.

    Args:
        documents: List of LangChain Document objects from the retriever.

    Returns:
        A list of formatted excerpt strings for display in the UI.
    """
    excerpts = []
    for i, doc in enumerate(documents, start=1):
        content = doc.page_content.strip()
        # Truncate very long chunks for readability in the UI
        if len(content) > 400:
            content = content[:397] + "..."
        excerpts.append(f"**Excerpt {i}:** {content}")
    return excerpts


# ── Input Sanitisation ────────────────────────────────────────────────────────

def sanitise_question(question: str) -> str:
    """
    Strip whitespace and collapse internal multiple spaces from a question.

    Args:
        question: Raw question string from the chat input.

    Returns:
        Cleaned question string.
    """
    return re.sub(r"\s+", " ", question.strip())

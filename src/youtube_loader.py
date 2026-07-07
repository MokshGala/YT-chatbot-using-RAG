"""
src/youtube_loader.py
---------------------
Handles all YouTube URL parsing and transcript extraction.

Responsibilities:
- Validate and normalise any YouTube URL format
- Extract the video ID
- Fetch the transcript via YouTubeTranscriptApi (v1.x)
- Raise descriptive, typed exceptions so the UI can surface friendly messages

Note: Compatible with youtube-transcript-api >= 1.0.0
      v1.x changed from static class methods to an instance-based API:
      - Old: YouTubeTranscriptApi.get_transcript(video_id)
      - New: YouTubeTranscriptApi().fetch(video_id)
"""

import re
from dataclasses import dataclass

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from config import TRANSCRIPT_LANGUAGES


# ── Custom Exceptions ─────────────────────────────────────────────────────────

class VideoLoadError(Exception):
    """Raised for any unrecoverable error during video loading."""


class InvalidYouTubeURLError(VideoLoadError):
    """Raised when the supplied string is not a recognisable YouTube URL."""


class TranscriptUnavailableError(VideoLoadError):
    """Raised when no suitable transcript can be fetched for the video."""


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TranscriptResult:
    """Holds the raw transcript and basic statistics."""
    video_id: str
    text: str
    word_count: int
    character_count: int


# ── URL Parsing ───────────────────────────────────────────────────────────────

# Regex covers the most common YouTube URL patterns:
#   https://www.youtube.com/watch?v=VIDEO_ID
#   https://youtu.be/VIDEO_ID
#   https://www.youtube.com/shorts/VIDEO_ID
#   https://www.youtube.com/embed/VIDEO_ID
#   https://m.youtube.com/watch?v=VIDEO_ID
_YOUTUBE_ID_PATTERN = re.compile(
    r"(?:youtube\.com/(?:watch\?.*?v=|shorts/|embed/|v/)|youtu\.be/)"
    r"([a-zA-Z0-9_-]{11})"
)


def extract_video_id(url: str) -> str:
    """
    Parse any common YouTube URL and return the 11-character video ID.

    Args:
        url: A YouTube video URL in any supported format.

    Returns:
        The 11-character video ID string.

    Raises:
        InvalidYouTubeURLError: If the URL does not match any known pattern.
    """
    if not url or not url.strip():
        raise InvalidYouTubeURLError("Please enter a YouTube URL.")

    match = _YOUTUBE_ID_PATTERN.search(url.strip())
    if not match:
        raise InvalidYouTubeURLError(
            "Could not find a valid YouTube video ID in the URL. "
            "Supported formats: youtube.com/watch?v=..., youtu.be/..., "
            "youtube.com/shorts/..."
        )

    return match.group(1)


# ── Transcript Fetching ───────────────────────────────────────────────────────

def fetch_transcript(video_id: str) -> TranscriptResult:
    """
    Fetch and concatenate the transcript for the given video ID.

    Uses youtube-transcript-api v1.x instance API:
      YouTubeTranscriptApi().fetch(video_id, languages=[...])

    Returns a FetchedTranscript which is iterable with each element
    being a Snippet object. Each snippet has a .text attribute.

    Args:
        video_id: The 11-character YouTube video ID.

    Returns:
        A TranscriptResult dataclass with the full text and statistics.

    Raises:
        TranscriptUnavailableError: If no transcript can be fetched.
        VideoLoadError: For unexpected API failures.
    """
    try:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id, languages=TRANSCRIPT_LANGUAGES)
        # FetchedTranscript is iterable; each element has a .text attribute
        full_text = " ".join(snippet.text for snippet in fetched)

    except TranscriptsDisabled:
        raise TranscriptUnavailableError(
            "Transcripts are disabled for this video. "
            "The video owner has turned off captions."
        )
    except NoTranscriptFound:
        raise TranscriptUnavailableError(
            "No English transcript was found for this video. "
            "Try a video with auto-generated or manual English captions."
        )
    except VideoUnavailable:
        raise TranscriptUnavailableError(
            "This video is unavailable. It may be private, deleted, or "
            "age-restricted."
        )
    except CouldNotRetrieveTranscript as exc:
        raise TranscriptUnavailableError(
            f"Could not retrieve transcript: {exc}"
        )
    except Exception as exc:
        raise VideoLoadError(
            f"An unexpected error occurred while fetching the transcript: {exc}"
        ) from exc

    full_text = _clean_transcript(full_text)

    return TranscriptResult(
        video_id=video_id,
        text=full_text,
        word_count=len(full_text.split()),
        character_count=len(full_text),
    )


def _clean_transcript(text: str) -> str:
    """
    Lightly normalise transcript text.

    Removes common artefacts from auto-generated captions such as
    consecutive spaces and music/sound-effect annotations.
    """
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    # Remove common auto-caption artefacts like [Music] or [Applause]
    text = re.sub(r"\[(?:Music|Applause|Laughter|Inaudible)\]", "", text, flags=re.IGNORECASE)
    return text.strip()

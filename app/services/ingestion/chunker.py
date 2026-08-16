"""
Chunking: splitting extracted text into retrievable pieces.

Concept note (this is the practical side of Phase 3's "chunking tradeoff"):
- Too big a chunk -> retrieval pulls back a wall of text, most of it
  irrelevant to the query, diluting what the LLM actually needs to see.
- Too small a chunk -> you lose surrounding context; a sentence like
  "It has this side effect at that dose" is useless without the
  paragraph that says what "it" and "that dose" refer to.
- Overlap between consecutive chunks exists so a sentence that happens to
  fall right on a chunk boundary doesn't get orphaned from its context on
  either side.

This is a word-based splitter, not a real tokenizer. A production system
would chunk by actual model tokens (via tiktoken or similar) since that's
what the context window and cost are measured in — a documented
simplification for Day 2, worth upgrading later, not a hidden bug.

chunk_size/overlap default to settings values (tunable via
FIXED_CHUNK_SIZE_WORDS / FIXED_CHUNK_OVERLAP_WORDS in .env) rather than
fixed constants — added after a real deployment finding: on a CPU-limited
free-tier host, more/smaller chunks means more sequential embedding
calls per upload, which can exceed the host's request timeout on large
documents. A larger chunk size trades a little retrieval precision for
meaningfully fewer embedding calls — worth tuning per-environment, not
hardcoding one value for both local dev and a constrained deployment.
"""

from app.core.config import settings

DEFAULT_CHUNK_SIZE_WORDS = 220   # roughly ~300 tokens for English text
DEFAULT_OVERLAP_WORDS = 40


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    if chunk_size is None:
        chunk_size = settings.fixed_chunk_size_words
    if overlap is None:
        overlap = settings.fixed_chunk_overlap_words

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start += step

    return chunks

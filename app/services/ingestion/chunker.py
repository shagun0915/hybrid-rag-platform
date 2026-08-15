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
"""

DEFAULT_CHUNK_SIZE_WORDS = 220   # roughly ~300 tokens for English text
DEFAULT_OVERLAP_WORDS = 40


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap: int = DEFAULT_OVERLAP_WORDS,
) -> list[str]:
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

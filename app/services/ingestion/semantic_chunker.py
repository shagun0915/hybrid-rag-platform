"""
Semantic chunking — directly built to fix a real, diagnosed bug: the
SonarQube retrieval failure documented in the README's Known Limitations
section. Root cause was word-count chunking cutting a sentence in half
mid-fact ("...remediation of security findings (Checkmarx, SonarQube)"
split so the retrieved chunk had "SonarQube" but not "remediation").
Fixed-size chunking has no concept of a sentence boundary, let alone a
topic boundary — it just counts words and cuts.

How this works instead:
1. Split text into sentences (not words).
2. Embed every sentence individually (reusing the same embedding model
   already used everywhere else — no new model, no new dependency).
3. Compute cosine similarity between each pair of *consecutive*
   sentences. High similarity = still on the same topic, keep them
   together. A similarity drop = likely topic shift, that's where a
   chunk boundary belongs.
4. Group consecutive sentences into chunks at those topic-shift points,
   with two safety caps: `semantic_chunk_max_words` (so a very
   topically-consistent section doesn't produce one giant chunk) and
   `semantic_chunk_min_words` (so a bullet-heavy document like a resume
   doesn't fragment into dozens of one-sentence chunks — bullets often
   score as "topic shifts" against each other even when they're all
   part of the same logical section).

Honest limitation: sentence splitting here is regex-based, not a real
NLP sentence tokenizer (spaCy/NLTK). It handles standard punctuation and
bullet characters correctly but will occasionally mis-split on
abbreviations ("approx.", "e.g.") — a known, documented tradeoff to
avoid pulling in a much heavier NLP dependency for one edge case.

Cost tradeoff, also documented rather than hidden: this embeds every
sentence for boundary detection, then the resulting chunks get embedded
again afterward for storage (in pipeline.py) — genuinely redundant
compute. A more optimized version would reuse/average the sentence
embeddings instead of re-embedding whole chunks. Not built here — this
version prioritizes being obviously correct over being maximally
efficient, which is the right tradeoff for a project at this scale.
"""

import re
import math

from app.core.config import settings
from app.services.ingestion.embedder import embed_texts

_BULLET_RE = re.compile(r"[•●▪‣·]")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Regex-based sentence splitting. Treats bullet characters as hard
    boundaries first (common in resumes/reports), then splits remaining
    text on sentence-ending punctuation followed by a capital letter or
    digit (avoids splitting on periods inside numbers/abbreviations in
    the common case, though not perfectly)."""
    bullet_parts = _BULLET_RE.split(text)

    sentences = []
    for part in bullet_parts:
        part = part.strip()
        if not part:
            continue
        for sentence in _SENTENCE_END_RE.split(part):
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)

    return sentences


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _group_sentences(
    sentences: list[str],
    similarities: list[float],
    threshold: float,
    max_words: int,
    min_words: int,
) -> list[str]:
    """Pure grouping logic, deliberately separated from the embedding
    call above — this is what makes it fully unit-testable with fake
    similarity scores, no real model required (see
    tests/test_semantic_chunker.py)."""
    if not sentences:
        return []

    chunks: list[str] = []
    current = [sentences[0]]
    current_word_count = len(sentences[0].split())

    for i in range(1, len(sentences)):
        similarity = similarities[i - 1]
        sentence_word_count = len(sentences[i].split())

        topic_shift = similarity < threshold
        below_min_size = current_word_count < min_words
        would_exceed_max = (current_word_count + sentence_word_count) > max_words

        if would_exceed_max or (topic_shift and not below_min_size):
            chunks.append(" ".join(current))
            current = [sentences[i]]
            current_word_count = sentence_word_count
        else:
            current.append(sentences[i])
            current_word_count += sentence_word_count

    if current:
        chunks.append(" ".join(current))

    return chunks


async def semantic_chunk_text(text: str) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences

    embeddings = await embed_texts(sentences)
    similarities = [
        cosine_similarity(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]

    return _group_sentences(
        sentences,
        similarities,
        threshold=settings.semantic_similarity_threshold,
        max_words=settings.semantic_chunk_max_words,
        min_words=settings.semantic_chunk_min_words,
    )

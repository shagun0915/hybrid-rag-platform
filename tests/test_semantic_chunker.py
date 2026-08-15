"""
Tests for semantic chunking. `split_sentences`, `cosine_similarity`, and
`_group_sentences` are all pure — no embedding model, no DB, no LLM —
so they're fully testable offline with fabricated similarity scores.
The full `semantic_chunk_text` (which actually calls the embedding
model) is exercised manually via document upload, same pattern as every
other model-dependent piece all week.
"""

from app.services.ingestion.semantic_chunker import (
    split_sentences,
    cosine_similarity,
    _group_sentences,
)


def test_split_sentences_basic():
    text = "This is one sentence. This is another sentence. And a third."
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "This is one sentence."
    assert sentences[2] == "And a third."


def test_split_sentences_handles_bullets():
    # This is the exact real-world case that motivated this whole
    # feature: resume-style bullet content where fixed-size chunking
    # previously split a fact from its context mid-sentence.
    text = "EARLIER EXPERIENCE•Raahee — Software Development Intern: Built the app.•Microsoft Engage mentee."
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert "Raahee" in sentences[1]
    assert "Microsoft Engage" in sentences[2]


def test_split_sentences_empty_input():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_cosine_similarity_identical_vectors_is_one():
    v = [1.0, 2.0, 3.0]
    assert round(cosine_similarity(v, v), 4) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert round(cosine_similarity(a, b), 4) == 0.0


def test_cosine_similarity_opposite_vectors_is_negative_one():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert round(cosine_similarity(a, b), 4) == -1.0


def test_cosine_similarity_zero_vector_handled_safely():
    # Must not raise a divide-by-zero error
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_group_sentences_keeps_similar_sentences_together():
    sentences = ["A about topic X.", "B also about topic X.", "C about totally different topic Y."]
    # High similarity between sentence 0-1, low similarity 1-2
    similarities = [0.9, 0.2]
    chunks = _group_sentences(sentences, similarities, threshold=0.5, max_words=1000, min_words=0)

    assert len(chunks) == 2
    assert "A about topic X." in chunks[0]
    assert "B also about topic X." in chunks[0]
    assert chunks[1] == "C about totally different topic Y."


def test_group_sentences_respects_max_words_cap():
    # Even with perfect similarity throughout, a max_words cap must
    # still force a break — this is the safety valve against one giant
    # chunk from a very topically-consistent section.
    sentences = ["word " * 10, "word " * 10, "word " * 10]
    similarities = [0.99, 0.99]  # very similar, would otherwise never break
    chunks = _group_sentences(sentences, similarities, threshold=0.5, max_words=15, min_words=0)

    assert len(chunks) > 1


def test_group_sentences_respects_min_words_floor():
    # A topic shift right after a tiny first sentence should NOT force
    # an immediate break — this is the guard against resume bullet
    # points fragmenting into dozens of one-sentence chunks.
    sentences = ["Short.", "Completely different topic here about something else entirely."]
    similarities = [0.1]  # strong topic shift signal
    chunks = _group_sentences(sentences, similarities, threshold=0.5, max_words=1000, min_words=20)

    # min_words=20 > "Short." word count, so it should NOT break despite low similarity
    assert len(chunks) == 1


def test_group_sentences_empty_input():
    assert _group_sentences([], [], threshold=0.5, max_words=100, min_words=10) == []

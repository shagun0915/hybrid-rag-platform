"""
Day 5 tests: `_sigmoid` (reranker.py) and `_should_stop`
(agentic_retrieval.py) are pure logic — no DB, no model, no LLM call, so
they're fully testable offline. The actual cross-encoder inference and
the live agentic loop (real hybrid_search + real reranking + real LLM
reformulation) are exercised manually via /query, same pattern as every
previous day.
"""

from app.services.retrieval.reranker import _sigmoid
from app.services.retrieval.agentic_retrieval import _should_stop, _summarize_candidates


def test_sigmoid_zero_is_midpoint():
    assert _sigmoid(0.0) == 0.5


def test_sigmoid_large_positive_approaches_one():
    assert _sigmoid(10.0) > 0.99


def test_sigmoid_large_negative_approaches_zero():
    assert _sigmoid(-10.0) < 0.01


def test_sigmoid_monotonic():
    # Higher raw score should always produce a higher (or equal) sigmoid score
    assert _sigmoid(1.0) > _sigmoid(0.0)
    assert _sigmoid(5.0) > _sigmoid(1.0)


def test_should_stop_when_score_meets_threshold():
    assert _should_stop(top_score=0.8, attempt=1, max_attempts=2, threshold=0.5) is True


def test_should_not_stop_when_score_below_threshold_and_attempts_remain():
    assert _should_stop(top_score=0.2, attempt=1, max_attempts=2, threshold=0.5) is False


def test_should_stop_when_attempts_exhausted_even_with_low_score():
    # This is the critical safety case: a persistently weak query must
    # not loop forever just because the score never improves.
    assert _should_stop(top_score=0.1, attempt=2, max_attempts=2, threshold=0.5) is True


def test_should_stop_when_no_results_at_all():
    # No chunks found -> top_score is None -> must still respect the cap
    assert _should_stop(top_score=None, attempt=2, max_attempts=2, threshold=0.5) is True
    assert _should_stop(top_score=None, attempt=1, max_attempts=2, threshold=0.5) is False


def test_summarize_candidates_extracts_only_needed_fields():
    candidates = [
        {"filename": "resume.pdf", "chunk_index": 2, "fusion_score": 0.032, "content": "long text..."},
        {"filename": "paper.pdf", "chunk_index": 5, "fusion_score": 0.015, "content": "more text..."},
    ]
    summary = _summarize_candidates(candidates)

    assert summary == [
        {"filename": "resume.pdf", "chunk_index": 2, "fusion_score": 0.032},
        {"filename": "paper.pdf", "chunk_index": 5, "fusion_score": 0.015},
    ]
    # Full chunk content should NOT be duplicated into the debug summary
    assert "content" not in summary[0]


def test_summarize_candidates_can_detect_missing_document():
    # This is the exact real diagnostic use case this function was built
    # for: confirming whether a specific document appeared in the
    # pre-rerank pool at all.
    candidates = [
        {"filename": "paper.pdf", "chunk_index": 0, "fusion_score": 0.02},
    ]
    summary = _summarize_candidates(candidates)
    filenames_present = {c["filename"] for c in summary}
    assert "resume.pdf" not in filenames_present


def test_summarize_candidates_empty_list():
    assert _summarize_candidates([]) == []

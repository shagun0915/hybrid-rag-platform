"""
Tests for query reformulation grounding. `_format_context_snippets` is
pure logic — no LLM call — so it's fully testable offline. The actual
`reformulate_query` call (which hits the LLM) is exercised manually via
/query, same pattern as every other model-dependent piece.
"""

from app.services.retrieval.query_reformulation import _format_context_snippets


def test_format_context_snippets_basic():
    candidates = [
        {"filename": "Document.md", "content": "The ERP VAMP (Visa Acquirer Monitoring Program) Remediation Portal requires..."},
    ]
    result = _format_context_snippets(candidates)

    assert "Document.md" in result
    assert "Visa Acquirer Monitoring Program" in result
    assert "Context snippets actually found" in result


def test_format_context_snippets_empty_candidates_returns_empty_string():
    # This is the graceful-degradation path — no candidates from a
    # previous attempt (e.g. first-ever attempt has nothing to ground
    # against yet) falls back to ungrounded behavior, not an error.
    assert _format_context_snippets([]) == ""


def test_format_context_snippets_respects_max_snippets():
    candidates = [
        {"filename": f"doc{i}.pdf", "content": f"Content {i}"} for i in range(10)
    ]
    result = _format_context_snippets(candidates, max_snippets=3)

    assert "doc0.pdf" in result
    assert "doc1.pdf" in result
    assert "doc2.pdf" in result
    assert "doc3.pdf" not in result


def test_format_context_snippets_truncates_long_content():
    long_content = "word " * 200  # way longer than any reasonable snippet
    candidates = [{"filename": "big.pdf", "content": long_content}]
    result = _format_context_snippets(candidates, snippet_length=50)

    # The formatted snippet portion should be meaningfully shorter than
    # the full original content
    assert len(result) < len(long_content)


def test_format_context_snippets_skips_candidates_with_no_content():
    candidates = [
        {"filename": "empty.pdf", "content": ""},
        {"filename": "real.pdf", "content": "Actual text here"},
    ]
    result = _format_context_snippets(candidates)

    assert "empty.pdf" not in result
    assert "real.pdf" in result


def test_format_context_snippets_handles_missing_content_key():
    # Defensive: a malformed candidate dict shouldn't crash formatting
    candidates = [{"filename": "weird.pdf"}]
    result = _format_context_snippets(candidates)
    assert result == ""


def test_regression_vamp_case_produces_grounding_not_silence():
    # The exact real-world case that motivated this fix: a candidate
    # containing the actual VAMP definition should end up in the
    # grounding block, giving reformulation real vocabulary to work
    # with instead of guessing an unrelated domain (the original bug
    # produced "Vascular Adhesion Molecule" — biology, wrong entirely).
    real_candidate = {
        "filename": "Document.md",
        "chunk_index": 0,
        "content": (
            "The ERP VAMP (Visa Acquirer Monitoring Program) Remediation "
            "Portal requires acquirers and merchants to fill out detailed "
            "questionnaire forms during remediation plan submissions."
        ),
        "fusion_score": 0.0164,
    }
    result = _format_context_snippets([real_candidate])

    assert "Visa Acquirer Monitoring Program" in result
    assert "Vascular" not in result  # sanity: we're not injecting the old wrong guess

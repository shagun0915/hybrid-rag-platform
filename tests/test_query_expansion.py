"""
Tests for query expansion. `_parse_variants` is pure logic — no LLM call
— so it's fully testable offline. `expand_query` (which calls the LLM)
and `expanded_hybrid_search` (which calls the DB) are exercised manually
via /query, same pattern as every other model/DB-dependent piece.
"""

from app.services.retrieval.query_expansion import _parse_variants


def test_parse_variants_basic():
    raw = "How does the system remediate security issues?\nWhat tools fix vulnerabilities?"
    result = _parse_variants(raw, "What security tools were used?", num_variants=2)

    assert result[0] == "What security tools were used?"
    assert "How does the system remediate security issues?" in result
    assert "What tools fix vulnerabilities?" in result
    assert len(result) == 3


def test_parse_variants_truncates_to_num_variants():
    raw = "Variant one\nVariant two\nVariant three\nVariant four"
    result = _parse_variants(raw, "Original question", num_variants=2)

    # original + only the first 2 variants, extras dropped
    assert len(result) == 3
    assert "Variant three" not in result
    assert "Variant four" not in result


def test_parse_variants_dedupes_case_insensitively():
    raw = "what security tools were used?\nA genuinely different phrasing"
    result = _parse_variants(raw, "What security tools were used?", num_variants=2)

    # the LLM echoed the original back (different case) — should not appear twice
    assert len(result) == 2
    assert result[0] == "What security tools were used?"
    assert "A genuinely different phrasing" in result


def test_parse_variants_strips_quotes_and_blank_lines():
    raw = '"A quoted variant"\n\n   \n"Another quoted one"'
    result = _parse_variants(raw, "Original", num_variants=2)

    assert "A quoted variant" in result
    assert "Another quoted one" in result
    assert '"A quoted variant"' not in result  # quotes stripped


def test_parse_variants_empty_raw_text_returns_just_original():
    result = _parse_variants("", "Original question", num_variants=2)
    assert result == ["Original question"]

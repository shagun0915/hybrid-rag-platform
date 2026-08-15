"""
Day 4 tests: `reciprocal_rank_fusion` is pure logic — no DB connection
needed, so it's fully testable offline. `keyword_search` and
`hybrid_search` themselves need a live Postgres connection and are
exercised manually via /query, same as vector_search was in Day 3.
"""

from app.services.retrieval.hybrid_search import reciprocal_rank_fusion


def _chunk(chunk_id, **extra):
    return {"chunk_id": chunk_id, **extra}


def test_rrf_ranks_items_in_both_lists_higher():
    vector_results = [_chunk("A"), _chunk("B"), _chunk("C")]
    keyword_results = [_chunk("B"), _chunk("A"), _chunk("D")]

    fused = reciprocal_rank_fusion([vector_results, keyword_results], k=60)
    fused_ids = [f["chunk_id"] for f in fused]

    # A and B appear in both lists (near the top of each) — they should
    # rank above C and D, which only appear in one list each.
    assert fused_ids.index("A") < fused_ids.index("C")
    assert fused_ids.index("B") < fused_ids.index("D")


def test_rrf_includes_items_from_only_one_list():
    vector_results = [_chunk("A"), _chunk("B")]
    keyword_results = []  # e.g. no keyword matches at all

    fused = reciprocal_rank_fusion([vector_results, keyword_results], k=60)
    fused_ids = {f["chunk_id"] for f in fused}

    assert fused_ids == {"A", "B"}


def test_rrf_score_formula():
    # chunk "A" at rank 1 in list 1 only: score = 1/(60+1)
    vector_results = [_chunk("A")]
    keyword_results = []

    fused = reciprocal_rank_fusion([vector_results, keyword_results], k=60)

    expected = round(1.0 / 61, 5)
    assert fused[0]["fusion_score"] == expected


def test_rrf_empty_lists_returns_empty():
    assert reciprocal_rank_fusion([[], []], k=60) == []

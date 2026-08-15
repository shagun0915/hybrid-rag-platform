"""
Day 3 tests: only `_build_context_block` is pure/testable without a live
DB and a real Anthropic API key. `vector_search` needs a real pgvector
connection, and `generate_answer` needs a real API call — both are
exercised manually via /query, not in this offline test suite.
"""

from app.services.generation.answer import _build_context_block


def test_build_context_block_numbers_excerpts_in_order():
    chunks = [
        {"filename": "doc1.pdf", "content": "First chunk content."},
        {"filename": "doc2.pdf", "content": "Second chunk content."},
    ]
    block = _build_context_block(chunks)

    assert "[Excerpt 1 — source: doc1.pdf]" in block
    assert "[Excerpt 2 — source: doc2.pdf]" in block
    assert "First chunk content." in block
    assert "Second chunk content." in block
    # Excerpt 1 must appear before Excerpt 2 in the string
    assert block.index("Excerpt 1") < block.index("Excerpt 2")


def test_build_context_block_empty_list():
    assert _build_context_block([]) == ""

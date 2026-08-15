"""
Day 2 tests: chunker and parser only. Both are pure functions — no
database, no embedding model, so these run instantly and don't need
Docker or a model download. Embedding generation itself isn't unit-tested
here since it requires downloading model weights on first run; that's
exercised manually via the /documents/upload endpoint instead.
"""

import pytest

from app.services.ingestion.chunker import chunk_text
from app.services.ingestion.parser import extract_text, UnsupportedFileType, EmptyDocumentError


def test_chunk_text_basic_split():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    # every chunk should have at most chunk_size words
    for c in chunks:
        assert len(c.split()) <= 100


def test_chunk_text_overlap_preserves_boundary_words():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=40, overlap=10)

    first_chunk_words = chunks[0].split()
    second_chunk_words = chunks[1].split()
    # last 10 words of chunk 1 should reappear as the first 10 of chunk 2
    assert first_chunk_words[-10:] == second_chunk_words[:10]


def test_chunk_text_empty_input_returns_empty_list():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size=10, overlap=10)


def test_extract_text_txt():
    content = "Hello world, this is a test document.".encode("utf-8")
    result = extract_text("notes.txt", content)
    assert "Hello world" in result


def test_extract_text_unsupported_extension_raises():
    with pytest.raises(UnsupportedFileType):
        extract_text("image.png", b"fake-bytes")


def test_extract_text_empty_txt_raises():
    with pytest.raises(EmptyDocumentError):
        extract_text("empty.txt", b"   ")

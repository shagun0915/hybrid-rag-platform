"""
Embedding generation.

Concept note (Phase 3 made concrete): an embedding model takes a string
and returns a fixed-length vector of numbers — here, 384 of them — such
that pieces of text with similar *meaning* end up as vectors that are
close together in that 384-dimensional space. "Close together" is
measured by cosine similarity, which pgvector computes natively via its
`<=>` operator (used in Day 3's retrieval query).

Model choice: BAAI/bge-small-en-v1.5, via the `fastembed` library rather
than `sentence-transformers`. Same idea (a local embedding model, no API
key, no per-call cost), but fastembed runs on ONNX Runtime instead of
full PyTorch — meaningfully smaller install, faster cold start, same
384-dimension output that already matches `settings.embedding_dimension`.

The model is loaded once and reused (a "singleton") because loading model
weights from disk takes real time — you don't want to pay that cost on
every single request.
"""

import asyncio

from fastembed import TextEmbedding

_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _model


def _embed_texts_sync(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    return [vector.tolist() for vector in model.embed(texts)]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Async wrapper around a CPU-bound embedding call.

    Why `asyncio.to_thread` matters here: FastAPI's event loop is single-
    threaded for async code. Embedding generation is pure CPU work (matrix
    math), not I/O — if we called `_embed_texts_sync` directly inside an
    `async def` endpoint, it would block the entire event loop for the
    duration, meaning every other in-flight request on the server would
    freeze until embedding finished. `asyncio.to_thread` runs it in a
    background thread instead, so the event loop stays free to handle
    other requests concurrently. This is the same "don't let one slow
    thing block everything else" instinct as the health-check design from
    Day 1 — just showing up in a different spot.
    """
    return await asyncio.to_thread(_embed_texts_sync, texts)

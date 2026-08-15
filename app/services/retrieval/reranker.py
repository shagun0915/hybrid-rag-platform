"""
Cross-encoder reranking.

Concept note — why rerank at all, when hybrid_search already ranked
things: vector and keyword search both score a query against a chunk
*independently* — the chunk's embedding was computed once, in isolation,
long before your question existed. A cross-encoder instead looks at the
query and a candidate chunk *together*, in a single forward pass, so it
can pick up on interactions between them that independent scoring
misses. This is much more accurate, but also much slower per comparison
— which is exactly why it only runs on the ~10-20 candidates hybrid
search already shortlisted, not the whole corpus. Two-stage retrieval
(cheap, approximate first pass -> expensive, precise second pass) is the
standard pattern for exactly this cost/accuracy tradeoff.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 via fastembed — same ONNX
runtime already used for embeddings in Day 2, so no new heavy dependency
(no full PyTorch) gets added just for this.

Score interpretation: the raw model output is an unbounded logit, not a
0-1 probability. We apply a sigmoid to convert it into something with an
intuitive "how confident is this a relevant match" reading, mainly so
`min_rerank_score` in config.py has a meaningful, human-interpretable
threshold to tune against.
"""

import asyncio
import math

from fastembed.rerank.cross_encoder import TextCrossEncoder

_model: TextCrossEncoder | None = None


def _get_model() -> TextCrossEncoder:
    global _model
    if _model is None:
        _model = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
    return _model


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _rerank_sync(query: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []

    model = _get_model()
    documents = [c["content"] for c in candidates]
    raw_scores = list(model.rerank(query, documents))

    scored = []
    for candidate, raw_score in zip(candidates, raw_scores):
        item = dict(candidate)
        item["rerank_score"] = round(_sigmoid(float(raw_score)), 4)
        scored.append(item)

    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored


async def rerank(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    """Async wrapper — cross-encoder inference is CPU-bound, so this runs
    in a background thread rather than blocking the event loop. Same
    reasoning as embed_texts() in Day 2."""
    ranked = await asyncio.to_thread(_rerank_sync, query, candidates)
    return ranked[:top_n]

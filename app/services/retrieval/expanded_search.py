"""
Query-expanded hybrid retrieval.

Runs hybrid_search (Day 4: vector + keyword, fused via RRF) independently
for the original question and each LLM-generated paraphrased variant
(query_expansion.py), then fuses every variant's *already-fused* results
together via the same reciprocal_rank_fusion function — RRF applied one
level up. This is a legitimate reuse, not a hack: RRF only needs ranked
lists as input and doesn't care what produced them, so feeding it
already-fused rankings to fuse again is standard practice for combining
multiple retrieval "runs."

Returns both the final merged candidates and a per-variant breakdown
(which phrasing found how many candidates), so the retrieval trace —
surfaced all the way to the demo UI — can show exactly what was
searched, not just the final blended result.
"""

from app.core.config import settings
from app.services.ingestion.embedder import embed_texts
from app.services.retrieval.hybrid_search import hybrid_search, reciprocal_rank_fusion
from app.services.retrieval.query_expansion import expand_query


async def expanded_hybrid_search(db, question: str, top_k: int) -> dict:
    if settings.query_expansion_enabled:
        queries = await expand_query(question, settings.query_expansion_variants)
    else:
        queries = [question]

    per_variant = []
    ranked_lists = []

    for query_text in queries:
        [query_vector] = await embed_texts([query_text])
        candidates = await hybrid_search(db, query_text, query_vector, top_k=top_k)
        per_variant.append({"query": query_text, "candidates_found": len(candidates)})
        ranked_lists.append(candidates)

    if len(ranked_lists) == 1:
        fused = ranked_lists[0]
    else:
        fused = reciprocal_rank_fusion(ranked_lists, k=settings.rrf_k)

    return {
        "candidates": fused[:top_k],
        "queries_used": queries,
        "per_variant": per_variant,
    }

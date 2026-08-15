"""
Hybrid retrieval: run vector search and keyword search independently,
then merge their two ranked lists into one via Reciprocal Rank Fusion
(RRF).

Concept note — why RRF, not just averaging the scores:
Vector search returns cosine similarity (roughly 0 to 1). Keyword search
returns a ts_rank score (an unbounded, differently-scaled number). You
cannot meaningfully average two scores that live on different scales
without some ad-hoc normalization — which one to use, how to weight it,
is a bunch of extra tuning surface. RRF sidesteps this entirely: it
ignores the raw scores and only looks at *rank position* within each
list. A chunk ranked #1 by vector search and #3 by keyword search gets:

    score = 1/(k + 1) + 1/(k + 3)

A chunk that appears in only one list still gets a score (from that list
alone), so it's not excluded — it just scores lower than something that
both retrieval methods agreed was relevant. `k` (default 60, the
standard value from the original RRF paper) softens the impact of exact
rank position — without it, the gap between rank 1 and rank 2 would
dominate the fused score disproportionately.
"""

from app.core.config import settings
from app.services.retrieval.vector_search import vector_search
from app.services.retrieval.keyword_search import keyword_search


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]], k: int = 60
) -> list[dict]:
    """
    ranked_lists: multiple already-ranked lists of chunk dicts, each
    containing a 'chunk_id' key. Order within each list matters — index 0
    is treated as rank 1.
    """
    fused: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for position, item in enumerate(ranked_list, start=1):
            chunk_id = item["chunk_id"]
            if chunk_id not in fused:
                fused[chunk_id] = {"score": 0.0, "chunk": item}
            fused[chunk_id]["score"] += 1.0 / (k + position)

    ordered = sorted(fused.values(), key=lambda entry: entry["score"], reverse=True)

    results = []
    for entry in ordered:
        chunk = dict(entry["chunk"])
        chunk["fusion_score"] = round(entry["score"], 5)
        results.append(chunk)
    return results


async def hybrid_search(db, query_text: str, query_embedding: list[float], top_k: int) -> list[dict]:
    # Overfetch each individual list before fusing — a chunk that's
    # borderline-relevant in both vector and keyword search can still
    # rank highly once fused, even if it wasn't in either list's top few.
    overfetch = top_k * 2

    vector_results = await vector_search(db, query_embedding, top_k=overfetch)
    keyword_results = await keyword_search(db, query_text, top_k=overfetch)

    fused = reciprocal_rank_fusion(
        [vector_results, keyword_results], k=settings.rrf_k
    )
    return fused[:top_k]

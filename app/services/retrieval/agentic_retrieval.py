"""
The agentic retrieval loop.

Flow, per attempt:
  1. Embed the current query (original, or a reformulated one from a
     previous attempt).
  2. hybrid_search (Day 4) for candidates.
  3. rerank (this file's sibling, reranker.py) to get a precise top-N.
  4. Check whether the best reranked result is confident enough.
     - Yes -> stop, return these results.
     - No, and attempts remain -> reformulate the query, try again.
     - No, and attempts exhausted -> stop anyway, return what we have.

The critical safety property: this loop CANNOT run forever. Every
agentic system needs an explicit stop condition that isn't "keep trying
until it works," because for some queries, no phrasing is going to find
evidence that doesn't exist in the corpus — an agent without a hard cap
here would burn API calls and time indefinitely on exactly those queries.
`max_retrieval_attempts` (default 2, i.e. one retry) is that cap.

Every attempt is logged and returned alongside the final chunks — not
just written to server logs — so both the API caller and a human
debugging a specific query can see exactly what the system tried, in
what order, and why it stopped. This is deliberately visible, not hidden
behind a "trust the black box" response.
"""

from app.core.config import settings
from app.services.ingestion.embedder import embed_texts
from app.services.retrieval.hybrid_search import hybrid_search
from app.services.retrieval.reranker import rerank
from app.services.retrieval.query_reformulation import reformulate_query


def _should_stop(top_score: float | None, attempt: int, max_attempts: int, threshold: float) -> bool:
    """Pure decision logic, deliberately separated from the I/O-heavy
    loop below so it can be unit tested without a database, an embedding
    model, or an LLM call."""
    if top_score is not None and top_score >= threshold:
        return True
    if attempt >= max_attempts:
        return True
    return False


def _summarize_candidates(candidates: list[dict]) -> list[dict]:
    """Compact summary of the pre-rerank hybrid-search candidate pool —
    filename, chunk index, and fusion score, not full chunk content
    (that's already visible in the final `sources`, no need to duplicate
    it here). Added after a real debugging session where a chunk's
    absence from the final results was ambiguous: was it never retrieved
    by hybrid search at all, or was it retrieved but reranked out? Those
    are different problems with different fixes, and the old attempts
    log (candidate *count* only) couldn't distinguish between them."""
    return [
        {
            "filename": c["filename"],
            "chunk_index": c["chunk_index"],
            "fusion_score": c.get("fusion_score"),
        }
        for c in candidates
    ]


async def agentic_retrieve(db, question: str) -> dict:
    current_query = question
    reranked: list[dict] = []
    attempts_log: list[dict] = []

    for attempt in range(1, settings.max_retrieval_attempts + 1):
        [query_vector] = await embed_texts([current_query])
        candidates = await hybrid_search(
            db, current_query, query_vector, top_k=settings.retrieval_top_k
        )
        reranked = await rerank(current_query, candidates, top_n=settings.rerank_top_n)

        top_score = reranked[0]["rerank_score"] if reranked else None
        attempts_log.append(
            {
                "attempt": attempt,
                "query": current_query,
                "candidates_found": len(candidates),
                "top_rerank_score": top_score,
                # Pre-rerank pool — lets you distinguish "never retrieved
                # by hybrid search" from "retrieved but reranked out."
                "pre_rerank_candidates": _summarize_candidates(candidates),
            }
        )

        if _should_stop(top_score, attempt, settings.max_retrieval_attempts, settings.min_rerank_score):
            break

        current_query = await reformulate_query(question)

    return {"chunks": reranked, "attempts": attempts_log}

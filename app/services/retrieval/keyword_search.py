"""
Lexical retrieval via Postgres full-text search.

Honesty note: this is deliberately NOT calling itself "BM25." Postgres's
`ts_rank()` is a related but different ranking function — it scores based
on term frequency and lexical coverage using its own formula, not the
Okapi BM25 formula specifically. The role it plays is the same as BM25
would play (exact keyword/term matching, as opposed to semantic vector
similarity), which is why it's the right piece for the "lexical" half of
hybrid search — but claiming it's literally BM25 would be overclaiming a
detail that doesn't hold up if someone asks about it directly.

Why this matters at all (the concept from Phase 3): vector search finds
"semantically similar" content — it can completely miss an exact product
code, a name, or an ID, because those don't carry rich semantic meaning
the way a full sentence does. Full-text search catches exactly that case:
literal term matches, regardless of surrounding meaning. Neither is
strictly better — that's the entire justification for fusing both
(hybrid_search.py) rather than picking one.

`to_tsvector`/`plainto_tsquery` are computed on the fly here rather than
stored in an indexed column. That's a real, documented tradeoff: simpler
schema, no migration needed, but slower at large scale than a persisted,
GIN-indexed tsvector column would be. Fine for a project at this stage;
worth flagging as a known scaling limitation, not a hidden one.
"""

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Chunk, Document


async def keyword_search(db: AsyncSession, query_text: str, top_k: int) -> list[dict]:
    tsvector = func.to_tsvector("english", Chunk.content)
    tsquery = func.plainto_tsquery("english", query_text)
    rank = func.ts_rank(tsvector, tsquery).label("rank")

    stmt = (
        select(Chunk, Document.filename, rank)
        .join(Document, Chunk.document_id == Document.id)
        .where(tsvector.op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(top_k)
    )

    result = await db.execute(stmt)
    rows = result.all()

    return [
        {
            "chunk_id": str(chunk.id),
            "document_id": str(chunk.document_id),
            "filename": filename,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "keyword_score": round(float(rank_value), 4),
        }
        for chunk, filename, rank_value in rows
    ]

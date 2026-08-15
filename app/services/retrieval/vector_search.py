"""
Baseline vector retrieval.

Concept note: pgvector adds a `<=>` operator to Postgres meaning "cosine
distance between these two vectors." The pgvector-python/SQLAlchemy
integration exposes this as `.cosine_distance()` on a Vector column, so
`Chunk.embedding.cosine_distance(query_vector)` compiles down to that
`<=>` operator in the actual SQL. Ordering by it ascending gives you the
chunks whose embeddings are *closest* to the query's embedding — i.e.
"most semantically similar," not "contains the same keywords."

Distance vs. similarity: pgvector's cosine_distance returns a value where
0 = identical direction (maximally similar), 2 = opposite direction. We
convert to a more intuitive "similarity score" via `1 - distance` before
returning it — mostly for human readability when inspecting results, not
because the math requires it.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Chunk, Document


async def vector_search(
    db: AsyncSession, query_embedding: list[float], top_k: int
) -> list[dict]:
    distance = Chunk.embedding.cosine_distance(query_embedding).label("distance")

    stmt = (
        select(Chunk, Document.filename, distance)
        .join(Document, Chunk.document_id == Document.id)
        .order_by(distance)
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
            "similarity_score": round(1 - dist, 4),
        }
        for chunk, filename, dist in rows
    ]

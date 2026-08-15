"""
The baseline RAG loop, end to end:

  question
    -> embed (same model as ingestion — this MUST match, otherwise the
       query vector and the stored chunk vectors live in incompatible
       spaces and "nearest neighbor" becomes meaningless)
    -> vector_search (nearest chunks in pgvector)
    -> generate_answer (LLM, grounded in those chunks only)
    -> answer + sources

This is intentionally the simplest version that could work — no hybrid
search, no reranking, no agentic retry, no citation verification. Those
come Days 4-9. Today's job is: prove the core loop produces a real,
grounded answer at all. Everything after this is quality improvements on
top of a working foundation.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.services.ingestion.embedder import embed_texts
from app.services.retrieval.vector_search import vector_search
from app.services.generation.answer import generate_answer, MissingAPIKeyError

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)


@router.post("")
async def query(request: QueryRequest, db: AsyncSession = Depends(get_db)):
    top_k = request.top_k or settings.retrieval_top_k

    # Same embedding model as ingestion, on purpose — see module docstring.
    [query_vector] = await embed_texts([request.question])

    chunks = await vector_search(db, query_vector, top_k=top_k)

    try:
        answer = await generate_answer(request.question, chunks)
    except MissingAPIKeyError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return {
        "question": request.question,
        "answer": answer,
        "sources": [
            {
                "filename": c["filename"],
                "chunk_index": c["chunk_index"],
                "similarity_score": c["similarity_score"],
                "content_preview": c["content"][:200] + ("..." if len(c["content"]) > 200 else ""),
            }
            for c in chunks
        ],
    }

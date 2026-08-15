"""
The RAG loop, end to end, as of Day 5:

  question
    -> agentic_retrieve:
         embed -> hybrid_search -> rerank
         -> confident enough? stop : reformulate query and retry
            (capped at settings.max_retrieval_attempts)
    -> generate_answer (LLM, grounded in the final reranked chunks only)
    -> answer + sources + retrieval_debug (what the agent actually tried)

Citation verification and the human-review queue are the remaining v2
items (see project README). Everything through Day 5 is now genuinely
agentic, not just a fixed pipeline — the system decides whether to retry
based on the outcome of its own previous step.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.retrieval.agentic_retrieval import agentic_retrieve
from app.services.generation.answer import generate_answer, MissingAPIKeyError

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


@router.post("")
async def query(request: QueryRequest, db: AsyncSession = Depends(get_db)):
    retrieval_result = await agentic_retrieve(db, request.question)
    chunks = retrieval_result["chunks"]

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
                "rerank_score": c.get("rerank_score"),
                "content_preview": c["content"][:200] + ("..." if len(c["content"]) > 200 else ""),
            }
            for c in chunks
        ],
        "retrieval_debug": {
            "attempts": retrieval_result["attempts"],
        },
    }

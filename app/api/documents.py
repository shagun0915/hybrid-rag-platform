"""
Document API.

Three endpoints, deliberately small in scope for Day 2:
- POST /documents/upload       -> ingest a file end-to-end
- GET  /documents              -> list what's been uploaded, with status
- GET  /documents/{id}/chunks  -> inspect how a document got chunked
                                   (this is your main debugging tool —
                                   "did chunking do something sane?" —
                                   long before there's any retrieval UI)
"""

import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.document import Document, Chunk
from app.services.ingestion.pipeline import ingest_document, FileTooLargeError
from app.services.ingestion.parser import UnsupportedFileType, EmptyDocumentError

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...), db: AsyncSession = Depends(get_db)
):
    content = await file.read()

    try:
        document = await ingest_document(
            db, filename=file.filename, content_type=file.content_type, content=content
        )
    except (UnsupportedFileType, EmptyDocumentError, FileTooLargeError) as exc:
        # Client's fault (bad input) -> 400, not 500. Distinguishing these
        # matters: a 500 tells the caller "we broke," a 400 tells them
        # "you sent something we can't process" — different fixes.
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "document_id": str(document.id),
        "filename": document.filename,
        "status": document.status,
        "chunk_count": document.chunk_count,
    }


@router.get("")
async def list_documents(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).order_by(Document.uploaded_at.desc()))
    documents = result.scalars().all()
    return [
        {
            "document_id": str(d.id),
            "filename": d.filename,
            "status": d.status,
            "chunk_count": d.chunk_count,
            "uploaded_at": d.uploaded_at,
            "error_message": d.error_message,
        }
        for d in documents
    ]


@router.get("/{document_id}/chunks")
async def get_document_chunks(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    doc_result = await db.execute(select(Document).where(Document.id == document_id))
    document = doc_result.scalar_one_or_none()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    chunk_result = await db.execute(
        select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
    )
    chunks = chunk_result.scalars().all()

    return {
        "document_id": str(document.id),
        "filename": document.filename,
        "chunk_count": len(chunks),
        "chunks": [
            {
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                # Preview only — dumping the full embedding vector (384
                # floats) into a JSON response is noise, not signal, for
                # a human debugging chunking quality.
                "content_preview": c.content[:300] + ("..." if len(c.content) > 300 else ""),
                "embedding_dimensions": len(c.embedding),
            }
            for c in chunks
        ],
    }

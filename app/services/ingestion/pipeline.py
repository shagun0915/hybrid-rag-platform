"""
Orchestrates the full ingestion flow for one uploaded document:
parse -> chunk -> embed -> store, updating Document.status along the way
so the API always has an honest answer to "what happened to my upload?"

Kept synchronous-in-request for Day 2 (the caller awaits the whole thing
before getting a response). For large documents or high upload volume,
this would move to a background task/queue in a later iteration — noted
here rather than silently deferred.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, Chunk, DocumentStatus
from app.services.ingestion.parser import extract_text
from app.services.ingestion.chunking_strategy import get_chunks
from app.services.ingestion.embedder import embed_texts

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


class FileTooLargeError(Exception):
    pass


async def ingest_document(
    db: AsyncSession, filename: str, content_type: str, content: bytes
) -> Document:
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise FileTooLargeError(
            f"File exceeds {MAX_FILE_SIZE_BYTES // (1024*1024)}MB limit."
        )

    document = Document(
        filename=filename,
        content_type=content_type or "application/octet-stream",
        status=DocumentStatus.PROCESSING,
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    try:
        text = extract_text(filename, content)
        pieces = await get_chunks(text)
        vectors = await embed_texts(pieces)

        for index, (piece, vector) in enumerate(zip(pieces, vectors)):
            db.add(
                Chunk(
                    document_id=document.id,
                    chunk_index=index,
                    content=piece,
                    token_count=len(piece.split()),
                    embedding=vector,
                )
            )

        document.status = DocumentStatus.READY
        document.chunk_count = len(pieces)
        await db.commit()
        await db.refresh(document)

    except Exception as exc:
        document.status = DocumentStatus.FAILED
        document.error_message = str(exc)[:500]
        await db.commit()
        raise

    return document

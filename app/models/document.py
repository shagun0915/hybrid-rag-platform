"""
Two tables, deliberately kept separate:

Document — one row per uploaded file. Tracks status through the pipeline
(uploaded -> processing -> ready / failed) so the API can always answer
"what happened to my upload?" without guessing.

Chunk — one row per chunk of a document, each with its own embedding
vector. This is the table retrieval will actually query in Day 3+.

Why separate tables instead of one? A document is metadata (filename,
status, when it was uploaded). A chunk is a unit of retrievable content.
Keeping them separate means you can re-chunk a document (different chunk
size, better splitter) without touching document-level metadata, and a
single query on `chunks` doesn't have to drag along document-level
columns it doesn't need.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, Text, Integer, DateTime, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from app.models.base import Base
from app.core.config import settings

import enum


class DocumentStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(String(512))
    content_type: Mapped[str] = mapped_column(String(128))
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus), default=DocumentStatus.UPLOADED
    )
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE")
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)

    # This is the pgvector column. `settings.embedding_dimension` (384 by
    # default) MUST match the actual output size of whatever embedding
    # model generates the vectors — a mismatch fails loudly at insert
    # time, which is the correct behavior (fail fast, not silently store
    # garbage).
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dimension))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")

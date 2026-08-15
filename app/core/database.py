"""
Database engine, session management, and startup initialization.

Concept note (Day 1): what pgvector actually is.
Postgres normally stores rows — text, numbers, dates. pgvector is an
extension that adds a new column TYPE called `vector(N)`, plus index types
(HNSW, IVFFlat) that let Postgres answer "which rows are closest to this
vector?" efficiently, instead of scanning every row and computing distance
by hand. That's it. It's not a separate database — it's Postgres that has
learned one new trick: nearest-neighbor search on vectors, alongside every
normal SQL feature you already know (joins, transactions, indexes on other
columns). That's *why* it's a good choice for an enterprise-style RAG
system: your document metadata (permissions, timestamps, source) and your
embeddings live in the same transactional database, instead of syncing two
separate systems.
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import text

from app.core.config import settings

engine = create_async_engine(settings.database_url, echo=False, future=True)

AsyncSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)


async def get_db() -> AsyncSession:
    """FastAPI dependency — yields a DB session per-request, closes it after."""
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    """
    Runs once at startup.
    Enables the pgvector extension and creates tables if they don't exist.
    Day 1: this just enables the extension so we can prove connectivity.
    Day 2: this will also create the documents/chunks tables.
    """
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


async def check_db_connection() -> bool:
    """Used by the /health endpoint. Returns False instead of raising,
    so a DB outage degrades the health check rather than crashing it —
    the same 'don't let one failure cascade' instinct from your
    production support experience at Visa."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False

"""
Dispatches to the configured chunking strategy. Same swap pattern as
llm_client.py's LLM_PROVIDER dispatch — the caller (pipeline.py) doesn't
know or care which strategy is active, just "give me chunks back."
"""

from app.core.config import settings
from app.services.ingestion.chunker import chunk_text
from app.services.ingestion.semantic_chunker import semantic_chunk_text


async def get_chunks(text: str) -> list[str]:
    if settings.chunking_strategy == "semantic":
        return await semantic_chunk_text(text)
    elif settings.chunking_strategy == "fixed":
        return chunk_text(text)
    else:
        raise ValueError(
            f"Unknown CHUNKING_STRATEGY '{settings.chunking_strategy}'. "
            "Valid options: 'semantic', 'fixed'."
        )

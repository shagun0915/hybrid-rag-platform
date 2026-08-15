"""
Generation: given a question and retrieved chunks, ask an LLM to answer
using ONLY that context.

The "grounding" instruction in SYSTEM_PROMPT is the single most important
part of this file. Without it, the model blends in facts from its own
training data alongside your documents, and you'd have no way to tell
which is which — defeating the purpose of RAG in the first place.

Provider dispatch (Ollama vs Anthropic) now lives in llm_client.py,
shared with query_reformulation.py — this file only owns the RAG-specific
prompt and context-building logic.
"""

from app.services.generation.llm_client import call_llm, MissingAPIKeyError, GenerationError  # noqa: F401 — re-exported for callers

SYSTEM_PROMPT = """You are a precise, factual assistant answering questions using ONLY the provided context excerpts.

Rules:
- Answer using only information found in the context below.
- If the context does not contain enough information to answer, say so plainly — do not guess or use outside knowledge.
- When you use a fact from a specific excerpt, mention which excerpt number it came from, like "(Excerpt 2)".
- Keep answers concise and directly responsive to the question."""


def _build_context_block(chunks: list[dict]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[Excerpt {i} — source: {chunk['filename']}]\n{chunk['content']}"
        )
    return "\n\n".join(blocks)


async def generate_answer(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return (
            "I don't have any relevant documents to answer this question. "
            "Try uploading a document first via /documents/upload."
        )

    context_block = _build_context_block(chunks)
    user_message = f"Context:\n\n{context_block}\n\nQuestion: {question}"

    return await call_llm(SYSTEM_PROMPT, user_message)

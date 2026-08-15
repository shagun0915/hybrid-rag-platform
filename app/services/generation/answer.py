"""
Generation: given a question and retrieved chunks, ask an LLM to answer
using ONLY that context.

Concept note (unchanged from the original design): the "grounding"
instruction below is the single most important part of this file. Without
it, the model blends in facts from its own training data alongside your
documents, and you'd have no way to tell which is which — defeating the
purpose of RAG in the first place.

Provider design: `generate_answer()` is the only function the rest of the
app calls. It dispatches to either Ollama (local, free, no key needed) or
Anthropic (cloud, paid, needs ANTHROPIC_API_KEY) based on
`settings.llm_provider` — the API endpoint, the prompt, and the caller
don't need to know or care which one is actually running. This is the
same "don't hard-code, keep it swappable" principle as chunk size and
retrieval K elsewhere in the project — here it also happens to solve a
very practical problem: free local development, with a one-line switch
to real Claude when you want it.
"""

import httpx

from app.core.config import settings

SYSTEM_PROMPT = """You are a precise, factual assistant answering questions using ONLY the provided context excerpts.

Rules:
- Answer using only information found in the context below.
- If the context does not contain enough information to answer, say so plainly — do not guess or use outside knowledge.
- When you use a fact from a specific excerpt, mention which excerpt number it came from, like "(Excerpt 2)".
- Keep answers concise and directly responsive to the question."""


class MissingAPIKeyError(Exception):
    pass


class GenerationError(RuntimeError):
    """Base class for provider failures. Subclasses RuntimeError so
    existing `except RuntimeError` handling (in api/query.py) still
    catches it without needing to know which provider is active."""
    pass


def _build_context_block(chunks: list[dict]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[Excerpt {i} — source: {chunk['filename']}]\n{chunk['content']}"
        )
    return "\n\n".join(blocks)


async def _generate_with_ollama(user_message: str) -> str:
    url = f"{settings.ollama_base_url}/api/chat"
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
    except httpx.ConnectError as exc:
        raise GenerationError(
            f"Could not reach Ollama at {settings.ollama_base_url}. "
            "Is the Ollama app running on your Mac? (check for the llama "
            "icon in your menu bar)"
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise GenerationError(
            f"Ollama returned an error: {exc.response.status_code} — "
            f"is the model '{settings.ollama_model}' pulled? "
            f"Try: ollama pull {settings.ollama_model}"
        ) from exc

    data = response.json()
    return data["message"]["content"]


async def _generate_with_anthropic(user_message: str) -> str:
    if not settings.anthropic_api_key:
        raise MissingAPIKeyError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file, or "
            "set LLM_PROVIDER=ollama to use free local generation instead."
        )

    try:
        from anthropic import AsyncAnthropic, APIError
    except ImportError as exc:
        raise GenerationError(
            "The 'anthropic' package isn't installed. Run: pip install anthropic"
        ) from exc

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    try:
        response = await client.messages.create(
            model=settings.anthropic_model,
            max_tokens=settings.llm_max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except APIError as exc:
        raise GenerationError(f"Anthropic API call failed: {exc}") from exc

    return response.content[0].text


async def generate_answer(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return (
            "I don't have any relevant documents to answer this question. "
            "Try uploading a document first via /documents/upload."
        )

    context_block = _build_context_block(chunks)
    user_message = f"Context:\n\n{context_block}\n\nQuestion: {question}"

    if settings.llm_provider == "ollama":
        return await _generate_with_ollama(user_message)
    elif settings.llm_provider == "anthropic":
        return await _generate_with_anthropic(user_message)
    else:
        raise GenerationError(
            f"Unknown LLM_PROVIDER '{settings.llm_provider}'. "
            "Valid options: 'ollama', 'anthropic'."
        )

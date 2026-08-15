"""
Low-level LLM client, shared by anything in the app that needs to call
an LLM for any reason — answer generation (answer.py) and agentic query
reformulation (retrieval/query_reformulation.py) both go through this,
rather than each having their own copy of the Ollama/Anthropic dispatch
logic.

This is the same provider-swap design as before: callers pass a system
prompt and a user message, and don't know or care whether the underlying
call goes to Ollama or Anthropic. That decoupling is what made it cheap
to add a second caller (reformulation) today — it needed zero knowledge
of provider details, just "give me text back."
"""

import httpx

from app.core.config import settings


class MissingAPIKeyError(Exception):
    pass


class GenerationError(RuntimeError):
    """Base class for LLM call failures. Subclasses RuntimeError so
    existing `except RuntimeError` handling (in api/query.py) still
    catches it without needing to know which provider is active."""
    pass


async def _call_ollama(system_prompt: str, user_message: str) -> str:
    url = f"{settings.ollama_base_url}/api/chat"
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
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


async def _call_anthropic(system_prompt: str, user_message: str) -> str:
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
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
    except APIError as exc:
        raise GenerationError(f"Anthropic API call failed: {exc}") from exc

    return response.content[0].text


async def call_llm(system_prompt: str, user_message: str) -> str:
    if settings.llm_provider == "ollama":
        return await _call_ollama(system_prompt, user_message)
    elif settings.llm_provider == "anthropic":
        return await _call_anthropic(system_prompt, user_message)
    else:
        raise GenerationError(
            f"Unknown LLM_PROVIDER '{settings.llm_provider}'. "
            "Valid options: 'ollama', 'anthropic'."
        )

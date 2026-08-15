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


async def _call_groq(system_prompt: str, user_message: str) -> str:
    """Groq — cloud, genuinely free (no credit card, not a trial), and
    the recommended provider for a public deployment, since Ollama
    can't run on typical free hosting tiers (needs several GB RAM an
    8B-parameter model doesn't have on a 512MB free instance) and
    Anthropic requires paid credits. Groq's API is OpenAI-compatible,
    so this is a plain httpx call — same pattern as Ollama, no new SDK
    dependency, matching how the rest of this file is built."""
    if not settings.groq_api_key:
        raise MissingAPIKeyError(
            "GROQ_API_KEY is not set. Get a free key (no credit card) at "
            "console.groq.com/keys, or set LLM_PROVIDER=ollama for local "
            "dev instead."
        )

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {settings.groq_api_key}"}
    payload = {
        "model": settings.groq_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": settings.llm_max_tokens,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            raise GenerationError(
                "Groq free-tier rate limit hit (30 requests/min, "
                "14,400/day). Wait a moment and try again."
            ) from exc
        raise GenerationError(
            f"Groq API call failed: {exc.response.status_code} — {exc.response.text[:200]}"
        ) from exc

    data = response.json()
    return data["choices"][0]["message"]["content"]


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
    elif settings.llm_provider == "groq":
        return await _call_groq(system_prompt, user_message)
    elif settings.llm_provider == "anthropic":
        return await _call_anthropic(system_prompt, user_message)
    else:
        raise GenerationError(
            f"Unknown LLM_PROVIDER '{settings.llm_provider}'. "
            "Valid options: 'ollama', 'groq', 'anthropic'."
        )

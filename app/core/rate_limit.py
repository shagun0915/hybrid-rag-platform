"""
Minimal per-IP rate limiting.

Deliberately not slowapi / redis / a token bucket library. The deployment
runs a single free-tier instance (see README Deployment), so an in-memory
fixed-window counter is genuinely sufficient — a distributed limiter would
be more moving parts for zero benefit here. Same "don't add a heavy
dependency for a small job" call as using fastembed instead of full
PyTorch for reranking.

What it protects: the endpoints that cost real money or CPU when abused
(`/query` makes LLM calls, `/documents/upload` runs embedding inference).
Everything is unauthenticated by design, so throttling by client IP is
the available lever.

Fixed-window, not sliding: a caller can burst up to 2x the limit across a
window boundary. Acceptable for "stop someone scripting the endpoint" —
tighten to a sliding window only if that ever turns out to matter.
"""

import time
from collections import defaultdict

from fastapi import Request, HTTPException


class _FixedWindowLimiter:
    def __init__(self) -> None:
        # key -> (window_start_epoch, count_in_window)
        self._hits: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))

    def check(self, key: str, limit: int, window_seconds: int = 60) -> None:
        now = time.time()

        # Opportunistic cleanup so the dict can't grow without bound if a
        # caller rotates through many IPs. Cheap: only when it gets large,
        # and only drops entries whose window has already expired.
        if len(self._hits) > 10_000:
            stale = [k for k, (ws, _) in self._hits.items() if now - ws >= window_seconds]
            for k in stale:
                del self._hits[k]

        window_start, count = self._hits[key]

        if now - window_start >= window_seconds:
            # New window.
            self._hits[key] = (now, 1)
            return

        if count >= limit:
            retry_after = int(window_seconds - (now - window_start)) + 1
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded ({limit}/min). Try again in ~{retry_after}s.",
                headers={"Retry-After": str(retry_after)},
            )

        self._hits[key] = (window_start, count + 1)


_limiter = _FixedWindowLimiter()


def _client_key(request: Request, bucket: str) -> str:
    # Render/Cloudflare put the real client IP in X-Forwarded-For (first
    # entry). Fall back to the socket peer for local/direct runs.
    fwd = request.headers.get("x-forwarded-for", "")
    client_ip = fwd.split(",")[0].strip() if fwd else (
        request.client.host if request.client else "unknown"
    )
    return f"{bucket}:{client_ip}"


def rate_limit(bucket: str, per_minute: int):
    """FastAPI dependency factory. Raises 429 when the caller is over the
    limit for this bucket, otherwise does nothing."""

    async def _dep(request: Request) -> None:
        _limiter.check(_client_key(request, bucket), per_minute)

    return _dep

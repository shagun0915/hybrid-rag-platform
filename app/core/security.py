"""
Two small security pieces: an optional admin gate, and response headers.

Admin gate — `require_admin_token` is a FastAPI dependency. If
`settings.admin_token` is unset (the default), it's a no-op, so local dev
and the offline test suite behave exactly as before. If it's set (in the
deployment), the guarded endpoint requires a matching `X-Admin-Token`
header. Used only on DELETE /documents/{id} — the one irreversible
operation — not on upload or query, which stay open for the public demo.

Security headers — added to every response via middleware. The CSP is
written to allow what the two HTML surfaces this app serves actually
need: `/ui` (Google Fonts + its own inline <style>/<script>) and `/docs`
(Swagger UI from jsdelivr + its inline bootstrap). Anything stricter
breaks one of those pages.
"""

import hmac

from fastapi import Header, HTTPException

from app.core.config import settings

_CSP = (
    "default-src 'self'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net; "
    "font-src 'self' https://fonts.gstatic.com; "
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "img-src 'self' data: https://fastapi.tiangolo.com https://cdn.jsdelivr.net; "
    "connect-src 'self'; "
    "worker-src 'self' blob:; "
    "frame-ancestors 'none'; "
    "base-uri 'self'"
)

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": _CSP,
}


async def security_headers_middleware(request, call_next):
    response = await call_next(request)
    for name, value in SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    return response


async def require_admin_token(x_admin_token: str | None = Header(default=None)) -> None:
    if not settings.admin_token:
        return  # gate disabled (local dev / tests / open demo)
    if not x_admin_token or not hmac.compare_digest(x_admin_token, settings.admin_token):
        raise HTTPException(status_code=401, detail="Valid X-Admin-Token header required.")

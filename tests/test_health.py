"""
Day 1 test: prove the app boots and the liveness endpoint responds
without needing a database at all. (The readiness endpoint needs a real
DB connection, so it's exercised via docker-compose, not in this
lightweight unit test.)
"""

from httpx import AsyncClient, ASGITransport
import pytest

from app.main import app


@pytest.mark.asyncio
async def test_liveness():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

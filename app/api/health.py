"""
Health endpoints.

Two, on purpose:
- /health/live  -> "is the process up at all?" (never touches the DB)
- /health/ready -> "is the process actually able to serve traffic?"
                   (checks the DB)

This liveness/readiness split is a standard production pattern — you want
a container orchestrator (or just you, debugging) to be able to tell the
difference between "the app crashed" and "the app is up but the database
is unreachable." One endpoint can't tell you both.
"""

from fastapi import APIRouter

from app.core.database import check_db_connection

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def liveness():
    return {"status": "ok"}


@router.get("/ready")
async def readiness():
    db_ok = await check_db_connection()
    return {
        "status": "ok" if db_ok else "degraded",
        "database": "connected" if db_ok else "unreachable",
    }

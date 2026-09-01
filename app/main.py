"""
App entrypoint.

Day 1 scope: FastAPI app + health checks + DB connectivity, running in
Docker. Nothing AI-related yet on purpose — a foundation you can't trust
makes every later phase harder to debug, because you can never be sure if
a bug is in your retrieval logic or in the plumbing underneath it.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.database import init_db
from app.api import health, documents, query

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the app starts (not per-request).
    #
    # init_db() opens a real DB connection. If the database is
    # unreachable at boot (Supabase free tier auto-pauses after
    # inactivity; a hosted Postgres may still be waking), letting this
    # raise takes the entire app down — Render then never passes a health
    # check and crash-loops on the "waking up" page, so even /health/live
    # (which never touches the DB) is unreachable. Instead: log it and
    # start anyway. /health/ready will report "database: unreachable",
    # and the first request that needs the DB once it's back will
    # reconnect via the pool.
    try:
        await init_db()
    except Exception:
        logger.exception(
            "init_db() failed at startup — serving without verified DB "
            "connectivity; check /health/ready and the database status"
        )
    yield
    # (nothing to clean up yet — connection pool disposal happens
    # automatically when the process exits)


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)


@app.get("/")
async def root():
    return {
        "app": settings.app_name,
        "environment": settings.environment,
        "docs": "/docs",
        "demo_ui": "/ui",
    }


# Demo UI — served directly by this same API (no separate frontend
# service, no CORS to configure). Mounted last so it doesn't shadow any
# API route above it. Visiting /ui serves app/static/index.html.
app.mount("/ui", StaticFiles(directory="app/static", html=True), name="ui")

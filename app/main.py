"""
App entrypoint.

Day 1 scope: FastAPI app + health checks + DB connectivity, running in
Docker. Nothing AI-related yet on purpose — a foundation you can't trust
makes every later phase harder to debug, because you can never be sure if
a bug is in your retrieval logic or in the plumbing underneath it.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.core.database import init_db
from app.api import health


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the app starts (not per-request)
    await init_db()
    yield
    # (nothing to clean up yet — connection pool disposal happens
    # automatically when the process exits)


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.include_router(health.router)


@app.get("/")
async def root():
    return {
        "app": settings.app_name,
        "environment": settings.environment,
        "docs": "/docs",
    }

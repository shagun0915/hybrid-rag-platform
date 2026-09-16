"""
Caps how many CPU/memory-heavy requests run their pipeline at once.

Why this exists — a real Render "exceeded its memory limit" restart, not
a hypothetical: the free-tier instance has ~512MB RAM and effectively one
CPU core (0.1 vCPU). The per-IP rate limits in `rate_limit.py` cap
requests *per minute*, but `/query` alone can take 20-40s on this tier —
several different visitors' requests easily overlap within that window
even while each individually stays under the per-minute cap. Each
concurrent `/query` holds its own embedded query vector, the hybrid-
search candidate pool (chunk content included), and cross-encoder
inference buffers in memory at the same time; a handful of overlapping
requests is enough to exceed 512MB. Uploads are the same shape — every
chunk gets embedded before the response returns.

A semaphore, not a hard reject: an extra request *waits* for its turn
rather than getting a 503. The instance is already documented as slow
(~34s/query) on this tier, so a longer wait under a burst reads the same
as "slow," not as a new failure mode.

Sized at 1 by default, deliberately: with one real CPU core, a second
concurrent CPU-bound inference call doesn't add throughput anyway — the
OS just time-slices between threads competing for the same core, which
is worse for both requests' wall-clock time than running them one after
another. Raise MAX_CONCURRENT_QUERIES/MAX_CONCURRENT_UPLOADS only if the
instance is upgraded to more than one core.
"""

import asyncio
from contextlib import asynccontextmanager

from app.core.config import settings

_semaphores: dict[str, asyncio.Semaphore] = {}


def _get_semaphore(bucket: str, limit: int) -> asyncio.Semaphore:
    sem = _semaphores.get(bucket)
    if sem is None:
        sem = asyncio.Semaphore(limit)
        _semaphores[bucket] = sem
    return sem


@asynccontextmanager
async def limit_concurrency(bucket: str, limit: int):
    async with _get_semaphore(bucket, limit):
        yield

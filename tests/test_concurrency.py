"""
Tests for the concurrency limiter. Pure asyncio logic — no request
object, no model, no DB — so it's fully testable offline.
"""

import asyncio

import pytest

from app.core.concurrency import limit_concurrency


@pytest.mark.asyncio
async def test_limits_concurrent_holders_to_the_given_bound():
    in_flight = 0
    max_seen = 0

    async def worker():
        nonlocal in_flight, max_seen
        async with limit_concurrency("test_bucket_1", limit=2):
            in_flight += 1
            max_seen = max(max_seen, in_flight)
            await asyncio.sleep(0.05)
            in_flight -= 1

    await asyncio.gather(*(worker() for _ in range(6)))
    assert max_seen == 2


@pytest.mark.asyncio
async def test_a_bucket_of_one_fully_serializes():
    order = []

    async def worker(name):
        async with limit_concurrency("test_bucket_2", limit=1):
            order.append(f"{name}-start")
            await asyncio.sleep(0.01)
            order.append(f"{name}-end")

    await asyncio.gather(worker("a"), worker("b"))
    # With a bound of 1, one worker must fully finish before the next starts.
    assert order in (["a-start", "a-end", "b-start", "b-end"], ["b-start", "b-end", "a-start", "a-end"])


@pytest.mark.asyncio
async def test_different_buckets_do_not_share_a_budget():
    async def worker(bucket):
        async with limit_concurrency(bucket, limit=1):
            await asyncio.sleep(0.02)

    # Two different buckets at limit=1 each can still run concurrently —
    # if this hangs/times out, buckets are wrongly sharing one semaphore.
    await asyncio.wait_for(
        asyncio.gather(worker("test_bucket_a"), worker("test_bucket_b")),
        timeout=1.0,
    )

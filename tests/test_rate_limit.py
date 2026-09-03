"""
Tests for the in-memory fixed-window rate limiter. `_FixedWindowLimiter`
is pure logic — no request object, no clock dependency beyond `time.time`
which the tests drive with a tiny real sleep — so it's fully testable
offline, same pattern as the rest of the suite.
"""

import time

import pytest
from fastapi import HTTPException

from app.core.rate_limit import _FixedWindowLimiter


def test_allows_up_to_limit_then_blocks():
    lim = _FixedWindowLimiter()
    for _ in range(3):
        lim.check("k", limit=3, window_seconds=60)  # no raise

    with pytest.raises(HTTPException) as exc:
        lim.check("k", limit=3, window_seconds=60)
    assert exc.value.status_code == 429
    assert "Retry-After" in exc.value.headers


def test_separate_keys_have_separate_budgets():
    lim = _FixedWindowLimiter()
    lim.check("a", limit=1, window_seconds=60)
    lim.check("b", limit=1, window_seconds=60)  # different key, still fine

    with pytest.raises(HTTPException):
        lim.check("a", limit=1, window_seconds=60)


def test_window_resets_after_it_expires():
    lim = _FixedWindowLimiter()
    lim.check("k", limit=1, window_seconds=0.05)
    with pytest.raises(HTTPException):
        lim.check("k", limit=1, window_seconds=0.05)

    time.sleep(0.06)
    lim.check("k", limit=1, window_seconds=0.05)  # new window, allowed again


def test_expired_entries_are_purged_when_map_grows_large():
    lim = _FixedWindowLimiter()
    # Seed >10k stale entries in an already-expired window.
    stale_start = time.time() - 3600
    for i in range(10_050):
        lim._hits[f"old-{i}"] = (stale_start, 1)

    lim.check("fresh", limit=5, window_seconds=60)  # triggers the cleanup

    assert not any(key.startswith("old-") for key in lim._hits)
    assert "fresh" in lim._hits

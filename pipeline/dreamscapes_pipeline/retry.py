"""Retry paid API calls on rate limits and server errors, with exponential backoff.

The SDKs' own retries are switched off so that this is the only place that decides how often
and how long to wait. Image requests are slow and rate limits on image models are tight, so the
backoff is longer than the SDK defaults, and a ``Retry-After`` hint from the server wins.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from .providers.base import ProviderError

MAX_ATTEMPTS = 5
BASE_DELAY_S = 2.0
MAX_DELAY_S = 60.0


def backoff_delay(attempt: int, retry_after: float | None = None) -> float:
    """Seconds to wait after failed attempt number ``attempt`` (1-based)."""
    if retry_after is not None:
        return min(retry_after, MAX_DELAY_S)
    return min(BASE_DELAY_S * 2 ** (attempt - 1), MAX_DELAY_S)


def with_retries[T](
    call: Callable[[], T],
    *,
    what: str,
    attempts: int = MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    echo: Callable[[str], None] = print,
) -> T:
    """Run ``call``; on a retryable `ProviderError` wait and try again, up to ``attempts``."""
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except ProviderError as error:
            if not error.retryable or attempt == attempts:
                raise
            delay = backoff_delay(attempt, error.retry_after)
            echo(f"  {what}: {error} (attempt {attempt}/{attempts}); retrying in {delay:.0f}s")
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover

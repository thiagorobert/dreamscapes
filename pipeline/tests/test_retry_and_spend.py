from __future__ import annotations

import pytest

from dreamscapes_pipeline import genlog
from dreamscapes_pipeline.providers import ProviderError
from dreamscapes_pipeline.retry import MAX_DELAY_S, backoff_delay, with_retries
from dreamscapes_pipeline.spend import (
    DEFAULT_LIMIT_USD,
    SpendGuard,
    SpendLimitExceeded,
    limit_from_env,
)
from tests.fakes import offline  # noqa: F401


def test_backoff_doubles_up_to_a_cap_and_respects_retry_after():
    assert [backoff_delay(n) for n in (1, 2, 3)] == [2.0, 4.0, 8.0]
    assert backoff_delay(20) == MAX_DELAY_S
    assert backoff_delay(1, retry_after=13) == 13
    assert backoff_delay(1, retry_after=600) == MAX_DELAY_S


def test_retries_retryable_errors_then_succeeds():
    sleeps, messages, calls = [], [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ProviderError("429", retryable=True, retry_after=1.5 if len(calls) == 2 else None)
        return "ok"

    result = with_retries(flaky, what="thing", sleep=sleeps.append, echo=messages.append)
    assert result == "ok"
    assert sleeps == [2.0, 1.5]
    assert len(messages) == 2 and "retrying" in messages[0]


def test_does_not_retry_permanent_errors_and_gives_up_eventually():
    sleeps = []

    def bad():
        raise ProviderError("400", retryable=False)

    with pytest.raises(ProviderError, match="400"):
        with_retries(bad, what="x", sleep=sleeps.append, echo=lambda _: None)
    assert sleeps == []

    def busy():
        raise ProviderError("503", retryable=True)

    with pytest.raises(ProviderError, match="503"):
        with_retries(busy, what="x", attempts=3, sleep=sleeps.append, echo=lambda _: None)
    assert sleeps == [2.0, 4.0]


@pytest.mark.parametrize(
    ("value", "expected"), [(None, DEFAULT_LIMIT_USD), ("", 15.0), ("  ", 15.0), ("2.5", 2.5)]
)
def test_limit_from_env_treats_empty_as_unset(value, expected):
    env = {} if value is None else {"DSP_MAX_SPEND_USD": value}
    assert limit_from_env(env) == expected


@pytest.mark.parametrize("value", ["lots", "-1"])
def test_limit_from_env_rejects_nonsense(value):
    with pytest.raises(ValueError, match="DSP_MAX_SPEND_USD"):
        limit_from_env({"DSP_MAX_SPEND_USD": value})


def test_guard_refuses_plans_and_calls_over_the_limit():
    guard = SpendGuard(limit=1.0)
    assert "estimated spend: $0.80" in guard.check_plan(0.8, "4 images")
    with pytest.raises(SpendLimitExceeded, match="refusing to start"):
        guard.check_plan(1.2, "6 images")
    guard.before_call(0.5, "first")
    guard.record("first", 0.7)
    with pytest.raises(SpendLimitExceeded, match=r"\$0.70 spent so far"):
        guard.before_call(0.5, "second")
    assert guard.spent == pytest.approx(0.7)


def test_log_helpers(tmp_path):
    log = tmp_path / "logs" / "generations.jsonl"
    assert genlog.read_rows(log) == []
    rows = [
        {"kind": "image", "scene": "a", "candidate": 1, "provider": "fake", "purpose": "generate",
         "status": "ok", "output_sha256": "aa", "id": "1", "cost_usd": 0.1},
        {"kind": "image", "scene": "a", "candidate": 1, "provider": "fake", "purpose": "generate",
         "status": "error", "id": "2", "cost_usd": None, "cost_estimate_usd": 0.1},
        {"kind": "score", "scene": "a", "status": "ok", "id": "3", "cost_usd": None,
         "cost_estimate_usd": 0.2},
    ]  # fmt: skip
    for row in rows:
        genlog.append_row(log, row)
    log.write_text(log.read_text() + "\n")
    read = genlog.read_rows(log)
    assert [r["id"] for r in read] == ["1", "2", "3"]
    assert genlog.image_attempts(read, "a", 1, "fake", "generate") == 2
    assert genlog.find_image_row(read, "a", 1, "aa")["id"] == "1"
    assert genlog.find_image_row(read, "a", 1, "bb") is None
    assert genlog.total_spend(read) == pytest.approx(0.3)
    log.write_text("{not json\n")
    with pytest.raises(ValueError, match="not valid JSON"):
        genlog.read_rows(log)

"""The spend guard: every command that calls a paid API prints what it expects to spend first,
and refuses to go beyond ``DSP_MAX_SPEND_USD`` (15 US dollars when unset) in one run.

The guard checks twice: once for the whole plan before the first call, and again before every
call against what has really been spent so far, in case actual costs run above the estimates.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

ENV_VAR = "DSP_MAX_SPEND_USD"
DEFAULT_LIMIT_USD = 15.0


class SpendLimitExceeded(RuntimeError):
    """A run would cost more than the configured limit; nothing more is sent."""


def limit_from_env(environ: dict[str, str] | None = None) -> float:
    """Read the limit; an unset or empty value means the default."""
    env = os.environ if environ is None else environ
    raw = env.get(ENV_VAR, "").strip()
    if not raw:
        return DEFAULT_LIMIT_USD
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{ENV_VAR} must be a number of US dollars, got {raw!r}") from error
    if value < 0:
        raise ValueError(f"{ENV_VAR} must not be negative, got {raw!r}")
    return value


@dataclass
class SpendGuard:
    limit: float = DEFAULT_LIMIT_USD
    spent: float = 0.0
    calls: list[tuple[str, float]] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> SpendGuard:
        return cls(limit=limit_from_env())

    def check_plan(self, estimate: float, description: str) -> str:
        """Refuse a plan that is over the limit; otherwise return the line to print."""
        line = (
            f"estimated spend: ${estimate:.2f} for {description} "
            f"(limit ${self.limit:.2f} per run, set by {ENV_VAR})"
        )
        if self.spent + estimate > self.limit + 1e-9:
            raise SpendLimitExceeded(
                f"{line}; refusing to start because it is over the limit. Raise {ENV_VAR} or "
                "process fewer scenes or candidates."
            )
        return line

    def before_call(self, estimate: float, what: str) -> None:
        """Refuse a call that would take the real total past the limit."""
        if self.spent + estimate > self.limit + 1e-9:
            raise SpendLimitExceeded(
                f"stopping before {what}: ${self.spent:.2f} spent so far plus ${estimate:.2f} "
                f"estimated would exceed the ${self.limit:.2f} limit ({ENV_VAR})"
            )

    def record(self, what: str, cost: float) -> None:
        self.spent += cost
        self.calls.append((what, cost))

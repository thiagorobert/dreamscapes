"""The generation log, ``pipeline/logs/generations.jsonl``: one JSON object per paid API call.

Image APIs are not deterministic, so the committed masters are the source of truth and this log
explains how each one was made. It is append-only and committed.

Every row has ``id``, ``timestamp`` (UTC, ISO 8601), ``kind`` (``image`` for a generation call,
``score`` for a Claude scoring call), ``purpose`` (``generate``, ``bakeoff`` or ``select``),
``provider``, ``model``, ``scene``, ``status`` (``ok`` or ``error``), ``error``,
``duration_s``, ``cost_estimate_usd`` and ``cost_usd`` (from the usage the provider reported,
or null when it reported none).

Image rows add ``candidate``, ``prompt`` (in full), ``parameters`` (everything else in the
request), ``references`` (path, size and SHA-256 of each attached image), ``seed`` (null when the
provider has none), ``request_id``, ``response_id``, ``revised_prompt``, ``output_path``,
``output_sha256``, ``output_size``, ``usage`` and ``style_block_sha256`` (so a changed style
block shows up as drift between scenes).

Score rows add ``prompt`` (system and text parts), ``parameters``, ``images`` (label, path and
SHA-256 of every image shown), ``request_id``, ``response_id``, ``served_model`` (differs from
``model`` when a server-side fallback answered) and ``usage``.

Morph rows (``kind: morph``, written by `dsp morph`) come in two purposes. ``morph`` is a paid
Veo call: ``scene`` and ``next_scene`` name the edge, plus ``prompt``, ``parameters``,
``references`` (the first and last frames with their SHA-256), ``seed`` (always null),
``style_block_sha256``, ``response_id`` (the operation name), ``polls``, the raw download
(``raw_path``, ``raw_sha256``, ``raw_bytes``, ``raw_info``) and the encoded clip
(``output_path``, ``output_sha256``, ``output_bytes``, ``output_info``), or
``postprocess_error`` when ffmpeg failed after the paid call. ``reencode`` rows cost nothing:
they record a clip rebuilt from an earlier download, pointing at that call with ``source_row``.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def new_row_id() -> str:
    return uuid.uuid4().hex[:16]


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Every row in file order; blank lines are skipped, a corrupt line is an error."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: not valid JSON ({error})") from error
        if isinstance(row, dict):
            rows.append(row)
    return rows


def image_attempts(
    rows: list[dict[str, Any]], scene: str, candidate: int, provider: str, purpose: str
) -> int:
    """How many image calls were already made for this slot (successful or not)."""
    return sum(
        1
        for r in rows
        if r.get("kind") == "image"
        and r.get("scene") == scene
        and r.get("candidate") == candidate
        and r.get("provider") == provider
        and r.get("purpose") == purpose
    )


def find_image_row(
    rows: list[dict[str, Any]], scene: str, candidate: int, sha256: str
) -> dict[str, Any] | None:
    """The latest successful image row that produced exactly this candidate file."""
    for row in reversed(rows):
        if (
            row.get("kind") == "image"
            and row.get("status") == "ok"
            and row.get("scene") == scene
            and row.get("candidate") == candidate
            and row.get("output_sha256") == sha256
        ):
            return row
    return None


def total_spend(rows: list[dict[str, Any]]) -> float:
    """Sum of reported costs, falling back to the estimate for rows without one."""
    total = 0.0
    for row in rows:
        cost = row.get("cost_usd")
        if cost is None and row.get("status") == "ok":
            cost = row.get("cost_estimate_usd")
        total += float(cost or 0.0)
    return total


def spend_summary(path: Path) -> str:
    """One line with everything the log says has been spent, printed after paid runs."""
    try:
        rows = read_rows(path)
    except ValueError as error:
        return f"could not total the generation log: {error}"
    return (
        f"spend recorded in the generation log so far: ${total_spend(rows):.2f} "
        f"over {len(rows)} call(s)"
    )

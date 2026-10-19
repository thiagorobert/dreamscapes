"""Fakes for the three SDK clients, a fake judge client, and a guard that keeps tests offline.

Test modules import ``offline`` (an autouse fixture) so that any attempt to open a socket fails
loudly and no API key from the developer's shell can reach a real client.
"""

from __future__ import annotations

import base64
import json
import re
import socket
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from dreamscapes_pipeline.layout import MASTER_HEIGHT, MASTER_WIDTH, RepoPaths
from dreamscapes_pipeline.providers.base import synthetic_png

KEY_VARS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "DSP_MAX_SPEND_USD",
    "DSP_IMAGE_PROVIDER",
)


class NetworkBlocked(RuntimeError):
    pass


def _blocked(*_args: Any, **_kwargs: Any) -> Any:
    raise NetworkBlocked("tests must not touch the network")


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)


# --------------------------------------------------------------------------------------------
# Repository helpers
# --------------------------------------------------------------------------------------------


def write_master(paths: RepoPaths, scene_id: str, colour: tuple[int, int, int]) -> Path:
    """A real (untagged) master, as `dsp select` would leave it."""
    paths.masters.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (MASTER_WIDTH, MASTER_HEIGHT), colour).save(paths.master(scene_id))
    return paths.master(scene_id)


def write_candidates(
    paths: RepoPaths, scene_id: str, count: int, size: tuple[int, int] = (320, 180)
) -> list[Path]:
    folder = paths.candidate_dir(scene_id)
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for n in range(1, count + 1):
        path = folder / f"cand-{n}.png"
        path.write_bytes(synthetic_png(f"{scene_id}-{n}", size))
        out.append(path)
    return out


def write_style(paths: RepoPaths, text: str = "A quiet painted place, softly lit.") -> None:
    paths.style_md.parent.mkdir(parents=True, exist_ok=True)
    paths.style_md.write_text(
        f"# Style\n\nIntro.\n\n## Style block\n\n{text}\n\n## Do not\n\n- x\n"
    )


# --------------------------------------------------------------------------------------------
# google-genai
# --------------------------------------------------------------------------------------------


class FakeGenaiClient:
    """Stands in for ``google.genai.Client``: records ``interactions.create`` calls."""

    def __init__(self, respond: Callable[[dict[str, Any]], Any] | None = None):
        self.calls: list[dict[str, Any]] = []
        self._respond = respond or self.default_response
        self.interactions = SimpleNamespace(create=self._create)

    def _create(self, **body: Any) -> Any:
        self.calls.append(body)
        return self._respond(body)

    @staticmethod
    def default_response(body: dict[str, Any]) -> Any:
        png = synthetic_png("gemini", (344, 192))
        return SimpleNamespace(
            id="interaction-123",
            status="completed",
            model=body["model"],
            output_image=SimpleNamespace(
                data=base64.b64encode(png).decode("ascii"), mime_type="image/png"
            ),
            output_text=None,
            usage=SimpleNamespace(
                total_input_tokens=1000,
                total_output_tokens=1680,
                total_thought_tokens=400,
                output_tokens_by_modality=[SimpleNamespace(modality="image", tokens=1680)],
            ),
        )


# --------------------------------------------------------------------------------------------
# openai
# --------------------------------------------------------------------------------------------


class FakeOpenAIClient:
    """Stands in for ``openai.OpenAI``: records ``images.generate`` and ``images.edit``."""

    def __init__(self, error: Exception | None = None):
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.error = error
        self.images = SimpleNamespace(generate=self._generate, edit=self._edit)

    def _response(self) -> Any:
        if self.error is not None:
            raise self.error
        png = synthetic_png("openai", (256, 144))
        response = SimpleNamespace(
            data=[
                SimpleNamespace(
                    b64_json=base64.b64encode(png).decode("ascii"), revised_prompt="rewritten"
                )
            ],
            usage={
                "input_tokens": 700,
                "input_tokens_details": {"image_tokens": 500, "text_tokens": 200},
                "output_tokens": 4000,
                "total_tokens": 4700,
            },
        )
        response._request_id = "req_openai_1"
        return response

    def _generate(self, **kwargs: Any) -> Any:
        self.calls.append(("generate", kwargs))
        return self._response()

    def _edit(self, **kwargs: Any) -> Any:
        self.calls.append(("edit", kwargs))
        return self._response()


# --------------------------------------------------------------------------------------------
# anthropic
# --------------------------------------------------------------------------------------------

_LABEL = re.compile(r"^Candidate (\d+):$")


def candidate_numbers(kwargs: dict[str, Any]) -> list[int]:
    content = kwargs["messages"][0]["content"]
    numbers = []
    for block in content:
        if block.get("type") == "text":
            match = _LABEL.match(block["text"])
            if match:
                numbers.append(int(match.group(1)))
    return numbers


def score_entry(number: int, base: int = 3, **overrides: Any) -> dict[str, Any]:
    entry = {
        "candidate": number,
        "style_match": base,
        "bridge_object": base,
        "cleanliness": base,
        "mood": base,
        "composition": base,
        "bridge_object_present": True,
        "has_text_or_logos": False,
        "has_people_or_faces": False,
        "notes": f"candidate {number} looks fine",
    }
    entry.update(overrides)
    return entry


def message(
    text: str,
    *,
    stop_reason: str = "end_turn",
    model: str = "claude-opus-5-5",
    blocks: list[Any] | None = None,
    stop_details: Any = None,
) -> Any:
    content = blocks if blocks is not None else []
    content = [SimpleNamespace(type="thinking", thinking=""), *content]
    content.append(SimpleNamespace(type="text", text=text))
    result = SimpleNamespace(
        id="msg_123",
        model=model,
        stop_reason=stop_reason,
        stop_details=stop_details,
        content=content,
        usage={"input_tokens": 9000, "output_tokens": 2000},
    )
    result._request_id = "req_claude_1"
    return result


class FakeAnthropicClient:
    """Stands in for ``anthropic.Anthropic``: records ``beta.messages.create`` calls.

    By default every candidate scores ``base`` on every criterion except the best one
    (``best``), which scores 5 across the board; ``overrides`` maps a candidate number to
    fields to change, e.g. ``{2: {"has_people_or_faces": True}}``.
    """

    def __init__(
        self,
        *,
        best: int | None = None,
        overrides: dict[int, dict[str, Any]] | None = None,
        respond: Callable[[dict[str, Any]], Any] | None = None,
    ):
        self.calls: list[dict[str, Any]] = []
        self.best = best
        self.overrides = overrides or {}
        self._respond = respond
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._respond is not None:
            return self._respond(kwargs)
        numbers = candidate_numbers(kwargs)
        best = self.best if self.best is not None else numbers[0]
        entries = [
            score_entry(n, 5 if n == best else 3, **self.overrides.get(n, {})) for n in numbers
        ]
        return message(json.dumps({"candidates": entries, "advice": ""}))

"""The image-provider contract shared by every generator, plus a fake for tests.

A provider turns a prompt, an aspect ratio, a size and optional reference images into one image.
Everything provider-specific (SDK, request shape, size vocabulary, pricing) stays inside the
provider module, so `dsp generate` and `dsp bakeoff` only ever talk to this protocol.

Providers never retry on their own: they raise `ProviderError` with ``retryable`` set for
rate limits and server errors, and `dreamscapes_pipeline.retry` decides how long to wait.
"""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from PIL import Image

ASPECT = "16:9"
"""Every Dreamscapes master is 16:9 landscape."""


@dataclass(frozen=True)
class ReferenceImage:
    """An image attached to a request, such as the style anchor master."""

    path: Path
    data: bytes
    mime_type: str = "image/png"

    @classmethod
    def from_path(cls, path: Path) -> ReferenceImage:
        data = path.read_bytes()
        suffix = path.suffix.lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(
            suffix, "image/png"
        )
        return cls(path=path, data=data, mime_type=mime)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    def base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    def describe(self, root: Path | None = None, sent_as: str = "base64") -> str:
        """How the image appears in printed requests: never the bytes, always the provenance."""
        shown = self.path
        if root is not None and self.path.is_relative_to(root):
            shown = self.path.relative_to(root)
        return (
            f"<{self.mime_type} {shown.as_posix()}, {len(self.data)} bytes, "
            f"sha256 {self.sha256}, sent as {sent_as}>"
        )


@dataclass(frozen=True)
class GenerationResult:
    """One generated image and everything the generation log needs to explain it."""

    image: bytes
    mime_type: str
    model: str
    cost_estimate_usd: float
    request_id: str | None = None
    response_id: str | None = None
    revised_prompt: str | None = None
    seed: int | None = None
    cost_usd: float | None = None
    """Cost computed from the usage the provider reported, when it reports any."""
    usage: dict[str, Any] | None = None
    text: str | None = None
    """Any text the model returned alongside the image."""


class ProviderError(RuntimeError):
    """A request failed. ``retryable`` is True for rate limits, timeouts and server errors."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after = retry_after


class MissingCredentials(ProviderError):
    """The API key for a provider is not set."""


def is_retryable_status(status: int | None) -> bool:
    """Rate limits (429), timeouts and conflicts (408, 409) and server errors (5xx)."""
    return status is not None and (status in (408, 409, 429) or status >= 500)


def retry_after_seconds(headers: Any) -> float | None:
    """Parse a ``Retry-After`` header given in seconds; dates and junk are ignored."""
    if headers is None:
        return None
    try:
        value = headers.get("retry-after")
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


@runtime_checkable
class ImageProvider(Protocol):
    """What `dsp generate` needs from an image generator."""

    name: str
    """Short provider name used on the command line and in the log, e.g. ``gemini``."""
    model: str

    def default_size(self) -> str:
        """The size used when the command line does not choose one (the 2K-class master size)."""
        ...

    def estimate_cost(self, size: str, references: int) -> float:
        """Conservative US-dollar estimate for one image, printed before paid calls."""
        ...

    def request(
        self,
        prompt: str,
        aspect: str,
        size: str,
        references: list[ReferenceImage],
        seed: int | None,
    ) -> dict[str, Any]:
        """The exact request `generate` would send, as JSON-ready data.

        Reference image bytes are replaced by a short description so the request can be printed
        by ``--dry-run`` and stored in the log. Raises ``ValueError`` for an unsupported
        aspect ratio or size before anything is sent.
        """
        ...

    def generate(
        self,
        prompt: str,
        aspect: str,
        size: str,
        references: list[ReferenceImage],
        seed: int | None,
    ) -> GenerationResult:
        """Send one request and return the image. Raises `ProviderError` on failure."""
        ...


# --------------------------------------------------------------------------------------------
# Fake provider for tests
# --------------------------------------------------------------------------------------------


def synthetic_png(key: str, size: tuple[int, int] = (256, 144)) -> bytes:
    """A deterministic soft-gradient PNG whose colours depend on ``key``."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    w, h = size
    top = np.array(list(digest[0:3]), dtype=np.float64)
    bottom = np.array(list(digest[3:6]), dtype=np.float64)
    v = (np.arange(h, dtype=np.float64) / max(h - 1, 1))[:, None, None]
    pixels = np.broadcast_to(top * (1 - v) + bottom * v, (h, w, 3))
    image = Image.fromarray(np.ascontiguousarray(np.rint(pixels).astype(np.uint8)), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@dataclass
class FakeCall:
    prompt: str
    aspect: str
    size: str
    references: list[ReferenceImage]
    seed: int | None


@dataclass
class FakeImageProvider:
    """Returns deterministic synthetic images and records every call; never touches a network.

    ``failures`` holds `ProviderError` instances that the first calls raise, in order, so tests
    can exercise retries and error handling.
    """

    name: str = "fake"
    model: str = "fake-image-1"
    cost: float = 0.01
    image_size: tuple[int, int] = (256, 144)
    failures: list[ProviderError] = field(default_factory=list)
    calls: list[FakeCall] = field(default_factory=list)
    attempts: int = 0

    def default_size(self) -> str:
        return "2K"

    def estimate_cost(self, size: str, references: int) -> float:
        return self.cost

    def request(
        self,
        prompt: str,
        aspect: str,
        size: str,
        references: list[ReferenceImage],
        seed: int | None,
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt": prompt,
            "aspect": aspect,
            "size": size,
            "references": [r.describe() for r in references],
            "seed": seed,
        }

    def generate(
        self,
        prompt: str,
        aspect: str,
        size: str,
        references: list[ReferenceImage],
        seed: int | None,
    ) -> GenerationResult:
        self.attempts += 1
        if self.failures:
            raise self.failures.pop(0)
        self.calls.append(FakeCall(prompt, aspect, size, list(references), seed))
        key = f"{self.name}|{self.model}|{prompt}|{seed}|{len(self.calls)}"
        return GenerationResult(
            image=synthetic_png(key, self.image_size),
            mime_type="image/png",
            model=self.model,
            cost_estimate_usd=self.cost,
            cost_usd=self.cost,
            request_id=f"req-{len(self.calls)}",
            response_id=f"resp-{len(self.calls)}",
            seed=seed,
        )

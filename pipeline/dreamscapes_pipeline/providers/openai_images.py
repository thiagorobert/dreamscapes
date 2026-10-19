"""OpenAI GPT Image generation (the bake-off contender), through the ``openai`` SDK.

Checked against https://developers.openai.com/api/docs/guides/image-generation on 2026-10-19:

* Models: ``gpt-image-2.5-flare`` (fast, high quality; our default) and
  ``gpt-image-2.5-sunburst`` (for edit precision). Both take the same parameters.
* Without references: ``client.images.generate(model=..., prompt=..., size=..., quality=...)``.
  With references: ``client.images.edit(model=..., image=[file, ...], prompt=...)``, which the
  SDK sends as repeated ``image[]`` multipart fields.
* ``size``: ``auto``, the listed sizes, or a custom ``WIDTHxHEIGHT`` where both edges are
  multiples of 16, no edge exceeds 3840, the aspect ratio is between 1:3 and 3:1 and the total
  is 655,360 to 8,294,400 pixels. Above 2560x1440 is experimental. Our master is 2048x1152.
* ``quality``: ``low``, ``medium``, ``high``, ``xhigh``, ``max`` or ``auto``.
* Output arrives base64-encoded in ``data[0].b64_json``; ``output_format="png"`` keeps it
  lossless. There is no seed parameter. ``revised_prompt`` is filled when the API rewrote the
  prompt.

Pricing is per token: $30 per million image output tokens, $8 per million image input tokens and
$5 per million text input tokens. The guide publishes no per-image token counts for these models,
so the estimate below assumes generous counts per megapixel by quality; the cost logged after a
call comes from the ``usage`` the API reports.
"""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import Any

from .base import (
    GenerationResult,
    MissingCredentials,
    ProviderError,
    ReferenceImage,
    is_retryable_status,
    retry_after_seconds,
)

DEFAULT_MODEL = "gpt-image-2.5-flare"
DEFAULT_SIZE = "2048x1152"
DEFAULT_QUALITY = "high"
QUALITIES = ("low", "medium", "high", "xhigh", "max", "auto")
REQUEST_TIMEOUT_S = 300.0

# Shorthands so `--size 1K/2K/4K` means the same class of image for both providers (16:9 only;
# 1024x576 would fall below the API's minimum pixel count, hence 1280x720).
SIZE_ALIASES = {"1K": "1280x720", "2K": "2048x1152", "4K": "3840x2160"}

MIN_PIXELS = 655_360
MAX_PIXELS = 8_294_400
MAX_EDGE = 3840

IMAGE_OUTPUT_PER_MILLION = 30.0
IMAGE_INPUT_PER_MILLION = 8.0
TEXT_INPUT_PER_MILLION = 5.0

# Assumed output image tokens per megapixel, deliberately on the high side.
OUTPUT_TOKENS_PER_MEGAPIXEL = {
    "low": 300,
    "medium": 1_100,
    "high": 4_200,
    "xhigh": 6_500,
    "max": 9_000,
    "auto": 4_200,
}
PROMPT_TOKENS = 600
REFERENCE_TOKENS = 3_000

_SIZE = re.compile(r"^(\d+)x(\d+)$")


def resolve_size(size: str) -> str:
    return SIZE_ALIASES.get(size, size)


def check_size(size: str) -> tuple[int, int]:
    """Validate a custom ``WIDTHxHEIGHT`` against the documented limits."""
    match = _SIZE.match(size)
    if not match:
        raise ValueError(f"openai: size must look like 2048x1152 (or 1K/2K/4K), got {size!r}")
    width, height = int(match.group(1)), int(match.group(2))
    problems = []
    if width % 16 or height % 16:
        problems.append("both edges must be multiples of 16")
    if max(width, height) > MAX_EDGE:
        problems.append(f"no edge may exceed {MAX_EDGE}")
    if not MIN_PIXELS <= width * height <= MAX_PIXELS:
        problems.append(f"total pixels must be between {MIN_PIXELS:,} and {MAX_PIXELS:,}")
    if not (1 / 3 <= width / height <= 3):
        problems.append("aspect ratio must be between 1:3 and 3:1")
    if problems:
        raise ValueError(f"openai: size {size} is not allowed: {'; '.join(problems)}")
    return width, height


def _aspect_of(width: int, height: int) -> float:
    return width / height


def _parse_aspect(aspect: str) -> float:
    w, _, h = aspect.partition(":")
    return float(w) / float(h)


class OpenAIImageProvider:
    """`ImageProvider` for GPT Image models.

    ``client`` may be injected (tests pass a fake); otherwise an ``openai.OpenAI`` client is
    built on first use from ``OPENAI_API_KEY`` with the SDK's retries off, because
    `dreamscapes_pipeline.retry` owns retrying.
    """

    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        quality: str | None = None,
        client: Any = None,
        root: Path | None = None,
    ):
        self.model = model or DEFAULT_MODEL
        self.quality = quality or DEFAULT_QUALITY
        if self.quality not in QUALITIES:
            raise ValueError(f"openai: quality must be one of {', '.join(QUALITIES)}")
        self._client = client
        self._root = root

    def default_size(self) -> str:
        return DEFAULT_SIZE

    def estimate_cost(self, size: str, references: int) -> float:
        width, height = check_size(resolve_size(size))
        megapixels = width * height / 1e6
        output = OUTPUT_TOKENS_PER_MEGAPIXEL[self.quality] * megapixels
        cost = (
            output * IMAGE_OUTPUT_PER_MILLION
            + REFERENCE_TOKENS * references * IMAGE_INPUT_PER_MILLION
            + PROMPT_TOKENS * TEXT_INPUT_PER_MILLION
        ) / 1e6
        return round(cost, 4)

    def _params(
        self, prompt: str, aspect: str, size: str, references: list[ReferenceImage]
    ) -> dict[str, Any]:
        resolved = resolve_size(size)
        width, height = check_size(resolved)
        if abs(_aspect_of(width, height) - _parse_aspect(aspect)) > 0.01:
            raise ValueError(f"openai: size {resolved} does not have the {aspect} aspect ratio")
        return {
            "model": self.model,
            "prompt": prompt,
            "size": resolved,
            "quality": self.quality,
            "output_format": "png",
            "n": 1,
        }

    def request(
        self,
        prompt: str,
        aspect: str,
        size: str,
        references: list[ReferenceImage],
        seed: int | None,
    ) -> dict[str, Any]:
        params = self._params(prompt, aspect, size, references)
        if references:
            return {
                "endpoint": "client.images.edit",
                **params,
                "image": [
                    ref.describe(self._root, sent_as="a multipart image[] file")
                    for ref in references
                ],
            }
        return {"endpoint": "client.images.generate", **params}

    def _get_client(self) -> Any:
        if self._client is None:
            key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not key:
                raise MissingCredentials("OPENAI_API_KEY is not set; add it to .env")
            import openai

            self._client = openai.OpenAI(api_key=key, max_retries=0, timeout=REQUEST_TIMEOUT_S)
        return self._client

    def generate(
        self,
        prompt: str,
        aspect: str,
        size: str,
        references: list[ReferenceImage],
        seed: int | None,
    ) -> GenerationResult:
        params = self._params(prompt, aspect, size, references)
        estimate = self.estimate_cost(size, len(references))
        client = self._get_client()
        try:
            if references:
                files = [
                    (ref.path.name, ref.data, ref.mime_type) for ref in references
                ]  # sent as repeated image[] fields
                response = client.images.edit(image=files, **params)
            else:
                response = client.images.generate(**params)
        except Exception as error:
            raise _translate(error) from error

        # Usage first: an answer without a usable image was still billed.
        usage = _usage_dict(getattr(response, "usage", None))
        cost = _cost_from_usage(usage)
        data = getattr(response, "data", None) or []
        if not data or not getattr(data[0], "b64_json", None):
            raise ProviderError("openai returned no image", retryable=False, cost_usd=cost)
        try:
            image = base64.b64decode(data[0].b64_json)
        except ValueError as error:
            raise ProviderError(
                f"openai returned image data that is not valid base64: {error}",
                retryable=False,
                cost_usd=cost,
            ) from error
        return GenerationResult(
            image=image,
            mime_type="image/png",
            model=self.model,
            cost_estimate_usd=estimate,
            cost_usd=cost,
            request_id=getattr(response, "_request_id", None),
            response_id=None,
            revised_prompt=getattr(data[0], "revised_prompt", None),
            seed=None,
            usage=usage,
        )


def _translate(error: Exception) -> ProviderError:
    if isinstance(error, ProviderError):
        return error
    import openai

    if isinstance(error, openai.APIStatusError):
        return ProviderError(
            f"openai request failed with HTTP {error.status_code}: {error.message}",
            retryable=is_retryable_status(error.status_code),
            status_code=error.status_code,
            retry_after=retry_after_seconds(getattr(error.response, "headers", None)),
        )
    if isinstance(error, (openai.APIConnectionError, openai.APITimeoutError)):
        return ProviderError(f"openai connection problem: {error}", retryable=True)
    return ProviderError(f"openai request failed: {error}", retryable=False)


def _usage_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json", exclude_none=True)
    return dict(usage) if isinstance(usage, dict) else None


def _cost_from_usage(usage: dict[str, Any] | None) -> float | None:
    if not usage or "output_tokens" not in usage:
        return None
    details_in = usage.get("input_tokens_details") or {}
    image_in = details_in.get("image_tokens", 0)
    text_in = details_in.get("text_tokens", usage.get("input_tokens", 0) - image_in)
    cost = (
        usage["output_tokens"] * IMAGE_OUTPUT_PER_MILLION
        + image_in * IMAGE_INPUT_PER_MILLION
        + text_in * TEXT_INPUT_PER_MILLION
    ) / 1e6
    return round(cost, 6)

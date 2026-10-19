"""Google Gemini image generation (the default provider), through the ``google-genai`` SDK.

Checked against https://ai.google.dev/gemini-api/docs/image-generation on 2026-10-19:

* Model: ``gemini-nano-banana-2.1`` ("Nano Banana 2.1") is the one the docs recommend for new
  projects. ``gemini-3-pro-image`` and ``gemini-3.1-flash-image`` take the same request.
* Request: the docs use the Interactions API, ``client.interactions.create(model=..., input=...,
  response_format=...)``, not the older ``models.generate_content``. ``input`` is a list of
  content items: the text prompt first, then reference images as
  ``{"type": "image", "data": <base64>, "mime_type": "image/png"}``.
* ``response_format={"type": "image", "aspect_ratio": "16:9", "image_size": "2K"}``. Sizes are
  ``1K``, ``2K`` and ``4K`` (an upper-case K is required; the 512 size is not available on this
  model). Aspect ratios include ``16:9`` and ``21:9``.
* Reference images: up to 14 in total (Nano Banana 2.1 takes up to 10 object images and up to
  5 character images). We attach at most one, the ungraded style anchor master.
* Seed: the image docs do not mention one. The SDK's ``generation_config`` has a generic
  ``seed`` field, so we send the seed there and log it, but nothing promises that the same seed
  reproduces an image; the committed master stays the source of truth.
* The final image is ``interaction.output_image`` (base64 ``data`` plus ``mime_type``); the
  response id is ``interaction.id``. Interim "thought images" appear only in ``steps``.
* Watermark: every generated image carries Google's invisible SynthID watermark. It is
  imperceptible, survives our grade and export, and needs no handling.

Pricing (https://ai.google.dev/gemini-api/docs/pricing, standard tier): image output is $30 per
million tokens, about $0.0336 for a 1K image, $0.0504 for 2K and $0.113 for 4K; input is $1.50
per million tokens and text or thinking output $7.50 per million.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import (
    GenerationResult,
    MissingCredentials,
    ProviderError,
    ReferenceImage,
    is_retryable_status,
)

DEFAULT_MODEL = "gemini-nano-banana-2.1"
DEFAULT_SIZE = "2K"
SIZES = ("1K", "2K", "4K")
ASPECT_RATIOS = ("1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9")
REQUEST_TIMEOUT_MS = 300_000


@dataclass(frozen=True)
class GeminiPrices:
    per_image: dict[str, float]
    input_per_million: float
    text_output_per_million: float
    image_output_per_million: float


PRICES: dict[str, GeminiPrices] = {
    "gemini-nano-banana-2.1": GeminiPrices(
        per_image={"1K": 0.0336, "2K": 0.0504, "4K": 0.113},
        input_per_million=1.50,
        text_output_per_million=7.50,
        image_output_per_million=30.0,
    ),
    "gemini-3.1-flash-image": GeminiPrices(
        per_image={"1K": 0.067, "2K": 0.101, "4K": 0.151},
        input_per_million=0.50,
        text_output_per_million=60.0,  # not published separately; priced as image output
        image_output_per_million=60.0,
    ),
    "gemini-3-pro-image": GeminiPrices(
        per_image={"1K": 0.134, "2K": 0.134, "4K": 0.24},
        input_per_million=2.00,
        text_output_per_million=120.0,  # not published separately; priced as image output
        image_output_per_million=120.0,
    ),
}

# Token counts behind the estimate: the prompt is a few hundred tokens, a reference image is
# billed like an input image, and the model thinks before drawing.
PROMPT_TOKENS = 600
REFERENCE_TOKENS = 1_800
THINKING_TOKENS = 2_000


def _prices(model: str) -> GeminiPrices:
    # An unknown model is priced like the most expensive known one, so the guard stays safe.
    return PRICES.get(model, PRICES["gemini-3-pro-image"])


class GeminiImageProvider:
    """`ImageProvider` for Gemini image models.

    ``client`` may be injected (tests pass a fake); otherwise a ``google.genai.Client`` is built
    on first use from ``GEMINI_API_KEY``, with the SDK's own retries switched off because
    `dreamscapes_pipeline.retry` owns retrying.
    """

    name = "gemini"

    def __init__(self, model: str | None = None, client: Any = None, root: Path | None = None):
        self.model = model or DEFAULT_MODEL
        self._client = client
        self._root = root

    def default_size(self) -> str:
        return DEFAULT_SIZE

    def estimate_cost(self, size: str, references: int) -> float:
        prices = _prices(self.model)
        image = prices.per_image.get(size, max(prices.per_image.values()))
        tokens_in = PROMPT_TOKENS + REFERENCE_TOKENS * references
        return round(
            image
            + tokens_in * prices.input_per_million / 1e6
            + THINKING_TOKENS * prices.text_output_per_million / 1e6,
            4,
        )

    def _check(self, aspect: str, size: str) -> None:
        if aspect not in ASPECT_RATIOS:
            raise ValueError(f"gemini: unsupported aspect ratio {aspect!r}")
        if size not in SIZES:
            raise ValueError(f"gemini: unsupported size {size!r}; use one of {', '.join(SIZES)}")

    def _body(
        self,
        prompt: str,
        aspect: str,
        size: str,
        references: list[ReferenceImage],
        seed: int | None,
        *,
        describe: bool,
    ) -> dict[str, Any]:
        self._check(aspect, size)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for ref in references:
            data = ref.describe(self._root) if describe else ref.base64()
            content.append({"type": "image", "data": data, "mime_type": ref.mime_type})
        body: dict[str, Any] = {
            "model": self.model,
            "input": content,
            "response_format": {"type": "image", "aspect_ratio": aspect, "image_size": size},
        }
        if seed is not None:
            body["generation_config"] = {"seed": seed}
        return body

    def request(
        self,
        prompt: str,
        aspect: str,
        size: str,
        references: list[ReferenceImage],
        seed: int | None,
    ) -> dict[str, Any]:
        return {
            "endpoint": "client.interactions.create",
            **self._body(prompt, aspect, size, references, seed, describe=True),
        }

    def _get_client(self) -> Any:
        if self._client is None:
            key = os.environ.get("GEMINI_API_KEY", "").strip()
            if not key:
                raise MissingCredentials("GEMINI_API_KEY is not set; add it to .env")
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    timeout=REQUEST_TIMEOUT_MS,
                    retry_options=types.HttpRetryOptions(attempts=0),
                ),
            )
        return self._client

    def generate(
        self,
        prompt: str,
        aspect: str,
        size: str,
        references: list[ReferenceImage],
        seed: int | None,
    ) -> GenerationResult:
        body = self._body(prompt, aspect, size, references, seed, describe=False)
        estimate = self.estimate_cost(size, len(references))
        client = self._get_client()
        try:
            interaction = client.interactions.create(**body)
        except Exception as error:
            raise _translate(error) from error

        status = getattr(interaction, "status", None)
        image = getattr(interaction, "output_image", None)
        text = getattr(interaction, "output_text", None)
        if status not in (None, "completed") or image is None or not getattr(image, "data", None):
            detail = f"status {status}" if status else "no image in the response"
            if text:
                detail += f"; model said: {text.strip()[:300]}"
            raise ProviderError(f"gemini returned no image ({detail})", retryable=False)

        data = image.data
        raw = base64.b64decode(data) if isinstance(data, str) else bytes(data)
        usage = _usage_dict(getattr(interaction, "usage", None))
        return GenerationResult(
            image=raw,
            mime_type=getattr(image, "mime_type", None) or "image/png",
            model=str(getattr(interaction, "model", None) or self.model),
            cost_estimate_usd=estimate,
            cost_usd=_cost_from_usage(self.model, usage),
            request_id=None,
            response_id=getattr(interaction, "id", None) or None,
            revised_prompt=None,
            seed=seed,
            usage=usage,
            text=text or None,
        )


def _translate(error: Exception) -> ProviderError:
    """Map SDK and transport errors to `ProviderError`, marking the ones worth retrying."""
    if isinstance(error, ProviderError):
        return error
    code = getattr(error, "code", None)
    if not isinstance(code, int):
        code = getattr(error, "status_code", None)
    if isinstance(code, int):
        return ProviderError(
            f"gemini request failed with HTTP {code}: {error}",
            retryable=is_retryable_status(code),
            status_code=code,
        )
    try:
        import httpx

        if isinstance(error, httpx.TransportError):
            return ProviderError(f"gemini connection problem: {error}", retryable=True)
    except ImportError:  # pragma: no cover - httpx ships with google-genai
        pass
    return ProviderError(f"gemini request failed: {error}", retryable=False)


def _usage_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    out: dict[str, Any] = {}
    for name in ("total_input_tokens", "total_output_tokens", "total_thought_tokens"):
        value = getattr(usage, name, None)
        if value is not None:
            out[name] = value
    by_modality = getattr(usage, "output_tokens_by_modality", None) or []
    modalities = {
        str(m.modality): m.tokens
        for m in by_modality
        if getattr(m, "modality", None) is not None and getattr(m, "tokens", None) is not None
    }
    if modalities:
        out["output_tokens_by_modality"] = modalities
    return out or None


def _cost_from_usage(model: str, usage: dict[str, Any] | None) -> float | None:
    """Dollar cost from reported tokens: input, image output, and text plus thinking output."""
    if not usage or "total_input_tokens" not in usage:
        return None
    prices = _prices(model)
    by_modality = usage.get("output_tokens_by_modality", {})
    image_tokens = by_modality.get("image", 0)
    text_tokens = by_modality.get("text", 0) + usage.get("total_thought_tokens", 0)
    if not by_modality:
        image_tokens = usage.get("total_output_tokens", 0)
    cost = (
        usage["total_input_tokens"] * prices.input_per_million
        + image_tokens * prices.image_output_per_million
        + text_tokens * prices.text_output_per_million
    ) / 1e6
    return round(cost, 6)

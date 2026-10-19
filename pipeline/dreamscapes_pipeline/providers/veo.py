"""Google Veo 3.1 first-frame/last-frame video generation, through the ``google-genai`` SDK.

Checked against https://ai.google.dev/gemini-api/docs/veo and
https://ai.google.dev/gemini-api/docs/pricing on 2026-10-19:

* Models: ``veo-3.1-generate-preview`` (Veo 3.1), ``veo-3.1-fast-generate-preview`` (Veo 3.1
  Fast, the default here) and ``veo-3.1-lite-generate-preview`` (Veo 3.1 Lite). The Veo 3 ids
  (``veo-3.0-generate-001``, ``veo-3.0-fast-generate-001``) are marked deprecated. Generating
  from a first and a last frame ("interpolation") is documented for the Veo 3.1 models.
* Request: ``client.models.generate_videos(model=..., prompt=..., image=<first frame>,
  config=types.GenerateVideosConfig(last_frame=<last frame>, aspect_ratio="16:9",
  resolution="1080p", duration_seconds=8))``. ``image`` is the first frame and ``last_frame``
  sits in the config; both are ``types.Image(image_bytes=..., mime_type=...)``, and
  ``last_frame`` is only valid together with ``image``. The docs' examples send PNGs.
* Durations are 4, 6 or 8 seconds; 1080p and 4k need 8 seconds (720p is the default). Aspect
  ratios are ``16:9`` (default) and ``9:16``. Lite has no 4k. Output is 24 frames per second.
* Audio is always generated ("always on"; the Gemini API has no switch for it, so the SDK's
  ``generate_audio`` field is not sent). `dsp morph` strips it with ffmpeg.
* ``person_generation`` can only be ``allow_adult`` for image-to-video and interpolation; it is
  left unset. ``seed`` is documented for Veo 3 only and is not sent. One video per request.
* The call is a long-running operation: poll ``client.operations.get(operation)`` (the docs
  sleep 10 seconds between polls) until ``operation.done``, then download
  ``operation.response.generated_videos[0].video`` with ``client.files.download(file=video)``,
  which returns the bytes. Latency is 11 seconds to about 6 minutes at peak. Generated videos
  are kept on Google's side for 2 days only.
* A video blocked by the safety filters is not charged; the response then carries
  ``rai_media_filtered_count`` and ``rai_media_filtered_reasons`` instead of a video.
* Price per second of output (paid tier, "video with audio", the only option): Veo 3.1 Fast
  $0.10 at 720p, $0.12 at 1080p, $0.30 at 4k; Veo 3.1 $0.40, $0.40, $0.60; Veo 3.1 Lite $0.05
  and $0.08 (no 4k). An 8-second Fast clip at 1080p costs $0.96.

The provider never retries or sleeps on its own: each step raises `ProviderError` with
``retryable`` set for rate limits and server errors, and `dsp morph` owns the polling loop and
the retries, exactly as `dreamscapes_pipeline.retry` does for images.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import (
    MissingCredentials,
    ProviderError,
    ReferenceImage,
    is_retryable_status,
    retry_after_seconds,
)

DEFAULT_MODEL = "veo-3.1-fast-generate-preview"
ASPECT = "16:9"
RESOLUTIONS = ("720p", "1080p", "4k")
DURATIONS = (4, 6, 8)
LONG_ONLY_RESOLUTIONS = ("1080p", "4k")
"""Resolutions the docs only allow for 8-second clips."""
LONG_DURATION = 8
REQUEST_TIMEOUT_MS = 300_000

PRICE_PER_SECOND: dict[str, dict[str, float]] = {
    "veo-3.1-fast-generate-preview": {"720p": 0.10, "1080p": 0.12, "4k": 0.30},
    "veo-3.1-generate-preview": {"720p": 0.40, "1080p": 0.40, "4k": 0.60},
    "veo-3.1-lite-generate-preview": {"720p": 0.05, "1080p": 0.08},
}


def price_per_second(model: str, resolution: str) -> float:
    """Dollars per second of video. Unknown models and resolutions are priced like the most
    expensive known option, so the spend guard errs on the safe side."""
    prices = PRICE_PER_SECOND.get(model)
    if prices and resolution in prices:
        return prices[resolution]
    return max(p for table in PRICE_PER_SECOND.values() for p in table.values())


@dataclass(frozen=True)
class GeneratedVideo:
    """The finished operation's video handle, before download."""

    handle: Any
    """The SDK's ``types.Video``, passed back to `VeoProvider.download`."""
    mime_type: str | None
    uri: str | None


class VeoProvider:
    """First-frame/last-frame clips from Veo.

    ``client`` may be injected (tests pass a fake); otherwise a ``google.genai.Client`` is built
    on first use from ``GEMINI_API_KEY``, with the SDK's own retries switched off.
    """

    name = "veo"

    def __init__(self, model: str | None = None, client: Any = None, root: Path | None = None):
        self.model = model or DEFAULT_MODEL
        self._client = client
        self._root = root

    # ----------------------------------------------------------------------------------------
    # Planning (no network)
    # ----------------------------------------------------------------------------------------

    def check(self, resolution: str, duration: int) -> None:
        """Refuse settings the docs do not allow, before anything is sent."""
        if resolution not in RESOLUTIONS:
            raise ValueError(
                f"veo: unsupported resolution {resolution!r}; use one of {', '.join(RESOLUTIONS)}"
            )
        if duration not in DURATIONS:
            raise ValueError(
                f"veo: unsupported duration {duration}; use one of "
                f"{', '.join(str(d) for d in DURATIONS)} seconds"
            )
        if resolution in LONG_ONLY_RESOLUTIONS and duration != LONG_DURATION:
            raise ValueError(f"veo: {resolution} needs {LONG_DURATION}-second clips")
        if self.model.startswith("veo-3.1-lite") and resolution == "4k":
            raise ValueError("veo: Veo 3.1 Lite has no 4k output")

    def estimate_cost(self, resolution: str, duration: int) -> float:
        return round(price_per_second(self.model, resolution) * duration, 4)

    def _config(self, last: Any, resolution: str, duration: int) -> dict[str, Any]:
        return {
            "last_frame": last,
            "aspect_ratio": ASPECT,
            "resolution": resolution,
            "duration_seconds": duration,
        }

    def request(
        self,
        prompt: str,
        first: ReferenceImage,
        last: ReferenceImage,
        resolution: str,
        duration: int,
    ) -> dict[str, Any]:
        """The exact call `start` makes, as JSON-ready data with the image bytes described."""
        self.check(resolution, duration)
        return {
            "endpoint": "client.models.generate_videos",
            "model": self.model,
            "prompt": prompt,
            "image": first.describe(self._root, sent_as="image_bytes"),
            "config": self._config(
                last.describe(self._root, sent_as="image_bytes"), resolution, duration
            ),
        }

    # ----------------------------------------------------------------------------------------
    # The three network steps
    # ----------------------------------------------------------------------------------------

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

    @staticmethod
    def _image(ref: ReferenceImage) -> Any:
        from google.genai import types

        return types.Image(image_bytes=ref.data, mime_type=ref.mime_type)

    def start(
        self,
        prompt: str,
        first: ReferenceImage,
        last: ReferenceImage,
        resolution: str,
        duration: int,
    ) -> Any:
        """Submit the job; returns the long-running operation."""
        self.check(resolution, duration)
        client = self._get_client()
        from google.genai import types

        config = types.GenerateVideosConfig(**self._config(self._image(last), resolution, duration))
        try:
            return client.models.generate_videos(
                model=self.model, prompt=prompt, image=self._image(first), config=config
            )
        except Exception as error:
            raise _translate(error, "starting the video") from error

    def refresh(self, operation: Any) -> Any:
        """One poll of the operation."""
        try:
            return self._get_client().operations.get(operation)
        except Exception as error:
            raise _translate(error, "polling the video operation") from error

    def finished_video(self, operation: Any) -> GeneratedVideo:
        """The video of a finished operation, or a `ProviderError` saying why there is none.

        A failed or blocked operation is not retryable: the same request would most likely be
        refused again. A blocked video is not charged (``cost_usd=0``).
        """
        error = getattr(operation, "error", None)
        if error:
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise ProviderError(f"veo operation failed: {message}", retryable=False)
        response = getattr(operation, "response", None) or getattr(operation, "result", None)
        videos = list(getattr(response, "generated_videos", None) or [])
        if not videos or getattr(videos[0], "video", None) is None:
            filtered = getattr(response, "rai_media_filtered_count", None)
            reasons = getattr(response, "rai_media_filtered_reasons", None) or []
            if filtered:
                raise ProviderError(
                    f"veo blocked the video ({filtered} filtered: {'; '.join(reasons)})",
                    retryable=False,
                    cost_usd=0.0,
                )
            raise ProviderError("veo finished without a video", retryable=False)
        video = videos[0].video
        return GeneratedVideo(
            handle=video,
            mime_type=getattr(video, "mime_type", None),
            uri=getattr(video, "uri", None),
        )

    def download(self, video: GeneratedVideo) -> bytes:
        """The MP4 bytes; inline bytes are used when the response already carries them."""
        inline = getattr(video.handle, "video_bytes", None)
        if inline:
            return bytes(inline)
        try:
            data = self._get_client().files.download(file=video.handle)
        except Exception as error:
            raise _translate(error, "downloading the video") from error
        if not data:
            raise ProviderError("veo download returned no bytes", retryable=True)
        return bytes(data)


def _translate(error: Exception, doing: str) -> ProviderError:
    """Map SDK and transport errors to `ProviderError`, marking the ones worth retrying."""
    if isinstance(error, ProviderError):
        return error
    code = getattr(error, "code", None)
    if not isinstance(code, int):
        code = getattr(error, "status_code", None)
    if isinstance(code, int):
        response = getattr(error, "response", None)
        return ProviderError(
            f"veo: {doing} failed with HTTP {code}: {error}",
            retryable=is_retryable_status(code),
            status_code=code,
            retry_after=retry_after_seconds(getattr(response, "headers", None)),
        )
    try:
        import httpx

        if isinstance(error, httpx.TransportError):
            return ProviderError(f"veo: connection problem while {doing}: {error}", retryable=True)
    except ImportError:  # pragma: no cover - httpx ships with google-genai
        pass
    return ProviderError(f"veo: {doing} failed: {error}", retryable=False)

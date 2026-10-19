"""Image providers behind one protocol; pick one with ``--provider`` or ``DSP_IMAGE_PROVIDER``."""

from __future__ import annotations

from pathlib import Path

from .base import (
    ASPECT,
    FakeImageProvider,
    GenerationResult,
    ImageProvider,
    MissingCredentials,
    ProviderError,
    ReferenceImage,
)

PROVIDER_NAMES = ("gemini", "openai")

__all__ = [
    "ASPECT",
    "PROVIDER_NAMES",
    "FakeImageProvider",
    "GenerationResult",
    "ImageProvider",
    "MissingCredentials",
    "ProviderError",
    "ReferenceImage",
    "make_image_provider",
]


def make_image_provider(
    name: str,
    *,
    model: str | None = None,
    quality: str | None = None,
    root: Path | None = None,
) -> ImageProvider:
    """Build a provider by name. Building one never calls the network or needs a key."""
    key = (name or "").strip().lower() or "gemini"
    if key == "gemini":
        from .gemini import GeminiImageProvider

        if quality is not None:
            raise ValueError("--quality only applies to the openai provider")
        return GeminiImageProvider(model=model, root=root)
    if key == "openai":
        from .openai_images import OpenAIImageProvider

        return OpenAIImageProvider(model=model, quality=quality, root=root)
    raise ValueError(f"unknown image provider {name!r}; use one of {', '.join(PROVIDER_NAMES)}")

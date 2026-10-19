"""`dsp grade`: the deterministic colour grade that gives every scene the same look.

Applied to each ungraded master ``assets/masters/<scene>.png`` and written to
``assets/masters/<scene>.graded.png``, in this order (plan section 5.5):

1. Desaturate: keep ``saturation`` of the colour (mix each pixel toward its luminance).
2. Tone: above ``highlight_shoulder`` brightness rolls off smoothly instead of clipping (the
   curve keeps its slope at the shoulder and reaches 0.95 for white), then black is lifted to
   ``black_lift`` and everything is remapped linearly into that range.
3. Split-tone: shadows lean toward ``shadow_colour`` and highlights toward ``highlight_colour``,
   by ``strength``, keeping each pixel's luminance.
4. Bloom: pixels brighter than ``threshold`` are blurred with a Gaussian whose standard
   deviation is ``radius`` times the image width and screen-blended back at ``mix``.
5. Vignette: the corners are darkened by ``strength``, fading smoothly to nothing at the centre.
6. Grain: Gaussian noise with standard deviation ``sigma`` (0-255 scale) added equally to all
   three channels, so it changes luminance only. The generator is seeded from the scene id, so
   grading a scene twice gives exactly the same pixels.

Parameters come from ``content/grade.toml`` (tables ``desaturate``, ``tone``, ``split_tone``,
``bloom``, ``vignette``, ``grain``); a missing file or key falls back to the defaults below,
which are the plan's values. Unknown keys are reported, never silently used.

Reproducibility across machines: the arithmetic uses only operations IEEE 754 rounds exactly
(addition, subtraction, multiplication and division; no ``exp`` or ``pow``), the blur kernel
comes from integer binomial coefficients, and the blur runs on an exactly averaged downsample.
The same input therefore grades to the same bytes on macOS and Linux, which the golden test
checks.

Placeholder masters (from `dsp placeholders`) are never overwritten: their graded copy belongs
to that command. ``--preview`` still shows them graded in ``assets/review/grade.html``.
"""

from __future__ import annotations

import hashlib
import math
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .imaging import png_bytes, webp_preview, write_if_changed
from .layout import RepoPaths
from .manifest import Manifest
from .placeholders import is_placeholder_image

Echo = Callable[[str], None]

LUMA = (0.2126, 0.7152, 0.0722)  # Rec. 709
Channel = Annotated[int, Field(ge=0, le=255)]
RGB = tuple[Channel, Channel, Channel]


class _Section(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Desaturate(_Section):
    saturation: float = Field(default=0.85, ge=0.0, le=2.0)


class Tone(_Section):
    black_lift: float = Field(default=18.0, ge=0.0, le=255.0)
    highlight_shoulder: float = Field(default=0.8, ge=0.0, le=1.0)


class SplitTone(_Section):
    shadow_colour: RGB = (64, 128, 132)
    highlight_colour: RGB = (228, 170, 96)
    strength: float = Field(default=0.12, ge=0.0, le=1.0)


class Bloom(_Section):
    threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    radius: float = Field(default=0.02, ge=0.0, le=0.25)
    mix: float = Field(default=0.25, ge=0.0, le=1.0)


class Vignette(_Section):
    strength: float = Field(default=0.15, ge=0.0, le=1.0)


class Grain(_Section):
    sigma: float = Field(default=2.5, ge=0.0, le=64.0)


class GradeParams(_Section):
    desaturate: Desaturate = Field(default_factory=Desaturate)
    tone: Tone = Field(default_factory=Tone)
    split_tone: SplitTone = Field(default_factory=SplitTone)
    bloom: Bloom = Field(default_factory=Bloom)
    vignette: Vignette = Field(default_factory=Vignette)
    grain: Grain = Field(default_factory=Grain)


class GradeConfigError(ValueError):
    """content/grade.toml cannot be read or holds an invalid value."""


@dataclass
class LoadedParams:
    params: GradeParams
    source: str
    warnings: list[str] = field(default_factory=list)


def _unknown_keys(raw: dict[str, Any]) -> list[str]:
    unknown = []
    sections = GradeParams.model_fields
    for name, value in raw.items():
        if name not in sections:
            unknown.append(name)
            continue
        model = sections[name].annotation
        if isinstance(value, dict) and isinstance(model, type) and issubclass(model, BaseModel):
            unknown.extend(f"{name}.{key}" for key in value if key not in model.model_fields)
    return unknown


def parse_grade_params(raw: dict[str, Any], source: str = "content/grade.toml") -> LoadedParams:
    try:
        params = GradeParams.model_validate(raw)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors()
        )
        raise GradeConfigError(f"{source}: {problems}") from error
    warnings = [
        f"{source}: unknown key '{key}' is ignored by the grade" for key in _unknown_keys(raw)
    ]
    return LoadedParams(params, source, warnings)


def load_grade_params(paths: RepoPaths) -> LoadedParams:
    path = paths.grade_toml
    if not path.exists():
        return LoadedParams(GradeParams(), "built-in defaults (content/grade.toml is missing)")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise GradeConfigError(f"content/grade.toml is not valid TOML: {error}") from error
    return parse_grade_params(raw)


# --------------------------------------------------------------------------------------------
# The grade itself
# --------------------------------------------------------------------------------------------


def scene_seed(scene_id: str) -> int:
    """The grain seed for a scene: stable, and different for every scene."""
    return int.from_bytes(hashlib.sha256(f"grain:{scene_id}".encode()).digest()[:8], "big")


def _luminance(x: np.ndarray) -> np.ndarray:
    return LUMA[0] * x[..., 0] + LUMA[1] * x[..., 1] + LUMA[2] * x[..., 2]


def binomial_kernel(sigma: float) -> np.ndarray:
    """A normalised 1-D Gaussian approximation built from exact integer arithmetic.

    The binomial distribution B(n, 1/2) has variance n/4, so n = 4 sigma^2 matches the
    Gaussian; the kernel is cut at three standard deviations and renormalised.
    """
    n = max(2, 2 * round(2 * sigma * sigma))
    half = n // 2
    radius = min(half, math.ceil(3 * sigma))
    coefficients = [math.comb(n, half + k) for k in range(-radius, radius + 1)]
    total = sum(coefficients)
    return np.array([c / total for c in coefficients], dtype=np.float64)


def _convolve_axis(a: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    pad = [(0, 0)] * a.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(a, pad, mode="edge")
    n = a.shape[axis]
    out = np.zeros_like(a)
    for i, weight in enumerate(kernel):
        index: list[slice] = [slice(None)] * a.ndim
        index[axis] = slice(i, i + n)
        out += weight * padded[tuple(index)]
    return out


def _downsample(a: np.ndarray, factor: int) -> np.ndarray:
    h, w = a.shape[:2]
    full_h, full_w = -(-h // factor) * factor, -(-w // factor) * factor
    if (full_h, full_w) != (h, w):
        a = np.pad(a, ((0, full_h - h), (0, full_w - w), (0, 0)), mode="edge")
    acc = np.zeros((full_h // factor, full_w // factor, a.shape[2]), dtype=np.float64)
    for dy in range(factor):
        for dx in range(factor):
            acc += a[dy::factor, dx::factor]
    return acc / (factor * factor)


def _upsample_axis(a: np.ndarray, factor: int, size: int, axis: int) -> np.ndarray:
    """Bilinear upsampling along one axis, matching pixel centres."""
    n_in = a.shape[axis]
    u = (np.arange(size, dtype=np.float64) + 0.5) / factor - 0.5
    u = np.clip(u, 0.0, float(n_in - 1))
    i0 = np.floor(u).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_in - 1)
    shape = [1] * a.ndim
    shape[axis] = size
    t = (u - i0).reshape(shape)
    return np.take(a, i0, axis=axis) * (1.0 - t) + np.take(a, i1, axis=axis) * t


def gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Blur an H x W x C float image. Large radii run on a downsample, which a glow tolerates."""
    if sigma <= 0:
        return a.copy()
    factor = 1
    while factor < 8 and sigma / (2 * factor) >= 4:
        factor *= 2
    h, w = a.shape[:2]
    small = _downsample(a, factor) if factor > 1 else a
    kernel = binomial_kernel(sigma / factor)
    blurred = _convolve_axis(_convolve_axis(small, kernel, 0), kernel, 1)
    if factor == 1:
        return blurred
    return _upsample_axis(_upsample_axis(blurred, factor, h, 0), factor, w, 1)


def grade_array(rgb: np.ndarray, params: GradeParams, seed: int) -> np.ndarray:
    """Grade an H x W x 3 uint8 array; returns a new uint8 array of the same shape."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("grade_array expects an H x W x 3 uint8 array")
    h, w = rgb.shape[:2]
    x = rgb.astype(np.float64) / 255.0

    # 1. Desaturate toward luminance.
    lum = _luminance(x)[..., None]
    x = lum + params.desaturate.saturation * (x - lum)
    x = np.clip(x, 0.0, 1.0)

    # 2. Highlight shoulder, then black lift. Above the knee k, with t = (x - k) / (1 - k), the
    #    curve is k + (1 - k)(t - t^2 / 4): same slope at the knee, 0.75 of the way up at white.
    knee = params.tone.highlight_shoulder
    if knee < 1.0:
        t = (x - knee) / (1.0 - knee)
        x = np.where(x > knee, knee + (1.0 - knee) * (t - t * t * 0.25), x)
    lift = params.tone.black_lift / 255.0
    x = lift + (1.0 - lift) * x

    # 3. Split-tone. Shadow weight (1 - L)^2 and highlight weight L^2 never add up to more
    #    than 1, so the mix stays a blend; targets are the tint colours scaled to the pixel's
    #    luminance, so only the hue moves.
    strength = params.split_tone.strength
    if strength > 0:
        lum = np.clip(_luminance(x), 0.0, 1.0)[..., None]
        shadow = np.array(params.split_tone.shadow_colour, dtype=np.float64) / 255.0
        highlight = np.array(params.split_tone.highlight_colour, dtype=np.float64) / 255.0
        shadow_target = shadow * (lum / max(float(_luminance(shadow)), 1e-6))
        highlight_target = highlight * (lum / max(float(_luminance(highlight)), 1e-6))
        ws = (1.0 - lum) * (1.0 - lum)
        wh = lum * lum
        x = x + strength * (ws * (shadow_target - x) + wh * (highlight_target - x))
        x = np.clip(x, 0.0, 1.0)

    # 4. Bloom: blur what is above the threshold, screen it back on at `mix`.
    mix = params.bloom.mix
    threshold = params.bloom.threshold
    if mix > 0 and threshold < 1.0 and params.bloom.radius > 0:
        lum = _luminance(x)
        mask = np.clip((lum - threshold) / (1.0 - threshold), 0.0, 1.0)[..., None]
        glow = gaussian_blur(x * mask, params.bloom.radius * w)
        x = 1.0 - (1.0 - x) * (1.0 - mix * glow)

    # 5. Vignette: darken by strength * r^4, where r is 1 at the corners and 0 at the centre.
    vignette = params.vignette.strength
    if vignette > 0:
        dx = (np.arange(w, dtype=np.float64) + 0.5) / w * 2.0 - 1.0
        dy = (np.arange(h, dtype=np.float64) + 0.5) / h * 2.0 - 1.0
        r2 = (dx[None, :] * dx[None, :] + dy[:, None] * dy[:, None]) * 0.5
        x = x * (1.0 - vignette * (r2 * r2))[..., None]

    # 6. Luminance grain, seeded per scene.
    sigma = params.grain.sigma
    if sigma > 0:
        noise = np.random.default_rng(seed).standard_normal((h, w), dtype=np.float64)
        x = x + (noise * (sigma / 255.0))[..., None]

    return np.rint(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def grade_image(image: Image.Image, params: GradeParams, scene_id: str) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return Image.fromarray(grade_array(rgb, params, scene_seed(scene_id)), mode="RGB")


# --------------------------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------------------------

PREVIEW_WIDTH = 1024


@dataclass
class GradeReport:
    written: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    placeholders: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    previews: list[tuple[str, str, str]] = field(default_factory=list)
    """(scene id, before path, after path), relative to assets/review."""
    preview_page: Path | None = None


def preview_dir(paths: RepoPaths) -> Path:
    return paths.review / "grade"


def run_grade(
    paths: RepoPaths,
    manifest: Manifest,
    *,
    preview: bool = False,
    loaded: LoadedParams | None = None,
) -> GradeReport:
    loaded = loaded or load_grade_params(paths)
    report = GradeReport()
    for scene in manifest.scenes:
        master_path = paths.master(scene.id)
        if not master_path.exists():
            report.missing.append(scene.id)
            continue
        placeholder = is_placeholder_image(master_path)
        if placeholder and not preview:
            report.placeholders.append(scene.id)
            continue
        with Image.open(master_path) as opened:
            master = opened.convert("RGB")
        graded = grade_image(master, loaded.params, scene.id)
        if placeholder:
            report.placeholders.append(scene.id)
        elif write_if_changed(paths.graded(scene.id), png_bytes(graded)):
            report.written.append(scene.id)
        else:
            report.unchanged.append(scene.id)
        if preview:
            folder = preview_dir(paths)
            before, after = folder / f"{scene.id}-before.webp", folder / f"{scene.id}-after.webp"
            write_if_changed(before, webp_preview(master, PREVIEW_WIDTH))
            write_if_changed(after, webp_preview(graded, PREVIEW_WIDTH))
            report.previews.append(
                (
                    scene.id,
                    before.relative_to(paths.review).as_posix(),
                    after.relative_to(paths.review).as_posix(),
                )
            )
    if preview:
        from .review import write_grade_page

        report.preview_page = write_grade_page(paths, manifest, loaded, report)
    return report

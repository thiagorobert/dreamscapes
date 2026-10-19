"""Small image helpers shared by generate, select, grade, review and bakeoff."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image

from .layout import MASTER_HEIGHT, MASTER_WIDTH


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_if_changed(path: Path, data: bytes) -> bool:
    """Write only when the bytes differ, so re-runs leave files (and git) untouched."""
    if path.exists() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def png_bytes(image: Image.Image) -> bytes:
    """Plain PNG without metadata, so the same pixels always give the same bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def as_png(data: bytes) -> tuple[bytes, tuple[int, int]]:
    """Return PNG bytes for a generated image: PNGs unchanged, anything else converted."""
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        size = image.size
        if image.format == "PNG":
            return data, size
        return png_bytes(image.convert("RGB")), size


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def fit_master(image: Image.Image) -> Image.Image:
    """Centre-crop to 16:9 and resize to the 2048x1152 master size."""
    image = image.convert("RGB")
    w, h = image.size
    target = MASTER_WIDTH / MASTER_HEIGHT
    if abs(w / h - target) > 1e-3:
        if w / h > target:
            new_w = round(h * target)
            left = (w - new_w) // 2
            image = image.crop((left, 0, left + new_w, h))
        else:
            new_h = round(w / target)
            top = (h - new_h) // 2
            image = image.crop((0, top, w, top + new_h))
    if image.size != (MASTER_WIDTH, MASTER_HEIGHT):
        image = image.resize((MASTER_WIDTH, MASTER_HEIGHT), Image.Resampling.LANCZOS)
    return image


def master_png_from(path: Path) -> bytes:
    """The master bytes a candidate becomes: 2048x1152 RGB PNG."""
    with Image.open(path) as opened:
        return png_bytes(fit_master(opened))


def jpeg_preview(path: Path, max_width: int, quality: int = 90) -> bytes:
    """A downscaled JPEG, small enough to send to a vision model or show in a review page."""
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    if image.width > max_width:
        height = round(image.height * max_width / image.width)
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def webp_preview(image: Image.Image, max_width: int, quality: int = 88) -> bytes:
    image = image.convert("RGB")
    if image.width > max_width:
        height = round(image.height * max_width / image.width)
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=quality, method=4)
    return buffer.getvalue()

"""Small image helpers shared by generate, select, grade, review and bakeoff."""

from __future__ import annotations

import hashlib
import io
import os
import uuid
from pathlib import Path

from PIL import Image

from .layout import MASTER_HEIGHT, MASTER_WIDTH


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def atomic_write(path: Path, data: bytes) -> None:
    """Write through a temporary file in the same folder, then rename it into place.

    A crash or a full disk mid-write leaves either the old file or the new one, never a
    truncated image that a later run would mistake for a finished candidate or master. The
    temporary name starts with a dot so no file pattern in the pipeline ever matches it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_if_changed(path: Path, data: bytes) -> bool:
    """Write (atomically) only when the bytes differ, so re-runs leave files and git alone."""
    if path.exists() and path.read_bytes() == data:
        return False
    atomic_write(path, data)
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


def master_crop(size: tuple[int, int]) -> tuple[int, int, int, int]:
    """The centred 16:9 box (left, top, right, bottom) that becomes the master."""
    w, h = size
    target = MASTER_WIDTH / MASTER_HEIGHT
    if abs(w / h - target) <= 1e-3:
        return (0, 0, w, h)
    if w / h > target:
        new_w = round(h * target)
        left = (w - new_w) // 2
        return (left, 0, left + new_w, h)
    new_h = round(w / target)
    top = (h - new_h) // 2
    return (0, top, w, top + new_h)


def upscale_needed(size: tuple[int, int]) -> bool:
    """True when fitting an image of this size to the master would enlarge it (and soften it)."""
    left, top, right, bottom = master_crop(size)
    return right - left < MASTER_WIDTH or bottom - top < MASTER_HEIGHT


def fit_master(image: Image.Image) -> Image.Image:
    """Centre-crop to 16:9 and resize to the 2048x1152 master size.

    Smaller images are enlarged; `dsp select` refuses that unless asked (see
    `upscale_needed`), because an enlarged master looks soft on large screens.
    """
    image = image.convert("RGB")
    box = master_crop(image.size)
    if box != (0, 0, *image.size):
        image = image.crop(box)
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

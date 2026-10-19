"""Image export helpers: responsive AVIF/WebP variants, the blurred placeholder, file naming.

`dsp placeholders` uses these today; `dsp export` must use the same functions so both commands
produce byte-identical files and therefore an identical ``scenes.json`` for the same masters.

Exported files are named ``<scene>-<width>.<hash>.<ext>`` where ``hash`` is the first eight hex
digits of the SHA-256 of the encoded bytes. The name changes whenever the bytes change, so a
browser can never show a stale image from its cache.
"""

from __future__ import annotations

import base64
import hashlib
import io
import re
from pathlib import Path

from PIL import Image, ImageFilter

from .layout import (
    EXPORT_FORMATS,
    EXPORT_WIDTHS,
    MASTER_HEIGHT,
    MASTER_WIDTH,
    RepoPaths,
    export_height,
)
from .manifest import ImageSource, ImageSources, SceneImage

WEBP_QUALITY = 82
AVIF_QUALITY = 60
AVIF_SPEED = 6
LQIP_SIZE = (32, 18)
LQIP_QUALITY = 40
HASH_LENGTH = 8


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:HASH_LENGTH]


def encode(image: Image.Image, fmt: str) -> bytes:
    """Encode with fixed settings so the same pixels always give the same bytes."""
    buffer = io.BytesIO()
    if fmt == "webp":
        image.save(buffer, format="WEBP", quality=WEBP_QUALITY, method=6)
    elif fmt == "avif":
        image.save(buffer, format="AVIF", quality=AVIF_QUALITY, speed=AVIF_SPEED)
    else:
        raise ValueError(f"unsupported export format: {fmt}")
    return buffer.getvalue()


def make_lqip(image: Image.Image) -> str:
    """A tiny blurred WebP as a data URL, painted instantly while the real image loads."""
    small = image.convert("RGB").resize(LQIP_SIZE, Image.Resampling.LANCZOS)
    small = small.filter(ImageFilter.GaussianBlur(1))
    buffer = io.BytesIO()
    small.save(buffer, format="WEBP", quality=LQIP_QUALITY, method=6)
    return "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def export_name(scene_id: str, width: int, digest: str, fmt: str) -> str:
    return f"{scene_id}-{width}.{digest}.{fmt}"


def _export_pattern(scene_id: str) -> re.Pattern[str]:
    formats = "|".join(EXPORT_FORMATS)
    return re.compile(rf"^{re.escape(scene_id)}-\d+\.[0-9a-f]{{{HASH_LENGTH}}}\.({formats})$")


def export_scene_images(paths: RepoPaths, scene_id: str, source: Path) -> SceneImage:
    """Write every variant of ``source`` (a graded master) and describe them for the manifest.

    Earlier variants of the same scene whose bytes are no longer current are removed, so the
    directory only ever holds the files the manifest points at.
    """
    with Image.open(source) as opened:
        master = opened.convert("RGB")
    if master.size != (MASTER_WIDTH, MASTER_HEIGHT):
        raise ValueError(
            f"{source} is {master.size[0]}x{master.size[1]}, "
            f"expected {MASTER_WIDTH}x{MASTER_HEIGHT}"
        )
    out_dir = paths.export_dir(scene_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources: dict[str, list[ImageSource]] = {fmt: [] for fmt in EXPORT_FORMATS}
    written: set[str] = set()
    for width in sorted(EXPORT_WIDTHS):
        size = (width, export_height(width))
        resized = master if size == master.size else master.resize(size, Image.Resampling.LANCZOS)
        for fmt in EXPORT_FORMATS:
            data = encode(resized, fmt)
            name = export_name(scene_id, width, content_hash(data), fmt)
            target = out_dir / name
            if not target.exists() or target.read_bytes() != data:
                target.write_bytes(data)
            written.add(name)
            src = target.relative_to(paths.web_public).as_posix()
            sources[fmt].append(ImageSource(width=width, src=src))

    pattern = _export_pattern(scene_id)
    for stale in out_dir.iterdir():
        if stale.is_file() and pattern.match(stale.name) and stale.name not in written:
            stale.unlink()

    return SceneImage(
        width=master.size[0],
        height=master.size[1],
        sources=ImageSources(avif=sources["avif"], webp=sources["webp"]),
        lqip=make_lqip(master),
    )

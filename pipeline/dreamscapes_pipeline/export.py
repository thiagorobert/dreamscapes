"""Image export: responsive AVIF/WebP variants, the blurred placeholder, file naming, and the
`dsp export` command that compiles ``web/public/scenes.json``.

`dsp placeholders` and `dsp export` share these functions, so both commands produce
byte-identical files and therefore an identical ``scenes.json`` for the same masters.

Exported files are named ``<scene>-<width>.<hash>.<ext>`` where ``hash`` is the first eight hex
digits of the SHA-256 of the encoded bytes. The name changes whenever the bytes change, so a
browser can never show a stale image from its cache. Pillow encodes AVIF natively.
"""

from __future__ import annotations

import base64
import hashlib
import io
import re
from dataclasses import dataclass, field
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
from .manifest import (
    BuiltManifest,
    ImageSource,
    ImageSources,
    Scene,
    SceneImage,
    build_scene,
    dump_built_manifest,
    load_manifest,
)

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


@dataclass(frozen=True)
class SceneExport:
    """Everything one scene exports, computed in memory: the files and their manifest entry."""

    scene_id: str
    image: SceneImage
    files: dict[str, bytes]
    """Encoded bytes by file name inside ``web/public/dreams/<scene>/``."""


def render_scene_exports(paths: RepoPaths, scene_id: str, source: Path) -> SceneExport:
    """Encode every variant of ``source`` (a graded master) without writing anything."""
    with Image.open(source) as opened:
        master = opened.convert("RGB")
    if master.size != (MASTER_WIDTH, MASTER_HEIGHT):
        raise ValueError(
            f"{source} is {master.size[0]}x{master.size[1]}, "
            f"expected {MASTER_WIDTH}x{MASTER_HEIGHT}"
        )
    out_dir = paths.export_dir(scene_id)
    sources: dict[str, list[ImageSource]] = {fmt: [] for fmt in EXPORT_FORMATS}
    files: dict[str, bytes] = {}
    for width in sorted(EXPORT_WIDTHS):
        size = (width, export_height(width))
        resized = master if size == master.size else master.resize(size, Image.Resampling.LANCZOS)
        for fmt in EXPORT_FORMATS:
            data = encode(resized, fmt)
            name = export_name(scene_id, width, content_hash(data), fmt)
            files[name] = data
            src = (out_dir / name).relative_to(paths.web_public).as_posix()
            sources[fmt].append(ImageSource(width=width, src=src))
    image = SceneImage(
        width=master.size[0],
        height=master.size[1],
        sources=ImageSources(avif=sources["avif"], webp=sources["webp"]),
        lqip=make_lqip(master),
    )
    return SceneExport(scene_id=scene_id, image=image, files=files)


def stale_exports(paths: RepoPaths, export: SceneExport) -> list[Path]:
    """Earlier variants of the scene that the new export no longer points at."""
    out_dir = paths.export_dir(export.scene_id)
    if not out_dir.is_dir():
        return []
    pattern = _export_pattern(export.scene_id)
    return sorted(
        p
        for p in out_dir.iterdir()
        if p.is_file() and pattern.match(p.name) and p.name not in export.files
    )


def pending_changes(paths: RepoPaths, export: SceneExport) -> list[Path]:
    """Files that writing this export would create, rewrite or delete."""
    out_dir = paths.export_dir(export.scene_id)
    changes = [
        out_dir / name
        for name, data in export.files.items()
        if not (out_dir / name).exists() or (out_dir / name).read_bytes() != data
    ]
    return changes + stale_exports(paths, export)


def write_scene_exports(paths: RepoPaths, export: SceneExport) -> list[Path]:
    """Write the export and remove stale variants; returns every path that changed.

    Afterwards the directory only holds the files the manifest points at.
    """
    out_dir = paths.export_dir(export.scene_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    changed: list[Path] = []
    for name, data in export.files.items():
        target = out_dir / name
        if not target.exists() or target.read_bytes() != data:
            target.write_bytes(data)
            changed.append(target)
    for stale in stale_exports(paths, export):
        stale.unlink()
        changed.append(stale)
    return changed


def export_scene_images(paths: RepoPaths, scene_id: str, source: Path) -> SceneImage:
    """Write every variant of ``source`` (a graded master) and describe them for the manifest.

    Earlier variants of the same scene whose bytes are no longer current are removed, so the
    directory only ever holds the files the manifest points at.
    """
    export = render_scene_exports(paths, scene_id, source)
    write_scene_exports(paths, export)
    return export.image


# --------------------------------------------------------------------------------------------
# `dsp export`
# --------------------------------------------------------------------------------------------


@dataclass
class ExportReport:
    errors: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    """Paths (relative to the repository root) written or removed, or that would be."""
    scenes: list[str] = field(default_factory=list)
    scenes_json: str = "unchanged"
    """``written``, ``unchanged``, ``would change`` (check mode) or ``skipped`` (errors)."""


def hotspot_polygon(
    paths: RepoPaths, scene: Scene
) -> tuple[list[tuple[float, float]] | None, str | None]:
    """The polygon to export for a scene, or the reason there is none.

    The authored polygon wins. A scene that still shows a placeholder master and has no polygon
    yet uses the placeholder shape's outline, exactly as `dsp placeholders` does, so both
    commands compile the same ``scenes.json``.
    """
    from .placeholders import PLACEHOLDERS_BY_ID, is_placeholder_image

    if len(scene.bridge.polygon) >= 3:
        return list(scene.bridge.polygon), None
    placeholder = PLACEHOLDERS_BY_ID.get(scene.id)
    if placeholder is not None and is_placeholder_image(paths.master(scene.id)):
        return placeholder.design.polygon(), None
    return None, "has no hotspot polygon yet; run `dsp hotspot` first"


def verify_exports(paths: RepoPaths, built: BuiltManifest) -> list[str]:
    """Every file the manifest points at exists and its bytes match the hash in its name."""
    problems = []
    for scene in built.scenes:
        for source in [*scene.image.sources.avif, *scene.image.sources.webp]:
            path = paths.public_file(source.src)
            digest = path.name.split(".")[-2] if path.name.count(".") >= 2 else ""
            if not path.exists():
                problems.append(f"{scene.id}: web/public/{source.src} is missing")
            elif content_hash(path.read_bytes()) != digest:
                problems.append(f"{scene.id}: web/public/{source.src} does not match its hash")
    return problems


def run_export(paths: RepoPaths, *, check: bool = False) -> ExportReport:
    """Compile graded masters into web exports and ``web/public/scenes.json``.

    Everything is encoded in memory first; nothing is written unless every scene can be
    exported. With ``check`` nothing is written at all and the report lists what would change.
    """
    report = ExportReport()
    manifest = load_manifest(paths.scenes_yaml)
    polygons: dict[str, list[tuple[float, float]]] = {}
    # Check every scene before encoding anything: encoding is the slow part.
    for scene in manifest.scenes:
        graded = paths.graded(scene.id)
        if not graded.exists():
            report.errors.append(
                f"{scene.id}: assets/masters/{scene.id}.graded.png is missing; run `dsp grade`"
            )
            continue
        with Image.open(graded) as opened:
            size = opened.size
        if size != (MASTER_WIDTH, MASTER_HEIGHT):
            report.errors.append(
                f"{scene.id}: assets/masters/{scene.id}.graded.png is {size[0]}x{size[1]}, "
                f"expected {MASTER_WIDTH}x{MASTER_HEIGHT}"
            )
            continue
        polygon, problem = hotspot_polygon(paths, scene)
        if polygon is None:
            report.errors.append(f"{scene.id}: {problem}")
            continue
        polygons[scene.id] = polygon
    if report.errors:
        report.scenes_json = "skipped"
        return report
    exports = {
        scene.id: render_scene_exports(paths, scene.id, paths.graded(scene.id))
        for scene in manifest.scenes
    }

    try:
        built = BuiltManifest(
            version=manifest.version,
            start=manifest.start,
            scenes=[build_scene(s, exports[s.id].image, polygons[s.id]) for s in manifest.scenes],
        )
    except ValueError as error:
        report.errors.append(f"could not build scenes.json: {error}")
        report.scenes_json = "skipped"
        return report
    text = dump_built_manifest(built).encode("utf-8")
    json_current = paths.scenes_json.exists() and paths.scenes_json.read_bytes() == text
    report.scenes = [s.id for s in manifest.scenes]

    def rel(path: Path) -> str:
        return path.relative_to(paths.root).as_posix()

    if check:
        for export in exports.values():
            report.changed.extend(rel(p) for p in pending_changes(paths, export))
        if not json_current:
            report.changed.append(rel(paths.scenes_json))
            report.scenes_json = "would change"
        return report

    for export in exports.values():
        report.changed.extend(rel(p) for p in write_scene_exports(paths, export))
    if not json_current:
        paths.scenes_json.parent.mkdir(parents=True, exist_ok=True)
        paths.scenes_json.write_bytes(text)
        report.changed.append(rel(paths.scenes_json))
        report.scenes_json = "written"
    report.errors.extend(verify_exports(paths, built))
    return report

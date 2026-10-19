"""Synthetic scenes so the website runs in a checkout with no API keys.

`dsp placeholders` paints one soft gradient per scene with a single softly glowing shape (the
stand-in bridge object), writes the matching hotspot mask, exports the responsive images and
compiles ``web/public/scenes.json``.

It is safe to re-run:

* Output is deterministic, so a second run rewrites nothing.
* Masters are tagged with a PNG text chunk. A master without that tag is real work (from
  `dsp select`) and is never overwritten unless ``--force`` is given.
* An existing ``content/scenes.yaml`` is never rewritten unless ``--force`` is given, so authored
  descriptions survive. Its titles, alt text and ``next`` pointers flow into ``scenes.json``;
  scenes whose polygon is still empty use the placeholder shape's outline instead.
* ``scenes.json`` is only written when every scene still has a placeholder master. Once real
  masters exist, compiling the manifest is `dsp export`'s job.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, PngImagePlugin

from .export import export_scene_images
from .layout import MASTER_HEIGHT, MASTER_WIDTH, RepoPaths
from .manifest import (
    Bridge,
    BuiltManifest,
    Manifest,
    Scene,
    SceneImage,
    build_scene,
    dump_built_manifest,
    dump_manifest,
    load_manifest,
)

PLACEHOLDER_KEY = "dreamscapes:placeholder"

RGB = tuple[int, int, int]

# The style palette from the plan: dusty teal, slate blue and lavender, with warm amber light.
DEEP_SLATE: RGB = (31, 42, 56)
SLATE_BLUE: RGB = (74, 90, 120)
DUSTY_TEAL: RGB = (95, 138, 139)
DEEP_TEAL: RGB = (52, 88, 92)
LAVENDER: RGB = (154, 143, 181)
DUSK_PLUM: RGB = (72, 62, 88)
MIST: RGB = (200, 206, 216)
AMBER: RGB = (224, 168, 90)
WARM_LIGHT: RGB = (246, 214, 160)
PALE_LAVENDER: RGB = (222, 212, 242)
EMBER: RGB = (232, 132, 96)


@dataclass(frozen=True)
class Design:
    """How one placeholder scene looks. Shape coordinates are in pixels of the master."""

    top: RGB
    bottom: RGB
    haze: RGB
    haze_at: tuple[float, float]
    glow: RGB
    outline: list[tuple[float, float]]

    def polygon(self) -> list[tuple[float, float]]:
        """The shape's outline in normalised coordinates, rounded to 0.2 px precision."""
        return [(round(x / MASTER_WIDTH, 4), round(y / MASTER_HEIGHT, 4)) for x, y in self.outline]


def _regular(cx: float, cy: float, radius: float, sides: int, rotation: float = 0.0):
    return [
        (
            cx * MASTER_WIDTH + radius * math.cos(rotation + 2 * math.pi * k / sides),
            cy * MASTER_HEIGHT + radius * math.sin(rotation + 2 * math.pi * k / sides),
        )
        for k in range(sides)
    ]


def _ellipse(cx: float, cy: float, rx: float, ry: float, steps: int = 32):
    return [
        (
            cx * MASTER_WIDTH + rx * math.cos(2 * math.pi * k / steps),
            cy * MASTER_HEIGHT + ry * math.sin(2 * math.pi * k / steps),
        )
        for k in range(steps)
    ]


def _rectangle(cx: float, cy: float, width: float, height: float):
    x, y = cx * MASTER_WIDTH, cy * MASTER_HEIGHT
    return [
        (x - width / 2, y - height / 2),
        (x + width / 2, y - height / 2),
        (x + width / 2, y + height / 2),
        (x - width / 2, y + height / 2),
    ]


def _star(cx: float, cy: float, outer: float, inner: float, points: int = 5):
    x, y = cx * MASTER_WIDTH, cy * MASTER_HEIGHT
    out = []
    for k in range(points * 2):
        radius = outer if k % 2 == 0 else inner
        angle = -math.pi / 2 + math.pi * k / points
        out.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))
    return out


def _triangle(cx: float, cy: float, base: float, height: float):
    x, y = cx * MASTER_WIDTH, cy * MASTER_HEIGHT
    return [
        (x, y - 2 * height / 3),
        (x + base / 2, y + height / 3),
        (x - base / 2, y + height / 3),
    ]


def _diamond(cx: float, cy: float, half_width: float, half_height: float):
    x, y = cx * MASTER_WIDTH, cy * MASTER_HEIGHT
    return [(x, y - half_height), (x + half_width, y), (x, y + half_height), (x - half_width, y)]


def _arch(cx: float, cy: float, width: float, height: float, steps: int = 12):
    """A door shape: a rectangle topped by a semicircle, ``height`` tall overall."""
    x, y = cx * MASTER_WIDTH, cy * MASTER_HEIGHT
    radius = width / 2
    bottom = y + height / 2
    spring = bottom - (height - radius)
    out = [(x - radius, bottom), (x + radius, bottom), (x + radius, spring)]
    for k in range(1, steps):
        angle = math.pi * k / steps
        out.append((x + radius * math.cos(angle), spring - radius * math.sin(angle)))
    out.append((x - radius, spring))
    return out


@dataclass(frozen=True)
class PlaceholderScene:
    id: str
    title: str
    scene: str
    bridge_object: str
    placement: str
    next: str
    design: Design


# The ten scenes of the loop. Each description mentions the previous scene's bridge object, so
# `dsp validate` is clean on placeholder content. The content work package replaces the prose.
PLACEHOLDER_SCENES: tuple[PlaceholderScene, ...] = (
    PlaceholderScene(
        id="cliff-cottage",
        title="The cliff cottage",
        scene="Placeholder. A stone cottage on a sea cliff under a storm, waves breaking far "
        "below; on a distant headland a slender lighthouse glows faintly.",
        bridge_object="lighthouse",
        placement="small, on the far headland, upper right of centre",
        next="prison-yard",
        design=Design(
            top=SLATE_BLUE,
            bottom=DUSTY_TEAL,
            haze=AMBER,
            haze_at=(0.78, 0.22),
            glow=WARM_LIGHT,
            outline=_ellipse(0.70, 0.30, 150, 150),
        ),
    ),
    PlaceholderScene(
        id="prison-yard",
        title="The yard",
        scene="Placeholder. An empty prison yard at night, floodlight towers sweeping the fog "
        "like a lighthouse beam over a cracked basketball court and chain-link fences.",
        bridge_object="basketball hoop",
        placement="small, right of centre, mid-distance",
        next="rooftop-court",
        design=Design(
            top=DEEP_SLATE,
            bottom=SLATE_BLUE,
            haze=MIST,
            haze_at=(0.45, 0.15),
            glow=MIST,
            outline=_rectangle(0.62, 0.42, 300, 250),
        ),
    ),
    PlaceholderScene(
        id="rooftop-court",
        title="The rooftop court",
        scene="Placeholder. A rooftop basketball court at blue hour above a sleeping city, one "
        "ball resting under the hoop; far off a lone hot-air balloon drifts between towers.",
        bridge_object="hot-air balloon",
        placement="small, upper left of centre, far away between two towers",
        next="balloon-salt-flats",
        design=Design(
            top=SLATE_BLUE,
            bottom=LAVENDER,
            haze=AMBER,
            haze_at=(0.30, 0.80),
            glow=AMBER,
            outline=_ellipse(0.36, 0.28, 210, 125),
        ),
    ),
    PlaceholderScene(
        id="balloon-salt-flats",
        title="The salt flats",
        scene="Placeholder. A fleet of hot-air balloons hangs motionless over a mirror-flat salt "
        "lake at dawn; on the flat stands a single free-standing wooden door.",
        bridge_object="door",
        placement="small, just right of centre, standing on the salt flat",
        next="endless-library",
        design=Design(
            top=LAVENDER,
            bottom=MIST,
            haze=AMBER,
            haze_at=(0.50, 0.90),
            glow=WARM_LIGHT,
            outline=_arch(0.55, 0.60, 200, 380),
        ),
    ),
    PlaceholderScene(
        id="endless-library",
        title="The endless library",
        scene="Placeholder. Through the door, an impossibly tall library of ladders and dusty "
        "shafts of light; on a reading table a brass birdcage stands with its door open.",
        bridge_object="birdcage",
        placement="small, left of centre, on a reading table in the middle distance",
        next="overgrown-greenhouse",
        design=Design(
            top=DUSK_PLUM,
            bottom=LAVENDER,
            haze=AMBER,
            haze_at=(0.62, 0.18),
            glow=AMBER,
            outline=_regular(0.42, 0.56, 170, 6),
        ),
    ),
    PlaceholderScene(
        id="overgrown-greenhouse",
        title="The greenhouse",
        scene="Placeholder. A Victorian greenhouse reclaimed by ferns, small birds roosting "
        "where an empty birdcage hangs; an old rusted bicycle leans against a bench.",
        bridge_object="bicycle",
        placement="small, lower left of centre, leaning against a bench",
        next="mountain-road-phonebox",
        design=Design(
            top=DUSTY_TEAL,
            bottom=DEEP_TEAL,
            haze=MIST,
            haze_at=(0.20, 0.25),
            glow=PALE_LAVENDER,
            outline=_star(0.30, 0.62, 215, 92),
        ),
    ),
    PlaceholderScene(
        id="mountain-road-phonebox",
        title="The mountain road",
        scene="Placeholder. A night mountain road, a bicycle leaning on the guardrail; at the "
        "hairpin a red telephone box glows, moths circling its light above the misty valley.",
        bridge_object="telephone box",
        placement="small, lower right of centre, at the hairpin bend",
        next="drowned-street",
        design=Design(
            top=DEEP_SLATE,
            bottom=SLATE_BLUE,
            haze=AMBER,
            haze_at=(0.70, 0.72),
            glow=EMBER,
            outline=_diamond(0.66, 0.66, 150, 215),
        ),
    ),
    PlaceholderScene(
        id="drowned-street",
        title="The drowned street",
        scene="Placeholder. A submerged city street seen from underwater, the handset of a "
        "telephone box floating on its cord; up at the surface, the dark hull of a passing ship.",
        bridge_object="ship's hull",
        placement="small, top centre, at the water surface",
        next="fog-ship-deck",
        design=Design(
            top=DUSTY_TEAL,
            bottom=DEEP_SLATE,
            haze=MIST,
            haze_at=(0.50, 0.00),
            glow=MIST,
            outline=_triangle(0.50, 0.24, 380, 300),
        ),
    ),
    PlaceholderScene(
        id="fog-ship-deck",
        title="The ship in the fog",
        scene="Placeholder. The deck of an old ship becalmed in thick fog at night, the wheel "
        "unattended above the dark hull; one paper lantern glows amber in the rigging.",
        bridge_object="paper lantern",
        placement="small, right of centre, hanging in the rigging",
        next="lighthouse-gallery",
        design=Design(
            top=MIST,
            bottom=SLATE_BLUE,
            haze=AMBER,
            haze_at=(0.76, 0.45),
            glow=AMBER,
            outline=_regular(0.74, 0.50, 170, 5, rotation=-math.pi / 2),
        ),
    ),
    PlaceholderScene(
        id="lighthouse-gallery",
        title="The lamp room",
        scene="Placeholder. Inside the lamp room of a lighthouse at dusk, the great lantern "
        "glowing; across the bay a tiny stone cottage on a cliff has one window lit.",
        bridge_object="cottage",
        placement="small, left of centre, across the bay seen through the glass",
        next="cliff-cottage",
        design=Design(
            top=LAVENDER,
            bottom=SLATE_BLUE,
            haze=AMBER,
            haze_at=(0.50, 0.55),
            glow=WARM_LIGHT,
            outline=_regular(0.27, 0.42, 165, 8, rotation=math.pi / 8),
        ),
    ),
)

PLACEHOLDERS_BY_ID = {p.id: p for p in PLACEHOLDER_SCENES}


def placeholder_manifest() -> Manifest:
    """The authored manifest that goes with the placeholder images."""
    scenes = [
        Scene(
            id=p.id,
            title=p.title,
            alt=f"Placeholder for {p.title.lower()}: a soft gradient in dusty teal and lavender.",
            scene=p.scene,
            bridge=Bridge(
                object=p.bridge_object,
                placement=p.placement,
                polygon=p.design.polygon(),
                source="placeholder",
            ),
            next=p.next,
        )
        for p in PLACEHOLDER_SCENES
    ]
    first = PLACEHOLDER_SCENES[0].id
    return Manifest(version=1, start=first, style_anchor=first, scenes=scenes)


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------


def _fill_mask(polygon_px: list[tuple[float, float]], antialias: bool) -> np.ndarray:
    shift = 4  # sixteenth-of-a-pixel vertex precision
    points = np.array(
        [[round(x * (1 << shift)), round(y * (1 << shift))] for x, y in polygon_px],
        dtype=np.int32,
    )
    mask = np.zeros((MASTER_HEIGHT, MASTER_WIDTH), dtype=np.uint8)
    line = cv2.LINE_AA if antialias else cv2.LINE_8
    cv2.fillPoly(mask, [points], 255, lineType=line, shift=shift)
    return mask


def _to_pixels(polygon: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(x * MASTER_WIDTH, y * MASTER_HEIGHT) for x, y in polygon]


def render_mask(polygon: list[tuple[float, float]]) -> Image.Image:
    """Hard-edged hotspot mask: 255 inside the polygon, 0 outside."""
    return Image.fromarray(_fill_mask(_to_pixels(polygon), antialias=False), mode="L")


def render_master(design: Design) -> Image.Image:
    """Paint the placeholder: diagonal gradient, ambient haze, and one glowing shape."""
    h, w = MASTER_HEIGHT, MASTER_WIDTH
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = xs / (w - 1), ys / (h - 1)

    top = np.array(design.top, dtype=np.float32) / 255
    bottom = np.array(design.bottom, dtype=np.float32) / 255
    s = np.clip(0.85 * v + 0.15 * u, 0, 1)
    s = s * s * (3 - 2 * s)  # smoothstep for a soft, banding-free blend
    image = top * (1 - s[..., None]) + bottom * s[..., None]

    hx, hy = design.haze_at
    dist2 = ((u - hx) * (w / h)) ** 2 + (v - hy) ** 2
    haze = 0.55 * np.exp(-dist2 / (2 * 0.35**2))
    haze_rgb = np.array(design.haze, dtype=np.float32) / 255
    image = image * (1 - haze[..., None]) + haze_rgb * haze[..., None]

    core = _fill_mask(design.outline, antialias=True).astype(np.float32) / 255
    halo = cv2.GaussianBlur(core, ksize=(0, 0), sigmaX=36)
    glow_rgb = np.array(design.glow, dtype=np.float32) / 255
    light = glow_rgb * (0.75 * halo)[..., None]
    image = 1 - (1 - image) * (1 - light)  # screen blend: light only ever brightens
    image = image * (1 - 0.8 * core[..., None]) + glow_rgb * (0.8 * core[..., None])

    pixels = np.clip(np.rint(image * 255), 0, 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def png_bytes(image: Image.Image) -> bytes:
    info = PngImagePlugin.PngInfo()
    info.add_text(PLACEHOLDER_KEY, "1")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", pnginfo=info)
    return buffer.getvalue()


def is_placeholder_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            return image.info.get(PLACEHOLDER_KEY) == "1"
    except (OSError, ValueError):
        return False


# --------------------------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------------------------


FileStatus = Literal["written", "unchanged", "kept", "skipped"]


@dataclass
class PlaceholderReport:
    rendered: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # "kept": an existing authored file was left alone; "skipped": nothing could be written.
    manifest: FileStatus = "kept"
    scenes_json: FileStatus = "skipped"


def _write_if_changed(path: Path, data: bytes) -> bool:
    if path.exists() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def run_placeholders(paths: RepoPaths, force: bool = False) -> PlaceholderReport:
    report = PlaceholderReport()
    yaml_exists = paths.scenes_yaml.exists()
    manifest = (
        load_manifest(paths.scenes_yaml) if yaml_exists and not force else placeholder_manifest()
    )

    images: dict[str, SceneImage] = {}
    polygons: dict[str, list[tuple[float, float]]] = {}
    for scene in manifest.scenes:
        master_path = paths.master(scene.id)
        owned = force or not master_path.exists() or is_placeholder_image(master_path)
        if not owned:
            report.kept.append(scene.id)
            continue
        placeholder = PLACEHOLDERS_BY_ID.get(scene.id)
        if placeholder is None:
            report.notes.append(f"{scene.id}: no placeholder design for this scene; skipped")
            continue

        polygon = (
            list(scene.bridge.polygon)
            if len(scene.bridge.polygon) >= 3
            else placeholder.design.polygon()
        )
        master = png_bytes(render_master(placeholder.design))
        _write_if_changed(master_path, master)
        _write_if_changed(paths.graded(scene.id), master)
        _write_if_changed(paths.mask(scene.id), png_bytes(render_mask(polygon)))
        images[scene.id] = export_scene_images(paths, scene.id, paths.graded(scene.id))
        polygons[scene.id] = polygon
        report.rendered.append(scene.id)

    if not yaml_exists or force:
        changed = _write_if_changed(paths.scenes_yaml, dump_manifest(manifest).encode("utf-8"))
        report.manifest = "written" if changed else "unchanged"

    if len(images) == len(manifest.scenes):
        built = BuiltManifest(
            version=manifest.version,
            start=manifest.start,
            scenes=[build_scene(s, images[s.id], polygons[s.id]) for s in manifest.scenes],
        )
        changed = _write_if_changed(paths.scenes_json, dump_built_manifest(built).encode("utf-8"))
        report.scenes_json = "written" if changed else "unchanged"
    else:
        missing = [s.id for s in manifest.scenes if s.id not in images]
        report.notes.append(
            "left web/public/scenes.json alone because these scenes have real or missing "
            f"placeholder masters: {', '.join(missing)}; run `dsp export` instead"
        )
    return report

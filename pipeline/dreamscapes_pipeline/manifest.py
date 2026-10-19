"""Pydantic models for the two scene manifests, plus their loaders and savers.

* The authored manifest, ``content/scenes.yaml``, is written by people and by pipeline commands.
  It holds prompts, bridge objects, hotspot polygons (possibly still empty) and ``next`` pointers.
* The built manifest, ``web/public/scenes.json``, is compiled from the authored one and is the
  only thing the website reads. It adds image sources, a blurred placeholder and the hotspot
  geometry derived from the polygon, and it drops authoring-only fields.

These models are the source of truth for both formats; ``dsp schema`` exports the built one as
JSON Schema for the website.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from . import geometry

SLUG_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"

Slug = Annotated[str, StringConstraints(pattern=SLUG_PATTERN)]
"""Lower-case words joined by single hyphens, e.g. ``prison-yard``."""

Unit = Annotated[float, Field(ge=0.0, le=1.0)]
Point = tuple[Unit, Unit]
"""An ``[x, y]`` pair in normalised image coordinates, origin top-left."""

BBox = tuple[Unit, Unit, Unit, Unit]
"""An ``[x0, y0, x1, y1]`` axis-aligned box in normalised image coordinates."""

NonEmptyStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

HotspotSource = Literal["gemini", "sam3", "manual", "placeholder"]


class _Strict(BaseModel):
    """Unknown keys are rejected so that typos in hand-edited YAML fail loudly."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------------------------
# Authored manifest: content/scenes.yaml
# --------------------------------------------------------------------------------------------


class Bridge(_Strict):
    """The single secretly clickable object in a scene."""

    object: NonEmptyStr = Field(
        description="Short noun phrase naming the object; also the segmentation prompt."
    )
    placement: NonEmptyStr = Field(
        description="Where the object sits in the frame and how big it is, for the image prompt."
    )
    polygon: list[Point] = Field(
        default_factory=list,
        description="Hotspot outline in normalised coordinates. Empty until `dsp hotspot` runs.",
    )
    source: HotspotSource | None = Field(
        default=None, description="Which tool produced the polygon."
    )
    verified: bool = Field(
        default=False,
        description="True once Claude (or a person) confirmed the polygon.",
    )
    verification: str | None = Field(
        default=None, description="Reason given by the verifier, kept for review."
    )
    verified_master_sha256: Sha256 | None = Field(
        default=None,
        description="SHA-256 of assets/masters/<id>.png when `dsp hotspot` verified the polygon. "
        "If the master changes afterwards (for example `dsp select` adopts another candidate), "
        "the polygon may no longer fit it, so `dsp hotspot --all` treats the scene as unverified.",
    )
    score: float | None = Field(default=None, description="Segmenter confidence, if reported.")
    needs_manual_hotspot: bool = Field(
        default=False,
        description="Set when automatic extraction failed twice and a person must annotate it.",
    )
    # Informational copies of the derived geometry. `dsp export` always recomputes them from the
    # polygon; `dsp validate` warns when they have drifted.
    bbox: BBox | None = None
    centroid: Point | None = None
    area: float | None = Field(default=None, ge=0.0, le=1.0)


class Generation(_Strict):
    """How the scene's master image was made."""

    provider: str | None = None
    model: str | None = None
    selected_candidate: int | None = Field(default=None, ge=0)
    log_ids: list[str] = Field(
        default_factory=list, description="Row ids in pipeline/logs/generations.jsonl."
    )


class Transition(_Strict):
    """How the site moves from this scene to its ``next`` scene."""

    type: Literal["zoom", "video"] = "zoom"
    video: str | None = Field(
        default=None,
        description="Clip path relative to web/public, e.g. morphs/a__b.mp4.",
    )
    duration: float | None = Field(default=None, gt=0.0, description="Clip length in seconds.")

    @model_validator(mode="after")
    def _video_needs_clip(self) -> Transition:
        if self.type == "video" and (not self.video or self.duration is None):
            raise ValueError("a video transition needs both 'video' and 'duration'")
        return self


class Scene(_Strict):
    id: Slug
    title: NonEmptyStr
    alt: NonEmptyStr = Field(description="Image alt text; describes the scene, not the hotspot.")
    scene: NonEmptyStr = Field(description="Scene description used in the image prompt.")
    bridge: Bridge
    next: Slug
    focal: Point | None = Field(
        default=None,
        description="Pan focus on narrow screens; defaults to the hotspot centroid.",
    )
    generation: Generation | None = None
    transition: Transition = Field(default_factory=Transition)

    @field_validator("bridge", mode="before")
    @classmethod
    def _exactly_one_bridge(cls, value: Any) -> Any:
        if isinstance(value, list):
            raise ValueError(
                f"exactly one bridge object per scene is supported, found a list of {len(value)}"
            )
        return value


class Manifest(_Strict):
    """The authored loop."""

    version: int = Field(default=1, ge=1)
    start: Slug = Field(description="Scene shown when the site opens without a hash.")
    style_anchor: Slug = Field(description="Scene whose master is the style reference.")
    scenes: list[Scene] = Field(min_length=1)

    def scene(self, scene_id: str) -> Scene:
        for scene in self.scenes:
            if scene.id == scene_id:
                return scene
        raise KeyError(scene_id)

    def by_id(self) -> dict[str, Scene]:
        return {scene.id: scene for scene in self.scenes}


# --------------------------------------------------------------------------------------------
# Built manifest: web/public/scenes.json
# --------------------------------------------------------------------------------------------


class ImageSource(_Strict):
    width: int = Field(gt=0, description="Pixel width of this file.")
    src: str = Field(
        min_length=1,
        description="Path relative to the site root, e.g. dreams/a/a-1024.1f2e3d4c.webp.",
    )


class ImageSources(_Strict):
    """Files per format, sorted by ascending width."""

    avif: list[ImageSource]
    webp: list[ImageSource] = Field(min_length=1)


class SceneImage(_Strict):
    width: int = Field(gt=0, description="Width of the master the exports were made from.")
    height: int = Field(gt=0, description="Height of the master the exports were made from.")
    sources: ImageSources
    lqip: str = Field(
        pattern=r"^data:image/(webp|png|jpeg|avif);base64,[A-Za-z0-9+/=]+$",
        description="Tiny blurred placeholder painted before the full image loads.",
    )


class BuiltBridge(_Strict):
    object: str
    polygon: list[Point] = Field(min_length=3)
    bbox: BBox
    centroid: Point
    area: float = Field(ge=0.0, le=1.0, description="Fraction of the frame the polygon covers.")


class ZoomTransition(_Strict):
    type: Literal["zoom"]


class VideoTransition(_Strict):
    type: Literal["video"]
    video: str = Field(min_length=1, description="Clip path relative to the site root.")
    duration: float = Field(gt=0.0, description="Clip length in seconds.")


class BuiltScene(_Strict):
    id: Slug
    title: str
    alt: str
    next: Slug
    focal: Point = Field(description="Resolved pan focus (the hotspot centroid by default).")
    bridge: BuiltBridge
    image: SceneImage
    transition: ZoomTransition | VideoTransition


class BuiltManifest(_Strict):
    """What the website loads."""

    model_config = ConfigDict(extra="forbid", title="Dreamscapes built manifest")

    version: int = Field(ge=1)
    start: Slug
    scenes: list[BuiltScene] = Field(min_length=1)


# --------------------------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------------------------

_PRECISION = 6


def _round_point(point: tuple[float, float]) -> tuple[float, float]:
    return (round(point[0], _PRECISION), round(point[1], _PRECISION))


def build_bridge(bridge: Bridge, polygon: list[tuple[float, float]] | None = None) -> BuiltBridge:
    """Derive the hotspot geometry from the authored polygon (or an override)."""
    points = list(polygon if polygon is not None else bridge.polygon)
    if len(points) < 3:
        raise ValueError(f"bridge '{bridge.object}' has no usable polygon yet")
    x0, y0, x1, y1 = geometry.bbox(points)
    return BuiltBridge(
        object=bridge.object,
        polygon=[_round_point(p) for p in points],
        bbox=(
            round(x0, _PRECISION),
            round(y0, _PRECISION),
            round(x1, _PRECISION),
            round(y1, _PRECISION),
        ),
        centroid=_round_point(geometry.centroid(points)),
        area=round(geometry.area(points), _PRECISION),
    )


def build_scene(
    scene: Scene, image: SceneImage, polygon: list[tuple[float, float]] | None = None
) -> BuiltScene:
    """Compile one authored scene plus its exported image into the website's format."""
    bridge = build_bridge(scene.bridge, polygon)
    transition: ZoomTransition | VideoTransition
    if scene.transition.type == "video":
        assert scene.transition.video is not None and scene.transition.duration is not None
        transition = VideoTransition(
            type="video",
            video=scene.transition.video,
            duration=scene.transition.duration,
        )
    else:
        transition = ZoomTransition(type="zoom")
    return BuiltScene(
        id=scene.id,
        title=scene.title,
        alt=scene.alt,
        next=scene.next,
        focal=_round_point(scene.focal) if scene.focal is not None else bridge.centroid,
        bridge=bridge,
        image=image,
        transition=transition,
    )


# --------------------------------------------------------------------------------------------
# Loading and saving
# --------------------------------------------------------------------------------------------


class ManifestError(ValueError):
    """A manifest file could not be read; ``problems`` lists every readable reason."""

    def __init__(self, path: Path, problems: list[str]):
        self.path = path
        self.problems = problems
        super().__init__(f"{path}: " + "; ".join(problems))


def _describe_location(loc: tuple[Any, ...], raw: Any) -> str:
    parts: list[str] = []
    node = raw
    for item in loc:
        if isinstance(item, int):
            label = f"[{item}]"
            if parts and parts[-1] == "scenes" and isinstance(node, list) and item < len(node):
                scene_id = node[item].get("id") if isinstance(node[item], dict) else None
                if scene_id:
                    label = f"[{item}] ({scene_id})"
            parts.append(label)
        else:
            parts.append(("." if parts else "") + str(item))
        try:
            node = node[item]
        except (KeyError, IndexError, TypeError):
            node = None
    return "".join(parts) or "(top level)"


def _validation_problems(error: ValidationError, raw: Any) -> list[str]:
    return [f"{_describe_location(e['loc'], raw)}: {e['msg']}" for e in error.errors()]


def parse_manifest(raw: Any, path: Path = Path("content/scenes.yaml")) -> Manifest:
    try:
        return Manifest.model_validate(raw)
    except ValidationError as error:
        raise ManifestError(path, _validation_problems(error, raw)) from error


def load_manifest(path: Path) -> Manifest:
    """Read and validate ``content/scenes.yaml``."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ManifestError(path, ["file not found"]) from error
    except yaml.YAMLError as error:
        raise ManifestError(path, [f"invalid YAML: {error}"]) from error
    return parse_manifest(raw, path)


def load_built_manifest(path: Path) -> BuiltManifest:
    """Read and validate ``web/public/scenes.json``."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ManifestError(path, ["file not found"]) from error
    except json.JSONDecodeError as error:
        raise ManifestError(path, [f"invalid JSON: {error}"]) from error
    try:
        return BuiltManifest.model_validate(raw)
    except ValidationError as error:
        raise ManifestError(path, _validation_problems(error, raw)) from error


class _ManifestDumper(yaml.SafeDumper):
    """Readable YAML: indented lists, numeric pairs on one line, long prose folded."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        super().increase_indent(flow, False)


def _represent_list(dumper: yaml.SafeDumper, data: list[Any]) -> yaml.Node:
    numeric = bool(data) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in data
    )
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=numeric)


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.Node:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    if len(data) > 80:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=">")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_ManifestDumper.add_representer(list, _represent_list)
_ManifestDumper.add_representer(str, _represent_str)


def dump_manifest(manifest: Manifest) -> str:
    data = manifest.model_dump(mode="json", exclude_none=True)
    return yaml.dump(data, Dumper=_ManifestDumper, sort_keys=False, allow_unicode=True, width=96)


def save_manifest(manifest: Manifest, path: Path) -> None:
    """Write ``content/scenes.yaml``. Comments in a hand-edited file are not preserved."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_manifest(manifest), encoding="utf-8")


# Matches an array of numbers as json.dumps(indent=2) lays it out: one element per line. JSON
# strings cannot contain a raw newline, so anchoring on the newlines guarantees this never
# touches bracketed numbers inside a title or alt text.
_NUMERIC_ARRAY = re.compile(r"\[\n\s*(-?[\d.eE+-]+(?:,\n\s*-?[\d.eE+-]+)*)\n\s*\]")


def dump_built_manifest(built: BuiltManifest) -> str:
    text = json.dumps(built.model_dump(mode="json"), indent=2, ensure_ascii=False)
    # Keep coordinate pairs and boxes on one line so diffs stay readable.
    text = _NUMERIC_ARRAY.sub(
        lambda m: "[" + ", ".join(v.strip() for v in m.group(1).split(",")) + "]", text
    )
    return text + "\n"


def save_built_manifest(built: BuiltManifest, path: Path) -> None:
    """Write ``web/public/scenes.json``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_built_manifest(built), encoding="utf-8")

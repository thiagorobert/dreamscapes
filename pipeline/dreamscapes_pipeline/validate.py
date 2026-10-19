"""`dsp validate`: every structural rule the loop must satisfy.

Errors make the command exit non-zero. Warnings flag soft rules and work that is still pending
(for example a polygon that `dsp hotspot` has not filled yet) without failing the run.

Rules:

* Authored manifest parses; scene ids are unique; ``start`` and ``style_anchor`` exist.
* Every scene has exactly one bridge object, and its ``next`` names an existing scene.
* The ``next`` pointers form one single loop that passes through every scene.
* Each hotspot polygon has at least 3 vertices, does not cross itself, covers 1.5-12% of the
  frame, and has its centroid inside the central 60% of the width and 80% of the height.
  A missing polygon is a warning.
* A scene's description mentions the previous scene's bridge object (warning).
* Referenced files exist with the expected dimensions: masters, graded masters, masks,
  transition clips, and every image the built manifest points at.
* The built manifest matches the authored one (warning when it is out of date).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from PIL import Image

from . import geometry
from .layout import MASTER_HEIGHT, MASTER_WIDTH, RepoPaths, export_height
from .manifest import (
    BuiltManifest,
    Manifest,
    ManifestError,
    Transition,
    VideoTransition,
    load_built_manifest,
    load_manifest,
)

AREA_MIN = 0.015
AREA_MAX = 0.12
SAFE_X = (0.2, 0.8)  # central 60% of the width
SAFE_Y = (0.1, 0.9)  # central 80% of the height
DERIVED_TOLERANCE = 1e-4

Level = Literal["error", "warning"]


@dataclass(frozen=True)
class Issue:
    level: Level
    message: str
    scene: str | None = None

    def __str__(self) -> str:
        where = f"{self.scene}: " if self.scene else ""
        return f"{self.level:<7} {where}{self.message}"


@dataclass
class ValidationResult:
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    def error(self, message: str, scene: str | None = None) -> None:
        self.issues.append(Issue("error", message, scene))

    def warning(self, message: str, scene: str | None = None) -> None:
        self.issues.append(Issue("warning", message, scene))


# --------------------------------------------------------------------------------------------
# Manifest rules (no file access)
# --------------------------------------------------------------------------------------------


def _check_graph(manifest: Manifest, result: ValidationResult) -> None:
    ids = [s.id for s in manifest.scenes]
    known = set(ids)
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    for dup in duplicates:
        result.error(f"scene id is used {ids.count(dup)} times; ids must be unique", dup)
    if manifest.start not in known:
        result.error(f"start scene '{manifest.start}' does not exist")
    if manifest.style_anchor not in known:
        result.error(f"style_anchor scene '{manifest.style_anchor}' does not exist")
    dangling = False
    for scene in manifest.scenes:
        if scene.next not in known:
            result.error(f"next scene '{scene.next}' does not exist", scene.id)
            dangling = True
    if duplicates or dangling:
        return  # the walk below is meaningless until those are fixed

    following = {s.id: s.next for s in manifest.scenes}
    origin = manifest.start if manifest.start in known else ids[0]
    walk: list[str] = []
    current = origin
    while current not in walk:
        walk.append(current)
        current = following[current]
    missed = [i for i in ids if i not in walk]
    if missed or current != origin:
        path = " -> ".join([*walk, current])
        detail = f"; never reaches {', '.join(missed)}" if missed else ""
        result.error(
            f"next pointers must form one loop through all {len(ids)} scenes, but from "
            f"'{origin}' the walk goes {path}{detail}"
        )


def _check_polygon(manifest: Manifest, result: ValidationResult) -> None:
    for scene in manifest.scenes:
        bridge = scene.bridge
        if bridge.needs_manual_hotspot:
            result.warning("hotspot is marked as needing manual annotation", scene.id)
        polygon = bridge.polygon
        if not polygon:
            result.warning(
                f"bridge object '{bridge.object}' has no polygon yet (run dsp hotspot)", scene.id
            )
            continue
        problem = geometry.simplicity_problem(polygon)
        if problem is not None:
            result.error(f"hotspot polygon is not a simple polygon: {problem}", scene.id)
            continue
        area = geometry.area(polygon)
        if not AREA_MIN <= area <= AREA_MAX:
            result.error(
                f"hotspot covers {area:.2%} of the frame; it must cover between "
                f"{AREA_MIN:.1%} and {AREA_MAX:.0%}",
                scene.id,
            )
        cx, cy = geometry.centroid(polygon)
        if not (SAFE_X[0] <= cx <= SAFE_X[1] and SAFE_Y[0] <= cy <= SAFE_Y[1]):
            result.error(
                f"hotspot centroid ({cx:.3f}, {cy:.3f}) is outside the safe area "
                f"(x {SAFE_X[0]}-{SAFE_X[1]}, y {SAFE_Y[0]}-{SAFE_Y[1]})",
                scene.id,
            )
        stale = []
        if bridge.area is not None and abs(bridge.area - area) > DERIVED_TOLERANCE:
            stale.append("area")
        if bridge.centroid is not None and _differs(bridge.centroid, (cx, cy)):
            stale.append("centroid")
        if bridge.bbox is not None and _differs(bridge.bbox, geometry.bbox(polygon)):
            stale.append("bbox")
        if stale:
            result.warning(
                f"stored {', '.join(stale)} no longer match the polygon; they are recomputed "
                "on export, so delete or refresh them",
                scene.id,
            )


def _differs(stored: tuple[float, ...], computed: tuple[float, ...]) -> bool:
    return any(abs(a - b) > DERIVED_TOLERANCE for a, b in zip(stored, computed, strict=True))


_ARTICLE = re.compile(r"^(the|a|an)\s+")
_POSSESSIVE = re.compile(r"['’]s$")


def mentions(description: str, bridge_object: str) -> bool:
    """Loose check that a description refers to an object.

    Passes when the whole phrase appears, or when its head noun (the last word, without a
    possessive) appears at the start of a word, so "balloon" matches "balloons".
    """
    text = description.lower()
    phrase = _ARTICLE.sub("", bridge_object.lower().strip())
    if phrase in text:
        return True
    head = _POSSESSIVE.sub("", phrase.split()[-1]) if phrase.split() else ""
    return bool(head) and re.search(rf"\b{re.escape(head)}", text) is not None


def _check_mentions(manifest: Manifest, result: ValidationResult) -> None:
    by_id = manifest.by_id()
    for previous in manifest.scenes:
        following = by_id.get(previous.next)
        if following is None or following.id == previous.id:
            continue
        if not mentions(following.scene, previous.bridge.object):
            result.warning(
                f"description does not mention '{previous.bridge.object}', the bridge object "
                f"of the previous scene '{previous.id}'",
                following.id,
            )


def check_manifest(manifest: Manifest) -> ValidationResult:
    """All rules that only need the authored manifest itself."""
    result = ValidationResult()
    _check_graph(manifest, result)
    _check_polygon(manifest, result)
    _check_mentions(manifest, result)
    return result


# --------------------------------------------------------------------------------------------
# File rules
# --------------------------------------------------------------------------------------------


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            return image.size
    except (OSError, ValueError):
        return None


def _check_image(
    result: ValidationResult,
    path: Path,
    expected: tuple[int, int],
    label: str,
    scene: str | None,
    root: Path,
    required: bool,
) -> None:
    rel = path.relative_to(root) if path.is_relative_to(root) else path
    if not path.exists():
        if required:
            result.error(f"{label} {rel} is missing", scene)
        return
    size = _image_size(path)
    if size is None:
        result.error(f"{label} {rel} is not a readable image", scene)
    elif size != expected:
        result.error(
            f"{label} {rel} is {size[0]}x{size[1]}, expected {expected[0]}x{expected[1]}", scene
        )


def check_files(manifest: Manifest, paths: RepoPaths, result: ValidationResult) -> None:
    master_size = (MASTER_WIDTH, MASTER_HEIGHT)
    for scene in manifest.scenes:
        sid = scene.id
        _check_image(result, paths.master(sid), master_size, "master", sid, paths.root, True)
        _check_image(
            result, paths.graded(sid), master_size, "graded master", sid, paths.root, False
        )
        _check_image(result, paths.mask(sid), master_size, "mask", sid, paths.root, False)
        if scene.transition.type == "video" and scene.transition.video:
            clip = paths.public_file(scene.transition.video)
            if not clip.exists():
                result.error(f"transition clip web/public/{scene.transition.video} is missing", sid)


def _built_drift(manifest: Manifest, built: BuiltManifest) -> list[str]:
    reasons: list[str] = []
    if built.start != manifest.start:
        reasons.append("start scene differs")
    authored = manifest.by_id()
    compiled = {s.id: s for s in built.scenes}
    if set(authored) != set(compiled):
        reasons.append("scene ids differ")
    for sid, scene in authored.items():
        other = compiled.get(sid)
        if other is None:
            continue
        authored_clip = _clip(scene.transition)
        built_clip = (
            (other.transition.video, other.transition.duration)
            if isinstance(other.transition, VideoTransition)
            else (None, None)
        )
        changed = [
            name
            for name, a, b in (
                ("next", scene.next, other.next),
                ("title", scene.title, other.title),
                ("alt", scene.alt, other.alt),
                ("bridge object", scene.bridge.object, other.bridge.object),
                ("transition", scene.transition.type, other.transition.type),
                ("transition clip", authored_clip, built_clip),
            )
            if a != b
        ]
        has_polygon = len(scene.bridge.polygon) >= 3
        if has_polygon and _polygon_differs(scene.bridge.polygon, other.bridge.polygon):
            changed.append("polygon")
        # An unset focal means "the hotspot centroid", exactly as the build resolves it. While
        # the authored polygon is still empty the build uses a stand-in polygon, whose centroid
        # is the built one.
        if scene.focal is not None:
            expected_focal = scene.focal
        elif has_polygon:
            expected_focal = geometry.centroid(scene.bridge.polygon)
        else:
            expected_focal = other.bridge.centroid
        if _differs(expected_focal, other.focal):
            changed.append("focal")
        if changed:
            reasons.append(f"{sid} ({', '.join(changed)})")
    return reasons


def _clip(transition: Transition) -> tuple[str | None, float | None]:
    """The clip a transition plays, or (None, None) for the zoom transition."""
    if transition.type != "video":
        return (None, None)
    return (transition.video, transition.duration)


def _polygon_differs(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
    if len(a) != len(b):
        return True
    return any(_differs(p, q) for p, q in zip(a, b, strict=True))


def check_built(manifest: Manifest, paths: RepoPaths, result: ValidationResult) -> None:
    if not paths.scenes_json.exists():
        result.warning("web/public/scenes.json does not exist yet (run dsp export)")
        return
    try:
        built = load_built_manifest(paths.scenes_json)
    except ManifestError as error:
        for problem in error.problems:
            result.error(f"web/public/scenes.json: {problem}")
        return

    for scene in built.scenes:
        if (scene.image.width, scene.image.height) != (MASTER_WIDTH, MASTER_HEIGHT):
            result.error(
                f"built image is {scene.image.width}x{scene.image.height}, expected "
                f"{MASTER_WIDTH}x{MASTER_HEIGHT}",
                scene.id,
            )
        for fmt, sources in (
            ("avif", scene.image.sources.avif),
            ("webp", scene.image.sources.webp),
        ):
            for source in sources:
                expected = (source.width, export_height(source.width))
                path = paths.public_file(source.src)
                if not path.name.endswith(f".{fmt}"):
                    result.error(f"{fmt} source {source.src} has the wrong extension", scene.id)
                _check_image(result, path, expected, "export", scene.id, paths.root, True)
        authored = manifest.by_id().get(scene.id)
        authored_clip = authored.transition.video if authored else None
        # Clips the authored manifest also names are already checked by check_files.
        if (
            scene.transition.type == "video"
            and scene.transition.video != authored_clip
            and not paths.public_file(scene.transition.video).exists()
        ):
            result.error(
                f"transition clip web/public/{scene.transition.video} is missing", scene.id
            )

    drift = _built_drift(manifest, built)
    if drift:
        result.warning(
            "web/public/scenes.json is out of date with content/scenes.yaml (run dsp export): "
            + "; ".join(drift)
        )


def validate_repo(paths: RepoPaths) -> ValidationResult:
    try:
        manifest = load_manifest(paths.scenes_yaml)
    except ManifestError as error:
        result = ValidationResult()
        for problem in error.problems:
            result.error(f"content/scenes.yaml: {problem}")
        return result
    result = check_manifest(manifest)
    check_files(manifest, paths, result)
    check_built(manifest, paths, result)
    return result

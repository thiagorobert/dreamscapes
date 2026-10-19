"""`dsp hotspot`: turn each scene's bridge object into its clickable polygon.

For one scene (plan section 6.2):

1. Segment the bridge object in the graded master (the master when no graded copy exists) with
   a `SegmentationProvider`: Gemini by default, SAM 3 on fal with
   ``DSP_SEGMENTATION_PROVIDER=sam3``. A graded copy older than its master is refused
   (`stale_graded_copy`): it may still show the previous master.
2. When several instances come back, keep the largest one whose centroid is inside the safe area.
3. Clean the outline (`clean_polygon`): rasterise it, keep the largest connected component, grow
   it by about 0.5% of the image width so the edges are forgiving, and simplify it to at most 40
   vertices.
4. Ask Claude whether the tinted region is exactly the object and nothing else
   (`verify_with_claude`). On "no", segment once more with the bounding box Claude suggests
   written into the prompt (only when there is a new box: otherwise the retry would repeat the
   same request); on a second "no", mark the scene ``needs_manual_hotspot`` so a person draws it
   with tools/annotate/index.html.
5. When verified and within the hotspot rules of `dsp validate`, write the polygon, its derived
   geometry, the verdict and what it was verified against (the master's hash, and the path and
   hash of the picture shown) back to content/scenes.yaml (only the hotspot fields; comments
   and everything else are kept) and the mask to assets/masks/<scene>.png. When not, write
   only ``verified: false``, the reasons and ``needs_manual_hotspot: true``: the existing
   polygon, its source and its mask stay as they were. Either way, save review overlays under
   assets/review/<scene>/ (hotspot.png is the hotspot the manifest now holds,
   hotspot-<n>-<reading>.png each attempt, named after how the mask points were read) and
   report the hotspot rules that `dsp validate` enforces.

What was checked against the live documentation on 2026-10-19
---------------------------------------------------------------

Gemini, https://ai.google.dev/gemini-api/docs/image-understanding (segmentation section):

* Model and call: the segmentation examples use ``gemini-3.8-flash`` through
  ``client.interactions.create`` (google-genai 2.31), with
  ``response_format={"type": "text", "mime_type": "application/json", "schema": ...}`` and
  ``generation_config={"thinking_level": "minimal"}``.
* Thinking: "For better results, disable thinking by setting the thinking level to 'minimal'."
  The live API rejects that for gemini-3.8-flash (400: "Thinking level THINKING_LEVEL_MINIMAL is
  not supported for this model"), so requests use "low", the lowest level it accepts, and step up
  automatically if a level is ever refused.
* Prompt: "Give the segmentation masks for the ... Output a JSON list of segmentation masks where
  each entry contains the 2D bounding box in the key "box_2d", the segmentation mask in key
  "mask", and the text label in the key "label". Use descriptive labels." `GEMINI_PROMPT` is that
  wording with the bridge object filled in; the response schema is the page's own.
* Response: ``box_2d`` is ``[ymin, xmin, ymax, xmax]`` normalised to 0-1000, and ``mask`` is "the
  segmentation mask inside the bounding box as a polygon of [x, y] coordinates normalized to
  0-1000". The page does not say whether the mask's 0-1000 span the box or the whole image.
  The real-API smoke test found the points 0-1000 across the whole image, in [x, y] order (a
  telephone box with box_2d [409, 569, 746, 665] came back with mask x in 571-664 and y in
  409-746). The first ten real scenes then showed that the order is not reliable: five of them
  came back with every mask point as [y, x] while box_2d kept its documented order, and a few
  masks switch from one order to the other part-way round the outline. So each entry is read
  all four ways ([x, y] or [y, x], across the image or across the box), and the reading that
  keeps the outline inside its own box and reaching across it wins (`choose_mask_reading`).
* Pricing, https://ai.google.dev/gemini-api/docs/pricing: $0.75 per million input tokens and
  $3.75 per million output tokens (thinking included) until 2026-12-31.

Claude, from the bundled claude-api skill: model ``claude-opus-5-5``; adaptive thinking by leaving
``thinking`` out (it cannot be disabled on this model); an explicit ``output_config.effort``
because this model's default is ``medium``; the answer constrained by a JSON schema in
``output_config.format`` and validated with pydantic; and server-side fallbacks
(``fallbacks="default"`` with the ``server-side-fallback-2026-07-01`` beta), so a declined request
is re-run on the model Anthropic recommends for that refusal category. $4 / $20 per million input
/ output tokens.

SAM 3 on fal, https://fal.ai/models/fal-ai/sam-3/image/api: ``image_url`` and ``prompt`` in
(``box_prompts`` optional), raster ``masks`` out with optional ``scores`` and ``boxes``
(``[cx, cy, w, h]`` normalised). Optional: ``uv sync --extra sam3`` and ``FAL_KEY``. Not exercised
against the live API because no fal account exists yet; the page lists no price.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import cv2
import numpy as np
import yaml
from PIL import Image
from pydantic import BaseModel, Field, ValidationError, field_validator

from . import genlog, geometry
from .imaging import sha256_file
from .layout import MASTER_HEIGHT, MASTER_WIDTH, RepoPaths
from .manifest import (
    Bridge,
    HotspotSource,
    Manifest,
    ManifestError,
    Scene,
    load_manifest,
    parse_manifest,
)
from .validate import SAFE_X, SAFE_Y, Issue, ValidationResult, _check_polygon
from .yaml_patch import render_fields, set_scene_fields

Polygon = list[tuple[float, float]]
"""Vertices in normalised image coordinates: x and y in 0-1, origin top-left."""

BBox = tuple[float, float, float, float]
"""``(x0, y0, x1, y1)`` in normalised image coordinates."""

Echo = Callable[[str], None]

# Polygon cleaning
MAX_VERTICES = 40
DILATE_FRACTION = 0.005  # of the image width: about 10 px on a 2048 px master
POLYGON_DECIMALS = 4  # 0.2 px on a 2048 px master, the precision the placeholders use
DERIVED_DECIMALS = 6  # bbox, centroid and area, as `dsp export` rounds them
RAW_VERTICES = 400  # budget for faithful outlines of raster masks before cleaning
_SHIFT = 4  # sixteenth-of-a-pixel vertex precision when rasterising

# Gemini
GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_INPUT_USD_PER_MTOK = 0.75
GEMINI_OUTPUT_USD_PER_MTOK = 3.75
GEMINI_RETRY_ATTEMPTS = 4
# The docs advise "minimal" for segmentation, but gemini-3.8-flash answers that with a 400
# ("Thinking level THINKING_LEVEL_MINIMAL is not supported for this model"), so the request
# starts at the lowest level the model accepts and steps up if a level is ever refused.
GEMINI_THINKING_LEVELS = ("minimal", "low", "medium", "high")
GEMINI_THINKING_LEVEL = "low"

# Claude
CLAUDE_MODEL = "claude-opus-5-5"
CLAUDE_EFFORT = "high"  # the verdict decides whether a person has to draw the hotspot
CLAUDE_MAX_TOKENS = 16000
CLAUDE_BETAS = ["server-side-fallback-2026-07-01"]
CLAUDE_INPUT_USD_PER_MTOK = 4.0
CLAUDE_OUTPUT_USD_PER_MTOK = 20.0
CLAUDE_IMAGE_LONG_EDGE = 1568
CLOSEUP_LONG_EDGE = 1024

# SAM 3 on fal
SAM3_APP = "fal-ai/sam-3/image"

# Worst-case cost of one call, used by the spend guard before anything is sent. Real costs are
# computed from the token usage each response reports.
GEMINI_CALL_ESTIMATE_USD = 0.01
SAM3_CALL_ESTIMATE_USD = 0.02
CLAUDE_CALL_ESTIMATE_USD = 0.09
DEFAULT_MAX_SPEND_USD = 15.0
ATTEMPTS = 2
SUGGESTION_MARGIN = 0.15  # Claude's suggested box is widened by this much of its size per side

# Magenta does not occur in the scene palette, so the tint cannot be mistaken for the scene.
HIGHLIGHT = (255, 43, 214)
HIGHLIGHT_ALPHA = 0.45

SegmenterName = Literal["gemini", "sam3"]

MaskReading = Literal["xy-image", "yx-image", "xy-box", "yx-box"]
"""How to read one of Gemini's mask points: ``[x, y]`` or ``[y, x]``, 0-1000 across the whole
image or across the entry's own ``box_2d``."""

# Every reading, in the order that wins a tie: the documented [x, y] across the image first, then
# [y, x] across the image (what Gemini sent for five of the first ten real scenes), then the two
# box-relative readings that the documentation's wording also allows.
MASK_READINGS: tuple[MaskReading, ...] = ("xy-image", "yx-image", "xy-box", "yx-box")
READING_NAMES: dict[MaskReading, str] = {
    "xy-image": "[x, y] across the image",
    "yx-image": "[y, x] across the image",
    "xy-box": "[x, y] across the box",
    "yx-box": "[y, x] across the box",
}
# Outlines brush or slightly overshoot their own box, so a point this close to it (as a fraction
# of the image, 10 on Gemini's 0-1000 scale) still counts as inside.
READING_TOLERANCE = 0.01
# Readings that score within this of the best one count as tied, and the tie goes to the one
# earliest in MASK_READINGS.
READING_TIE_MARGIN = 0.02
# When both whole-image orders score at least this and tie, either could be the right one: the
# points fit the box both ways round. That is reported rather than decided silently.
READING_AMBIGUOUS_MIN = 0.5

# The model's field order, used to place hotspot fields that a scene does not have yet.
BRIDGE_FIELD_ORDER = tuple(Bridge.model_fields)


# --------------------------------------------------------------------------------------------
# Candidates and the provider protocol
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadingScore:
    """How well one reading of a Gemini mask fits the box Gemini gave for the same entry.

    ``box_2d`` is reliable even when the mask points are not in the documented order, and an
    outline of the object has to stay inside the object's box and reach across it. So a reading
    is scored by the share of its points that land inside the box and by how much of the box's
    width and height those points span.
    """

    reading: MaskReading
    inside: float
    """Share of the mask points inside ``box_2d`` (within `READING_TOLERANCE`), 0-1."""
    span: tuple[float, float]
    """Width and height of the extent of the points inside the box, as fractions of the box's."""

    @property
    def total(self) -> float:
        """``inside`` times both spans: 1.0 for an outline that stays in its box and reaches
        all four sides, near 0 for the slivers and stray points that a wrong reading gives."""
        return self.inside * self.span[0] * self.span[1]

    @property
    def outside(self) -> float:
        return 1.0 - self.inside

    def describe(self) -> str:
        return (
            f"{READING_NAMES[self.reading]} ({self.reading}): {self.inside:.0%} of its points "
            f"inside its box, spanning {self.span[0]:.0%} x {self.span[1]:.0%} of it"
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "inside": round(self.inside, 4),
            "span": [round(self.span[0], 4), round(self.span[1], 4)],
            "total": round(self.total, 4),
        }


@dataclass(frozen=True)
class Candidate:
    """One instance of the object that a segmenter found."""

    polygon: Polygon
    bbox: BBox
    label: str
    score: float | None = None
    reading: ReadingScore | None = None
    """For a Gemini mask: the reading of its points that was used. None for other sources."""
    readings: tuple[ReadingScore, ...] = ()
    """For a Gemini mask: every reading that was considered, in `MASK_READINGS` order."""

    @property
    def area(self) -> float:
        return geometry.area(self.polygon) if len(self.polygon) >= 3 else 0.0

    @property
    def centroid(self) -> tuple[float, float]:
        return geometry.centroid(self.polygon)


class SegmentationProvider(Protocol):
    """Finds every instance of a named object in an image."""

    source: HotspotSource
    """What ``bridge.source`` records for polygons from this provider."""
    provider: str
    """The company or service called, as the generation log names it."""
    model: str
    call_estimate_usd: float
    last_cost_usd: float | None
    """Cost of the most recent `segment` call, from the reported usage; 0 when the call failed
    before an answer came back, None when an answer came back without usage (the cost is then
    unknown, and the spend guard counts the estimate). Providers may also set
    ``last_request_id``, ``last_response_id`` and ``last_usage`` for the generation log."""

    def segment(
        self, image_path: Path, concept: str, hint_box: BBox | None = None
    ) -> list[Candidate]:
        """Candidates for ``concept``; ``hint_box`` narrows the search when given."""
        ...


# --------------------------------------------------------------------------------------------
# Polygon tools
# --------------------------------------------------------------------------------------------


def _fixed_point(polygon: Sequence[Sequence[float]], width: int, height: int) -> np.ndarray:
    # Normalised 0 is the left edge of the first pixel, which is -0.5 in OpenCV's pixel-centre
    # coordinates.
    scale = 1 << _SHIFT
    return np.array(
        [[round((x * width - 0.5) * scale), round((y * height - 0.5) * scale)] for x, y in polygon],
        dtype=np.int32,
    )


def rasterize(polygon: Sequence[Sequence[float]], width: int, height: int) -> np.ndarray:
    """A ``height x width`` uint8 mask: 255 inside the polygon, 0 outside."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(polygon) >= 3:
        cv2.fillPoly(mask, [_fixed_point(polygon, width, height)], 255, cv2.LINE_8, _SHIFT)
    return mask


def _simplify(contour: np.ndarray, max_vertices: int) -> np.ndarray:
    """Douglas-Peucker with the smallest tolerance that meets the vertex budget."""
    epsilon = 1.5  # pixels; finer keeps the stair-steps of the rasterised edge
    approx = cv2.approxPolyDP(contour, epsilon, True)
    while len(approx) > max_vertices:
        epsilon *= 1.2
        approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 3:
        approx = cv2.boxPoints(cv2.minAreaRect(contour))
    return np.asarray(approx, dtype=np.float64).reshape(-1, 2)


def _normalise(points: np.ndarray, width: int, height: int) -> Polygon:
    polygon: Polygon = []
    for px, py in points.tolist():  # plain floats: YAML cannot represent numpy scalars
        x = round(min(max((px + 0.5) / width, 0.0), 1.0), POLYGON_DECIMALS)
        y = round(min(max((py + 0.5) / height, 0.0), 1.0), POLYGON_DECIMALS)
        if not polygon or polygon[-1] != (x, y):
            polygon.append((x, y))
    while len(polygon) > 1 and polygon[0] == polygon[-1]:
        polygon.pop()
    return polygon


def _as_binary(mask: Any, threshold: int) -> np.ndarray:
    array = np.asarray(mask)
    if array.dtype == bool:
        return array.astype(np.uint8)
    if array.ndim == 3:
        gray = array[..., :3].max(axis=2)
        if array.shape[2] == 4:  # transparent background counts as outside
            gray = np.minimum(gray, array[..., 3])
        array = gray
    if array.ndim != 2:
        raise ValueError(f"a mask must be a 2-D array, got shape {array.shape}")
    return (array > threshold).astype(np.uint8)


def mask_to_polygon(
    mask: Any,
    *,
    max_vertices: int = MAX_VERTICES,
    dilate_fraction: float = DILATE_FRACTION,
    threshold: int = 127,
) -> Polygon:
    """Outline the largest connected region of a raster mask as a normalised polygon.

    The region is grown by ``dilate_fraction`` of the mask width before it is outlined, so the
    simplified outline still covers the object, then simplified to at most ``max_vertices``.
    Returns an empty list when the mask is empty. The result is always a simple polygon: if the
    simplified outline would cross itself, its convex hull is used instead.
    """
    binary = _as_binary(mask, threshold)
    height, width = binary.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return []
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = np.where(labels == largest, 255, 0).astype(np.uint8)
    radius = round(dilate_fraction * width)
    if radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        component = cv2.dilate(component, kernel)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    polygon = _normalise(_simplify(contour, max_vertices), width, height)
    if len(polygon) < 3 or geometry.simplicity_problem(polygon) is not None:
        hull = cv2.convexHull(contour)
        polygon = _normalise(_simplify(hull, max_vertices), width, height)
    return polygon if len(polygon) >= 3 else []


def clean_polygon(
    polygon: Sequence[Sequence[float]],
    image_size: tuple[int, int] = (MASTER_WIDTH, MASTER_HEIGHT),
    *,
    max_vertices: int = MAX_VERTICES,
    dilate_fraction: float = DILATE_FRACTION,
) -> Polygon:
    """Rasterise at the image's resolution, keep the largest piece, grow, simplify, normalise.

    Segmenter polygons can cross themselves, come in several pieces or carry hundreds of
    vertices; the result of this is a simple polygon with at most ``max_vertices`` vertices.
    """
    width, height = image_size
    mask = rasterize(polygon, width, height)
    return mask_to_polygon(mask, max_vertices=max_vertices, dilate_fraction=dilate_fraction)


def in_safe_area(point: Sequence[float]) -> bool:
    """The rule `dsp validate` applies to a hotspot centroid."""
    x, y = point
    return SAFE_X[0] <= x <= SAFE_X[1] and SAFE_Y[0] <= y <= SAFE_Y[1]


def choose_candidate(candidates: Sequence[Candidate]) -> Candidate | None:
    """The largest candidate whose centroid is in the safe area, else the largest overall."""
    usable = [c for c in candidates if len(c.polygon) >= 3 and c.area > 0]
    if not usable:
        return None
    inside = [c for c in usable if in_safe_area(c.centroid)]
    return max(inside or usable, key=lambda c: c.area)


def expand_box(box: BBox, margin: float) -> BBox:
    """Grow a box by ``margin`` of its width and height on every side, kept inside the image."""
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * margin, (y1 - y0) * margin
    return (max(0.0, x0 - dx), max(0.0, y0 - dy), min(1.0, x1 + dx), min(1.0, y1 + dy))


def check_box(box: Sequence[float]) -> BBox:
    """Validate a normalised ``(x0, y0, x1, y1)`` box given on the command line."""
    if len(box) != 4:
        raise ValueError("a box needs exactly four numbers: x0 y0 x1 y1")
    x0, y0, x1, y1 = (float(v) for v in box)
    if not all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1)):
        raise ValueError("box coordinates are normalised and must lie between 0 and 1")
    if not (x0 < x1 and y0 < y1):
        raise ValueError("a box needs x0 < x1 and y0 < y1")
    return (x0, y0, x1, y1)


# --------------------------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------------------------

GEMINI_PROMPT = """Give the segmentation masks for the {concept}.
Output a JSON list of segmentation masks where each entry contains the 2D
bounding box in the key "box_2d", the segmentation mask in key "mask", and
the text label in the key "label". Use descriptive labels."""

GEMINI_HINT = (
    "The {concept} is inside the bounding box {box}, given as [ymin, xmin, ymax, xmax] "
    "normalized to 0-1000. Only segment the {concept} inside that box."
)


class GeminiMask(BaseModel):
    """The response entry from the Gemini segmentation example, field for field."""

    box_2d: list[int] = Field(
        description="The 2D bounding box of the item as [ymin, xmin, ymax, xmax] normalized "
        "to 0-1000."
    )
    mask: list[list[int]] = Field(
        description="The segmentation mask of the item as a polygon of [x,y] coordinates, "
        "normalized to 0-1000."
    )
    label: str = Field(description="A descriptive label for the item.")


class GeminiMasks(BaseModel):
    boxes: list[GeminiMask]


class _LenientMask(BaseModel):
    """What is actually accepted back: numbers may be floats and the label may be missing."""

    box_2d: tuple[float, float, float, float]
    mask: list[tuple[float, float]] = Field(default_factory=list)
    label: str = ""


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def parse_gemini_masks(text: str) -> list[_LenientMask]:
    """Entries of a segmentation response; malformed entries are skipped."""
    data = json.loads(_FENCE.sub("", text.strip()))
    if isinstance(data, dict):
        data = data.get("boxes", data.get("masks", []))
    if not isinstance(data, list):
        raise ValueError("the segmentation response is not a list of masks")
    entries = []
    for item in data:
        try:
            entries.append(_LenientMask.model_validate(item))
        except ValidationError:
            continue
    return entries


def _box_from_1000(box_2d: Sequence[float]) -> BBox:
    ymin, xmin, ymax, xmax = (min(max(float(v), 0.0), 1000.0) / 1000.0 for v in box_2d)
    return (min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax))


def box_to_1000(box: BBox) -> list[int]:
    """A normalised ``(x0, y0, x1, y1)`` box in Gemini's ``[ymin, xmin, ymax, xmax]`` 0-1000."""
    x0, y0, x1, y1 = box
    return [round(y0 * 1000), round(x0 * 1000), round(y1 * 1000), round(x1 * 1000)]


def read_mask_points(
    box_2d: Sequence[float],
    mask: Sequence[Sequence[float]],
    reading: MaskReading,
    *,
    clamp: bool = True,
) -> Polygon:
    """Gemini mask points (0-1000) in normalised image coordinates, read as ``reading`` says.

    With ``clamp`` (the default) every point is kept inside the image, or inside the box for a
    box-relative reading; scoring a reading turns it off so that stray points show up as such.
    """
    order, frame = reading.split("-")
    x0, y0, x1, y1 = _box_from_1000(box_2d)
    polygon: Polygon = []
    for first, second in mask:
        px, py = (first, second) if order == "xy" else (second, first)
        u, v = px / 1000.0, py / 1000.0
        if clamp:
            u, v = min(max(u, 0.0), 1.0), min(max(v, 0.0), 1.0)
        if frame == "box":
            polygon.append((x0 + u * (x1 - x0), y0 + v * (y1 - y0)))
        else:
            polygon.append((u, v))
    return polygon


def _extent(values: Sequence[float], low: float, high: float) -> float:
    if not values or high <= low:
        return 0.0
    return min((max(values) - min(values)) / (high - low), 1.0)


def score_mask_reading(
    box_2d: Sequence[float], mask: Sequence[Sequence[float]], reading: MaskReading
) -> ReadingScore:
    """How well ``reading`` of the mask points fits the entry's own box (see `ReadingScore`)."""
    x0, y0, x1, y1 = _box_from_1000(box_2d)
    tol = READING_TOLERANCE
    points = read_mask_points(box_2d, mask, reading, clamp=False)
    inside = [
        (min(max(x, x0), x1), min(max(y, y0), y1))
        for x, y in points
        if x0 - tol <= x <= x1 + tol and y0 - tol <= y <= y1 + tol
    ]
    share = len(inside) / len(points) if points else 0.0
    span = (_extent([x for x, _ in inside], x0, x1), _extent([y for _, y in inside], y0, y1))
    return ReadingScore(reading=reading, inside=share, span=span)


def score_mask_readings(
    box_2d: Sequence[float], mask: Sequence[Sequence[float]]
) -> tuple[ReadingScore, ...]:
    """Every reading of the mask, scored, in `MASK_READINGS` order."""
    return tuple(score_mask_reading(box_2d, mask, reading) for reading in MASK_READINGS)


def choose_mask_reading(scores: Sequence[ReadingScore]) -> ReadingScore:
    """The best-scoring reading; near-ties go to the earliest in `MASK_READINGS`.

    Gemini's documentation says ``[x, y]`` and the smoke test saw whole-image points, but on
    five of the first ten real scenes every mask came back as ``[y, x]`` while ``box_2d`` stayed
    ``[ymin, xmin, ymax, xmax]``. Read the documented way, those outlines collapsed into slivers
    that Claude rightly rejected. Only one reading puts an outline inside its own box and across
    it, so on the saved responses the right reading wins by a wide margin (see the tests).

    Two limits, both because only the box is used to judge, never the image:

    * The two box-relative readings always score the same, because a box is square in its own
      0-1000 frame, so an ``[y, x]`` box-relative mask would be read as ``[x, y]``. Gemini has
      not been seen to send either.
    * When a near-square box straddles the diagonal x = y (for example ``[400, 400, 600, 600]``),
      an outline and its mirror image across that diagonal both fit it, so ``[x, y]`` and
      ``[y, x]`` across the image can tie and ``[x, y]`` is used. `order_is_ambiguous` spots
      this, and `reading_report` prints a warning so the overlay gets a look.
    """
    ordered = sorted(scores, key=lambda s: MASK_READINGS.index(s.reading))
    best = max(s.total for s in ordered)
    return next(s for s in ordered if s.total >= best - READING_TIE_MARGIN)


def order_is_ambiguous(scores: Sequence[ReadingScore]) -> bool:
    """Whether ``[x, y]`` and ``[y, x]`` across the image both fit the box well and tie."""
    totals = {s.reading: s.total for s in scores}
    xy, yx = totals.get("xy-image", 0.0), totals.get("yx-image", 0.0)
    return min(xy, yx) >= READING_AMBIGUOUS_MIN and abs(xy - yx) <= READING_TIE_MARGIN


def gemini_mask_to_polygon(
    box_2d: Sequence[float],
    mask: Sequence[Sequence[float]],
    reading: MaskReading | Literal["auto"] = "auto",
) -> Polygon:
    """Convert a Gemini mask polygon (0-1000) to normalised image coordinates.

    ``reading`` fixes the point order and frame (see `MaskReading`); ``"auto"`` picks the one
    that fits the entry's box best with `choose_mask_reading`.
    """
    if reading == "auto":
        reading = choose_mask_reading(score_mask_readings(box_2d, mask)).reading
    return read_mask_points(box_2d, mask, reading)


def gemini_candidate(
    entry: _LenientMask, reading: MaskReading | Literal["auto"] = "auto"
) -> Candidate | None:
    """A candidate from one response entry; a missing mask falls back to the box itself.

    The candidate records which reading of the mask points was used and how every reading
    scored, so the review files can show it.
    """
    bbox = _box_from_1000(entry.box_2d)
    chosen: ReadingScore | None = None
    scores: tuple[ReadingScore, ...] = ()
    if len(entry.mask) >= 3:
        scores = score_mask_readings(entry.box_2d, entry.mask)
        if reading == "auto":
            chosen = choose_mask_reading(scores)
        else:
            chosen = next(s for s in scores if s.reading == reading)
        polygon = dedupe_vertices(read_mask_points(entry.box_2d, entry.mask, chosen.reading))
        if geometry.simplicity_problem(polygon) is not None:
            # A crossing outline has its vertices in the wrong order, and filling it leaves
            # holes inside the object (seen in the smoke test: a C shape that missed the body
            # of the telephone box). The points still mark the object's extent, so use their
            # convex hull, which is always simple.
            polygon = convex_hull(polygon)
    else:
        x0, y0, x1, y1 = bbox
        polygon = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if len(polygon) < 3 or geometry.area(polygon) <= 0:
        return None
    return Candidate(polygon=polygon, bbox=bbox, label=entry.label, reading=chosen, readings=scores)


def dedupe_vertices(polygon: Sequence[Sequence[float]]) -> Polygon:
    """Drop consecutive repeats and a closing vertex equal to the first.

    Outlines often repeat a point or close themselves explicitly. Neither makes the shape cross
    itself, but both fail the simplicity check, which would otherwise trigger the convex-hull
    fallback and fill in concave parts of the object.
    """
    out: Polygon = []
    for x, y in polygon:
        point = (float(x), float(y))
        if not out or out[-1] != point:
            out.append(point)
    while len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def convex_hull(polygon: Sequence[Sequence[float]]) -> Polygon:
    """Convex hull of the points (Andrew's monotone chain), in normalised coordinates."""
    points = sorted({(float(x), float(y)) for x, y in polygon})
    if len(points) < 3:
        return list(points)

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: Polygon = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: Polygon = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _concept(bridge_object: str) -> str:
    return re.sub(r"^(the|a|an)\s+", "", bridge_object.strip(), flags=re.IGNORECASE)


def _jpeg_base64(image: Image.Image, quality: int = 92) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return base64.standard_b64encode(buffer.getvalue()).decode("ascii")


def _elided(data: str) -> str:
    return f"<{len(data) * 3 // 4} bytes of base64 JPEG>"


class GeminiSegmenter:
    """Gemini segmentation through the Interactions API, as the image-understanding page shows."""

    source: HotspotSource = "gemini"
    provider = "gemini"
    call_estimate_usd = GEMINI_CALL_ESTIMATE_USD

    def __init__(
        self,
        client: Any | None = None,
        model: str = GEMINI_MODEL,
        reading: MaskReading | Literal["auto"] = "auto",
    ):
        self._client = client
        self.model = model
        self.reading = reading
        self.thinking_level = GEMINI_THINKING_LEVEL
        self.last_cost_usd = 0.0
        self.last_raw: str | None = None
        self.last_response_id: str | None = None
        self.last_usage: dict[str, Any] | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from google import genai
            from google.genai import types

            # Reads GEMINI_API_KEY; retries 429 and 5xx with exponential backoff.
            self._client = genai.Client(
                http_options=types.HttpOptions(
                    retry_options=types.HttpRetryOptions(attempts=GEMINI_RETRY_ATTEMPTS)
                )
            )
        return self._client

    def prompt(self, concept: str, hint_box: BBox | None = None) -> str:
        concept = _concept(concept)
        text = GEMINI_PROMPT.format(concept=concept)
        if hint_box is not None:
            text += "\n" + GEMINI_HINT.format(concept=concept, box=box_to_1000(hint_box))
        return text

    def request(
        self, image_path: Path, concept: str, hint_box: BBox | None = None, *, elide: bool = False
    ) -> dict[str, Any]:
        """Keyword arguments for ``client.interactions.create``."""
        with Image.open(image_path) as image:
            data = _jpeg_base64(image)
        return {
            "model": self.model,
            "input": [
                {"type": "text", "text": self.prompt(concept, hint_box)},
                {
                    "type": "image",
                    "data": _elided(data) if elide else data,
                    "mime_type": "image/jpeg",
                },
            ],
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": GeminiMasks.model_json_schema(),
            },
            "generation_config": {"thinking_level": self.thinking_level},
            # A one-off request; nothing needs to be kept server-side for later turns.
            "store": False,
        }

    def segment(
        self, image_path: Path, concept: str, hint_box: BBox | None = None
    ) -> list[Candidate]:
        # Nothing from an earlier call may be mistaken for this one's if it fails.
        self.last_cost_usd, self.last_raw = 0.0, None
        self.last_response_id, self.last_usage = None, None
        while True:
            try:
                interaction = self.client.interactions.create(
                    **self.request(image_path, concept, hint_box)
                )
                break
            except Exception as error:
                stronger = _stronger_thinking_level(self.thinking_level)
                if not _rejects_thinking_level(error) or stronger is None:
                    raise
                self.thinking_level = stronger  # a refused request is not billed
        usage = getattr(interaction, "usage", None)
        self.last_raw = interaction.output_text or ""
        self.last_cost_usd = gemini_cost(usage)
        self.last_response_id = getattr(interaction, "id", None) or None
        self.last_usage = _usage_record(
            usage, ("total_input_tokens", "total_output_tokens", "total_thought_tokens")
        )
        candidates = (gemini_candidate(e, self.reading) for e in parse_gemini_masks(self.last_raw))
        return [c for c in candidates if c is not None]


def _usage_record(usage: Any, names: Sequence[str]) -> dict[str, Any] | None:
    """The token counts a provider reported, as plain data for the generation log."""
    if usage is None:
        return None
    record = {name: getattr(usage, name, None) for name in names}
    record = {name: value for name, value in record.items() if isinstance(value, int)}
    return record or None


def _rejects_thinking_level(error: Exception) -> bool:
    return getattr(error, "status_code", None) == 400 and "thinking level" in str(error).lower()


def _stronger_thinking_level(level: str) -> str | None:
    if level not in GEMINI_THINKING_LEVELS:
        return None
    index = GEMINI_THINKING_LEVELS.index(level) + 1
    return GEMINI_THINKING_LEVELS[index] if index < len(GEMINI_THINKING_LEVELS) else None


def gemini_cost(usage: Any) -> float | None:
    """Dollar cost from the reported token usage; None when no usage was reported."""
    if usage is None:
        return None
    tokens_in = getattr(usage, "total_input_tokens", None) or 0
    tokens_out = (getattr(usage, "total_output_tokens", None) or 0) + (
        getattr(usage, "total_thought_tokens", None) or 0
    )
    return (
        tokens_in * GEMINI_INPUT_USD_PER_MTOK + tokens_out * GEMINI_OUTPUT_USD_PER_MTOK
    ) / 1_000_000


# --------------------------------------------------------------------------------------------
# SAM 3 on fal (optional)
# --------------------------------------------------------------------------------------------


def _decode_image(url: str, fetch: Callable[[str], bytes] | None = None) -> Image.Image:
    if url.startswith("data:"):
        try:
            payload = base64.b64decode(url.split(",", 1)[1])
        except (IndexError, binascii.Error) as error:
            raise ValueError("malformed data URL in the SAM 3 response") from error
    else:
        if fetch is None:
            import httpx

            response = httpx.get(url, timeout=60.0, follow_redirects=True)
            response.raise_for_status()
            payload = response.content
        else:
            payload = fetch(url)
    image = Image.open(io.BytesIO(payload))
    image.load()
    return image


class Sam3Segmenter:
    """SAM 3 on fal: text-prompted raster masks, outlined with `mask_to_polygon`."""

    source: HotspotSource = "sam3"
    provider = "fal"
    model = SAM3_APP
    call_estimate_usd = SAM3_CALL_ESTIMATE_USD

    def __init__(self, client: Any | None = None, fetch: Callable[[str], bytes] | None = None):
        self._client = client
        self._fetch = fetch
        self.last_cost_usd = 0.0
        self.last_raw: str | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import fal_client
            except ImportError as error:  # an optional dependency
                raise RuntimeError(
                    "the SAM 3 provider needs fal-client: run `uv sync --extra sam3` and set "
                    "FAL_KEY"
                ) from error
            self._client = fal_client.SyncClient()
        return self._client

    def request(
        self, image_path: Path, concept: str, hint_box: BBox | None = None, *, elide: bool = False
    ) -> dict[str, Any]:
        """Arguments for ``fal_client.subscribe("fal-ai/sam-3/image", arguments=...)``."""
        with Image.open(image_path) as image:
            width, height = image.size
            data = _jpeg_base64(image)
        arguments: dict[str, Any] = {
            "image_url": _elided(data) if elide else f"data:image/jpeg;base64,{data}",
            "prompt": _concept(concept),
            "apply_mask": False,
            "sync_mode": True,
            "output_format": "png",
            "return_multiple_masks": True,
            "max_masks": 3,
            "include_scores": True,
            "include_boxes": True,
        }
        if hint_box is not None:
            x0, y0, x1, y1 = hint_box
            arguments["box_prompts"] = [
                {
                    "x_min": round(x0 * width),
                    "y_min": round(y0 * height),
                    "x_max": round(x1 * width),
                    "y_max": round(y1 * height),
                }
            ]
        return arguments

    def segment(
        self, image_path: Path, concept: str, hint_box: BBox | None = None
    ) -> list[Candidate]:
        self.last_cost_usd, self.last_raw = 0.0, None
        result = self.client.subscribe(
            SAM3_APP, arguments=self.request(image_path, concept, hint_box)
        )
        # fal reports no usage or price for this app, so the cost of an answered call is
        # unknown; the spend guard counts the estimate for it.
        self.last_cost_usd = None
        self.last_raw = json.dumps(
            {k: v for k, v in result.items() if k not in ("masks", "image")}, default=str
        )
        metadata = result.get("metadata") or []
        scores = result.get("scores") or [m.get("score") for m in metadata]
        boxes = result.get("boxes") or [m.get("box") for m in metadata]
        candidates = []
        for i, image in enumerate(result.get("masks") or []):
            mask = _decode_image(image["url"], self._fetch)
            polygon = mask_to_polygon(
                np.asarray(mask), max_vertices=RAW_VERTICES, dilate_fraction=0
            )
            if len(polygon) < 3:
                continue
            bbox = geometry.bbox(polygon)
            if i < len(boxes) and boxes[i] and len(boxes[i]) == 4:
                cx, cy, w, h = (float(v) for v in boxes[i])
                bbox = (
                    max(0.0, cx - w / 2),
                    max(0.0, cy - h / 2),
                    min(1.0, cx + w / 2),
                    min(1.0, cy + h / 2),
                )
            score = scores[i] if i < len(scores) and scores[i] is not None else None
            candidates.append(
                Candidate(
                    polygon=polygon,
                    bbox=bbox,
                    label=_concept(concept),
                    score=float(score) if score is not None else None,
                )
            )
        return candidates


def make_segmenter(name: str) -> GeminiSegmenter | Sam3Segmenter:
    if name == "gemini":
        return GeminiSegmenter()
    if name == "sam3":
        return Sam3Segmenter()
    raise ValueError(f"unknown segmentation provider '{name}'; use gemini or sam3")


def segmenter_name_from_env() -> SegmenterName:
    name = (os.environ.get("DSP_SEGMENTATION_PROVIDER") or "gemini").strip().lower()
    if name not in ("gemini", "sam3"):
        raise ValueError(f"DSP_SEGMENTATION_PROVIDER must be gemini or sam3, not '{name}'")
    return name  # type: ignore[return-value]


# --------------------------------------------------------------------------------------------
# Fakes (tests and dry runs)
# --------------------------------------------------------------------------------------------


class FakeSegmenter:
    """Returns prepared candidate lists in turn (the last one repeats) and records each call."""

    provider = "fake"
    model = "fake segmenter"
    call_estimate_usd = 0.0

    def __init__(
        self,
        responses: Sequence[Sequence[Candidate]],
        source: HotspotSource = "gemini",
        cost_usd: float = 0.0,
    ):
        if not responses:
            raise ValueError("FakeSegmenter needs at least one response")
        self.responses = [list(r) for r in responses]
        self.source = source
        self.cost_usd = cost_usd
        self.calls: list[tuple[Path, str, BBox | None]] = []
        self.last_cost_usd = 0.0

    def segment(
        self, image_path: Path, concept: str, hint_box: BBox | None = None
    ) -> list[Candidate]:
        self.calls.append((Path(image_path), concept, hint_box))
        self.last_cost_usd = self.cost_usd
        return list(self.responses[min(len(self.calls), len(self.responses)) - 1])


class MaskFileSegmenter:
    """Dry-run stand-in: outlines the scene's committed mask instead of calling a model.

    Placeholder scenes ship a mask of their glowing shape, which is exactly what a segmenter
    should find. Without a mask file it returns the polygon already in the manifest.
    """

    provider = "none"
    model = "the scene's mask file (dry run)"
    call_estimate_usd = 0.0

    def __init__(self, mask_path: Path, fallback: Polygon, source: HotspotSource):
        self.mask_path = mask_path
        self.fallback = list(fallback)
        self.source = source
        self.last_cost_usd = 0.0

    def segment(
        self, image_path: Path, concept: str, hint_box: BBox | None = None
    ) -> list[Candidate]:
        if self.mask_path.exists():
            with Image.open(self.mask_path) as mask:
                polygon = mask_to_polygon(
                    np.asarray(mask.convert("L")), max_vertices=RAW_VERTICES, dilate_fraction=0
                )
        else:
            polygon = self.fallback
        if len(polygon) < 3:
            return []
        return [Candidate(polygon=polygon, bbox=geometry.bbox(polygon), label=_concept(concept))]


# --------------------------------------------------------------------------------------------
# Overlays
# --------------------------------------------------------------------------------------------


def _outline_width(width: int) -> int:
    return max(2, round(width / 640))


def render_overlay(image: Image.Image, polygon: Sequence[Sequence[float]]) -> Image.Image:
    """The image with the polygon tinted magenta and outlined, at the image's own size."""
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    height, width = rgb.shape[:2]
    inside = rasterize(polygon, width, height) > 0
    color = np.array(HIGHLIGHT, dtype=np.float32)
    rgb[inside] = rgb[inside] * (1 - HIGHLIGHT_ALPHA) + color * HIGHLIGHT_ALPHA
    out = np.ascontiguousarray(np.clip(np.rint(rgb), 0, 255).astype(np.uint8))
    points = _fixed_point(polygon, width, height)
    cv2.polylines(out, [points], True, HIGHLIGHT, _outline_width(width), cv2.LINE_AA, _SHIFT)
    return Image.fromarray(out)


def render_closeup(image: Image.Image, polygon: Sequence[Sequence[float]]) -> Image.Image:
    """A crop around the polygon with only its outline drawn, so the object stays visible."""
    rgb = np.array(image.convert("RGB"))  # a writable copy to draw on
    height, width = rgb.shape[:2]
    points = _fixed_point(polygon, width, height)
    cv2.polylines(rgb, [points], True, HIGHLIGHT, _outline_width(width), cv2.LINE_AA, _SHIFT)
    x0, y0, x1, y1 = geometry.bbox(polygon)
    box_w, box_h = (x1 - x0) * width, (y1 - y0) * height
    margin = max(0.6 * max(box_w, box_h), 0.05 * width)
    left = max(0, int(x0 * width - margin))
    top = max(0, int(y0 * height - margin))
    right = min(width, int(x1 * width + margin) + 1)
    bottom = min(height, int(y1 * height + margin) + 1)
    crop = Image.fromarray(rgb[top:bottom, left:right])
    return _fit(crop, CLOSEUP_LONG_EDGE, upscale=True)


def _fit(image: Image.Image, long_edge: int, upscale: bool = False) -> Image.Image:
    scale = long_edge / max(image.size)
    if scale >= 1 and not upscale:
        return image
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


# --------------------------------------------------------------------------------------------
# Verification with Claude
# --------------------------------------------------------------------------------------------


class Verdict(BaseModel):
    """Claude's judgement of one hotspot outline."""

    matches: bool = Field(
        description="True only when the highlighted region is exactly the named object."
    )
    reason: str = Field(description="One or two sentences naming what the region covers.")
    suggested_box: BBox | None = Field(
        default=None,
        description="Normalised (x0, y0, x1, y1) box around the object, used for a retry.",
    )

    @field_validator("suggested_box", mode="before")
    @classmethod
    def _usable_box(cls, value: Any) -> Any:
        """Keep a suggestion only when it is a real box; a bad one just means no hint."""
        if value is None:
            return None
        if isinstance(value, dict):
            value = [value.get(k) for k in ("x0", "y0", "x1", "y1")]
        try:
            x0, y0, x1, y1 = (min(max(float(v), 0.0), 1.0) for v in value)
        except (TypeError, ValueError):
            return None
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        if x1 - x0 < 1e-3 or y1 - y0 < 1e-3:
            return None
        return (x0, y0, x1, y1)


_COORD = {"type": "number"}
VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "matches": {
            "type": "boolean",
            "description": "True only when the highlighted region is exactly the named object "
            "and nothing else.",
        },
        "reason": {
            "type": "string",
            "description": "One or two sentences naming what the highlighted region covers.",
        },
        "suggested_box": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"x0": _COORD, "y0": _COORD, "x1": _COORD, "y1": _COORD},
                    "required": ["x0", "y0", "x1", "y1"],
                    "additionalProperties": False,
                },
                {"type": "null"},
            ],
            "description": "When matches is false and the object is visible: a tight box "
            "around it as fractions of the full image (0-1, origin top-left). Otherwise null.",
        },
    },
    "required": ["matches", "reason", "suggested_box"],
    "additionalProperties": False,
}

VERIFY_SYSTEM = (
    "You check hotspot outlines for a picture-exploration game. Each scene has one secretly "
    "clickable object, and an automatic segmenter has outlined it. You decide whether the "
    "outline is right, looking carefully at the image rather than assuming."
)

VERIFY_QUESTION = """Is the highlighted region exactly the {concept} and nothing else?

Answer matches = true only when both hold:
- The region covers the {concept}: the whole object, not just a part of it, and not a different \
object that merely looks similar. If several are visible, any one complete instance is fine.
- The region does not include other objects or a large area of background. The outline is \
deliberately grown a few pixels past the object's edges so it is easier to click; that margin \
is expected and is not a reason to say no.
{context}
In reason, say in one or two sentences what the region actually covers. When matches is false \
and you can see the {concept} somewhere in the first image, set suggested_box to a tight box \
around it, as fractions of the first image's width and height (x0, y0 at the top-left corner, \
x1, y1 at the bottom-right, each between 0 and 1). Otherwise set suggested_box to null."""


class VerificationError(RuntimeError):
    """Claude's answer could not be read as a verdict."""


class Verifier(Protocol):
    provider: str
    """The company or service called, as the generation log names it."""
    model: str
    call_estimate_usd: float
    last_cost_usd: float | None
    """Cost of the most recent `verify` call; 0 when it failed before an answer came back, None
    when an answer came back without usage. Verifiers may also set ``last_request_id``,
    ``last_response_id``, ``last_usage`` and ``last_model`` (the model that answered) for the
    generation log."""

    def verify(
        self,
        image_path: Path,
        polygon: Sequence[Sequence[float]],
        concept: str,
        context: str | None = None,
    ) -> Verdict: ...


def _image_block(image: Image.Image, elide: bool) -> dict[str, Any]:
    data = _jpeg_base64(image, quality=90)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": _elided(data) if elide else data,
        },
    }


def claude_cost(usage: Any) -> float | None:
    """Dollar cost from the reported token usage; None when no usage was reported."""
    if usage is None:
        return None
    tokens_in = sum(
        getattr(usage, name, None) or 0
        for name in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )
    tokens_out = getattr(usage, "output_tokens", None) or 0
    return (
        tokens_in * CLAUDE_INPUT_USD_PER_MTOK + tokens_out * CLAUDE_OUTPUT_USD_PER_MTOK
    ) / 1_000_000


def parse_verdict(response: Any) -> Verdict:
    """Read a Messages API response; a refusal becomes a "no" with the reason recorded."""
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) or "unspecified"
        return Verdict(
            matches=False,
            reason=f"Claude declined to judge this region (refusal category: {category}).",
        )
    text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
    if response.stop_reason == "max_tokens":
        raise VerificationError("Claude ran out of output tokens before answering")
    try:
        return Verdict.model_validate_json(text)
    except ValidationError as error:
        raise VerificationError(f"Claude's answer is not a valid verdict: {error}") from error


class ClaudeVerifier:
    """Asks Claude whether a tinted region is exactly the named object."""

    provider = "anthropic"
    model = CLAUDE_MODEL
    call_estimate_usd = CLAUDE_CALL_ESTIMATE_USD

    def __init__(self, client: Any | None = None, effort: str = CLAUDE_EFFORT):
        self._client = client
        self.effort = effort
        self.last_cost_usd = 0.0
        self.last_model: str | None = None
        self.last_request_id: str | None = None
        self.last_response_id: str | None = None
        self.last_usage: dict[str, Any] | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        return self._client

    def request(
        self,
        image_path: Path,
        polygon: Sequence[Sequence[float]],
        concept: str,
        context: str | None = None,
        *,
        elide: bool = False,
    ) -> dict[str, Any]:
        """Keyword arguments for ``client.beta.messages.create``."""
        concept = _concept(concept)
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        overlay = _fit(render_overlay(image, polygon), CLAUDE_IMAGE_LONG_EDGE)
        closeup = render_closeup(image, polygon)
        hint = (
            f"\nFor orientation only (it may be imprecise), the scene was written with the "
            f"{concept} placed: {context}.\n"
            if context
            else ""
        )
        content = [
            {
                "type": "text",
                "text": f"Image 1: the whole scene. The region proposed as the clickable "
                f"{concept} is tinted magenta and outlined.",
            },
            _image_block(overlay, elide),
            {
                "type": "text",
                "text": "Image 2: a close-up of the same area with only the magenta outline "
                "drawn, so the object underneath is visible.",
            },
            _image_block(closeup, elide),
            {"type": "text", "text": VERIFY_QUESTION.format(concept=concept, context=hint)},
        ]
        return {
            "model": self.model,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "betas": CLAUDE_BETAS,
            "fallbacks": "default",
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": VERDICT_SCHEMA},
            },
            "system": VERIFY_SYSTEM,
            "messages": [{"role": "user", "content": content}],
        }

    def verify(
        self,
        image_path: Path,
        polygon: Sequence[Sequence[float]],
        concept: str,
        context: str | None = None,
    ) -> Verdict:
        # Nothing from an earlier call may be mistaken for this one's if it fails.
        self.last_cost_usd, self.last_model, self.last_request_id = 0.0, None, None
        self.last_response_id, self.last_usage = None, None
        response = self.client.beta.messages.create(
            **self.request(image_path, polygon, concept, context)
        )
        usage = getattr(response, "usage", None)
        self.last_cost_usd = claude_cost(usage)
        self.last_model = getattr(response, "model", None)
        self.last_request_id = getattr(response, "_request_id", None)
        self.last_response_id = getattr(response, "id", None)
        self.last_usage = _usage_record(
            usage,
            (
                "input_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "output_tokens",
            ),
        )
        return parse_verdict(response)


def verify_with_claude(
    image_path: Path,
    polygon: Sequence[Sequence[float]],
    concept: str,
    *,
    context: str | None = None,
    client: Any | None = None,
) -> Verdict:
    """Render the polygon over the image and ask Claude whether it is exactly ``concept``."""
    return ClaudeVerifier(client).verify(image_path, polygon, concept, context)


class FakeVerifier:
    """Returns prepared verdicts in turn (the last one repeats) and records each call."""

    provider = "fake"
    model = "fake verifier"
    call_estimate_usd = 0.0

    def __init__(self, verdicts: Sequence[Verdict], cost_usd: float = 0.0):
        if not verdicts:
            raise ValueError("FakeVerifier needs at least one verdict")
        self.verdicts = list(verdicts)
        self.cost_usd = cost_usd
        self.calls: list[tuple[Path, Polygon, str, str | None]] = []
        self.last_cost_usd = 0.0

    def verify(self, image_path, polygon, concept, context=None) -> Verdict:
        self.calls.append((Path(image_path), [tuple(p) for p in polygon], concept, context))
        self.last_cost_usd = self.cost_usd
        return self.verdicts[min(len(self.calls), len(self.verdicts)) - 1]


class DryRunVerifier:
    """Prints the request Claude would get and assumes "yes" so the rest of the run shows."""

    provider = "none"
    model = "nobody (dry run)"
    call_estimate_usd = 0.0

    def __init__(self, echo: Echo):
        self.echo = echo
        self.last_cost_usd = 0.0

    def verify(self, image_path, polygon, concept, context=None) -> Verdict:
        request = ClaudeVerifier().request(image_path, polygon, concept, context, elide=True)
        self.echo("  would ask Claude to verify it with client.beta.messages.create:")
        self.echo(_indent(json.dumps(request, indent=2), 4))
        return Verdict(
            matches=True, reason="Dry run: Claude was not asked, so this outline is unverified."
        )


# --------------------------------------------------------------------------------------------
# Spend guard
# --------------------------------------------------------------------------------------------


class SpendLimitExceeded(RuntimeError):
    pass


@dataclass
class Spend:
    """Refuses a paid call that could take the run past its cap."""

    cap_usd: float
    spent_usd: float = 0.0

    def reserve(self, estimate_usd: float, what: str) -> None:
        if self.spent_usd + estimate_usd > self.cap_usd + 1e-12:
            raise SpendLimitExceeded(
                f"{what} could cost up to ${estimate_usd:.2f}, which would take this run past "
                f"its ${self.cap_usd:.2f} cap (${self.spent_usd:.4f} spent so far); raise "
                "DSP_MAX_SPEND_USD to continue"
            )

    def add(self, usd: float) -> None:
        self.spent_usd += usd


def max_spend_from_env() -> float:
    raw = (os.environ.get("DSP_MAX_SPEND_USD") or "").strip()
    if not raw:
        return DEFAULT_MAX_SPEND_USD
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"DSP_MAX_SPEND_USD must be a number of dollars, not '{raw}'") from error
    if value < 0:
        raise ValueError("DSP_MAX_SPEND_USD cannot be negative")
    return value


# --------------------------------------------------------------------------------------------
# Extraction with one retry
# --------------------------------------------------------------------------------------------


@dataclass
class CallLog:
    """One row in pipeline/logs/generations.jsonl per segmentation or verification call.

    Rows share the generation log's common fields (``id``, ``timestamp``, ``kind``,
    ``purpose``, ``provider``, ``model``, ``scene``, ``status``, ``error``, ``duration_s``,
    ``cost_estimate_usd``, ``cost_usd``) so `genlog.total_spend` counts hotspot calls with the
    image and scoring calls. ``kind`` is ``segment`` or ``verify`` and ``purpose`` is
    ``hotspot``. Both kinds add ``attempt``, ``concept``, ``image`` (path and SHA-256 of the
    picture sent), ``request_id``, ``response_id`` and ``usage``; segment rows add
    ``hint_box``, verify rows ``polygon_vertices``, ``served_model`` and ``matches``.
    """

    path: Path
    scene: str
    image: dict[str, str] | None = None

    def start(self, kind: str, provider: Any, attempt: int, **fields: Any) -> dict[str, Any]:
        return {
            "id": genlog.new_row_id(),
            "timestamp": genlog.utc_now(),
            "kind": kind,
            "purpose": "hotspot",
            "provider": getattr(provider, "provider", None),
            "model": getattr(provider, "model", None),
            "scene": self.scene,
            "attempt": attempt,
            "image": self.image,
            **fields,
            "cost_estimate_usd": getattr(provider, "call_estimate_usd", None),
        }

    def finish(
        self,
        row: dict[str, Any],
        provider: Any,
        started: float,
        error: BaseException | None = None,
        **fields: Any,
    ) -> str:
        """Complete the row from what the provider recorded about the call and append it.

        ``cost_usd`` is the cost the provider's reported usage gives, and null, as elsewhere in
        the log, when it is not known: when the answer carried no usage, or when the call failed
        before any answer came back. `genlog.total_spend` then counts the estimate for an
        answered call. A call that failed after an answer came back (one that could not be
        read) was still paid for, so its cost is kept.
        """
        cost = getattr(provider, "last_cost_usd", 0.0)
        row.update(
            status="error" if error is not None else "ok",
            error=f"{type(error).__name__}: {error}" if error is not None else None,
            request_id=getattr(provider, "last_request_id", None),
            response_id=getattr(provider, "last_response_id", None),
            usage=getattr(provider, "last_usage", None),
            **fields,
            cost_usd=round(cost, 6) if cost is not None and (error is None or cost) else None,
            duration_s=round(time.monotonic() - started, 2),
        )
        genlog.append_row(self.path, row)
        return row["id"]


def charged_usd(provider: SegmentationProvider | Verifier) -> float:
    """What the provider's last call counts against the spend cap: its reported cost, or its
    estimate when an answer came back without usage."""
    cost = provider.last_cost_usd
    return cost if cost is not None else provider.call_estimate_usd


class _NoCallLog:
    """Stands in for `CallLog` when nothing is to be logged (dry runs and unit tests)."""

    def start(self, *_args: Any, **_fields: Any) -> dict[str, Any]:
        return {}

    def finish(self, *_args: Any, **_fields: Any) -> None:
        return None


@dataclass
class Attempt:
    number: int
    hint_box: BBox | None
    candidates: list[Candidate]
    chosen: Candidate | None
    polygon: Polygon | None
    verdict: Verdict
    segment_cost_usd: float = 0.0
    verify_cost_usd: float = 0.0
    segmenter_raw: str | None = None
    segment_log_id: str | None = None
    verify_log_id: str | None = None


@dataclass
class HotspotResult:
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return bool(self.attempts) and self.attempts[-1].verdict.matches

    @property
    def final(self) -> Attempt | None:
        """The verified attempt, the only one whose outline may replace the scene's hotspot."""
        return self.attempts[-1] if self.verified else None

    @property
    def polygon(self) -> Polygon | None:
        """The verified outline, or None: a rejected outline must never reach the manifest."""
        final = self.final
        return final.polygon if final else None

    def verification_text(self) -> str:
        if self.verified:
            return self.attempts[-1].verdict.reason
        count = len(self.attempts)
        reasons = " ".join(f"Attempt {a.number}: {a.verdict.reason}" for a in self.attempts)
        return f"Not verified after {count} attempt{'s' if count != 1 else ''}. {reasons}"


def extract_hotspot(
    image_path: Path,
    concept: str,
    segmenter: SegmentationProvider,
    verifier: Verifier,
    *,
    image_size: tuple[int, int],
    hint_box: BBox | None = None,
    context: str | None = None,
    spend: Spend | None = None,
    log: CallLog | None = None,
    echo: Echo = lambda _: None,
) -> HotspotResult:
    """Segment, choose, clean and verify; on a "no", retry once with Claude's suggested box.

    The retry only happens when it would send something new: without a fresh suggestion the
    second request would be identical to the first, so the scene goes straight to manual work.
    With ``log``, every segmentation and verification call gets a generation-log row, failed
    ones included.
    """
    spend = spend if spend is not None else Spend(cap_usd=float("inf"))
    calls: CallLog | _NoCallLog = log if log is not None else _NoCallLog()
    result = HotspotResult()
    hint = hint_box
    for number in range(1, ATTEMPTS + 1):
        where = f" inside box {_fmt_box(hint)}" if hint else ""
        spend.reserve(segmenter.call_estimate_usd, "segmentation")
        row = calls.start(
            "segment",
            segmenter,
            number,
            concept=_concept(concept),
            hint_box=list(hint) if hint else None,
        )
        started = time.monotonic()
        try:
            candidates = segmenter.segment(image_path, concept, hint_box=hint)
        except Exception as error:
            spend.add(charged_usd(segmenter))  # an unreadable answer was still paid for
            calls.finish(row, segmenter, started, error)
            raise
        segment_cost = charged_usd(segmenter)
        spend.add(segment_cost)
        segment_log_id = calls.finish(row, segmenter, started)
        chosen = choose_candidate(candidates)
        echo(f"  attempt {number}{where}: {len(candidates)} candidate(s)")
        polygon: Polygon | None = None
        verify_cost = 0.0
        verify_log_id: str | None = None
        if chosen is None:
            verdict = Verdict(matches=False, reason=f"The segmenter found no {_concept(concept)}.")
        else:
            echo(
                f"    chose '{chosen.label}': {chosen.area:.2%} of the frame, centroid "
                f"{_fmt_point(chosen.centroid)}"
            )
            for line in reading_report(chosen):
                echo(f"    {line}")
            polygon = clean_polygon(chosen.polygon, image_size) or None
            if polygon is None:
                verdict = Verdict(matches=False, reason="The chosen outline was empty.")
            else:
                echo(
                    f"    cleaned: {len(polygon)} vertices, {geometry.area(polygon):.2%} of the "
                    f"frame, centroid {_fmt_point(geometry.centroid(polygon))}"
                )
                spend.reserve(verifier.call_estimate_usd, "verification")
                row = calls.start(
                    "verify",
                    verifier,
                    number,
                    concept=_concept(concept),
                    polygon_vertices=len(polygon),
                )
                started = time.monotonic()
                unreadable: VerificationError | None = None
                try:
                    verdict = verifier.verify(image_path, polygon, concept, context)
                except VerificationError as error:
                    unreadable = error
                    verdict = Verdict(matches=False, reason=f"Verification failed: {error}.")
                except Exception as error:
                    spend.add(charged_usd(verifier))
                    calls.finish(row, verifier, started, error)
                    raise
                verify_cost = charged_usd(verifier)
                spend.add(verify_cost)
                verify_log_id = calls.finish(
                    row,
                    verifier,
                    started,
                    unreadable,
                    served_model=getattr(verifier, "last_model", None),
                    matches=None if unreadable else verdict.matches,
                )
                echo(f"    verifier says {'yes' if verdict.matches else 'no'}: {verdict.reason}")
        result.attempts.append(
            Attempt(
                number=number,
                hint_box=hint,
                candidates=list(candidates),
                chosen=chosen,
                polygon=polygon,
                verdict=verdict,
                segment_cost_usd=segment_cost,
                verify_cost_usd=verify_cost,
                segmenter_raw=getattr(segmenter, "last_raw", None),
                segment_log_id=segment_log_id,
                verify_log_id=verify_log_id,
            )
        )
        if verdict.matches:
            break
        if verdict.suggested_box is None:
            break
        next_hint = expand_box(verdict.suggested_box, SUGGESTION_MARGIN)
        if next_hint == hint:
            break
        hint = next_hint
    return result


def reading_report(candidate: Candidate) -> list[str]:
    """What to print about how a Gemini mask was read: the reading used and, when even that
    reading leaves points outside the box, a warning. Nothing for other segmenters."""
    reading = candidate.reading
    if reading is None:
        return []
    lines = [f"read the mask points as {reading.describe()}"]
    if reading.outside > 0:
        lines.append(
            f"warning: even the best reading leaves {reading.outside:.0%} of the mask points "
            "outside the box Gemini gave, so the outline is probably garbled in places"
        )
    if order_is_ambiguous(candidate.readings):
        totals = {s.reading: s.total for s in candidate.readings}
        lines.append(
            f"warning: [x, y] and [y, x] fit the box equally well ({totals['xy-image']:.2f} "
            f"and {totals['yx-image']:.2f}), so the point order cannot be told from the box; "
            f"{READING_NAMES[reading.reading]} was used, so check the overlay"
        )
    return lines


def overlay_name(attempt: Attempt) -> str:
    """``hotspot-<n>.png``, with the mask reading when there is one: ``hotspot-1-yx-image.png``."""
    reading = attempt.chosen.reading if attempt.chosen else None
    suffix = f"-{reading.reading}" if reading else ""
    return f"hotspot-{attempt.number}{suffix}.png"


def _fmt_point(point: Sequence[float]) -> str:
    return f"({point[0]:.3f}, {point[1]:.3f})"


def _fmt_box(box: Sequence[float]) -> str:
    return "(" + ", ".join(f"{v:.3f}" for v in box) + ")"


def _indent(text: str, spaces: int) -> str:
    return "\n".join((" " * spaces + line) if line else line for line in text.splitlines())


# --------------------------------------------------------------------------------------------
# Writing results
# --------------------------------------------------------------------------------------------


# The fields that say what a verified hotspot was checked against; they go with the verdict.
PROVENANCE_FIELDS = ("verified_master_sha256", "verified_image", "verified_image_sha256")


def bridge_updates(
    result: HotspotResult,
    source: HotspotSource,
    master_sha256: str | None = None,
    image: str | None = None,
    image_sha256: str | None = None,
) -> dict[str, Any]:
    """The hotspot fields of ``bridge`` after a run; None removes a field.

    Only a verified outline replaces the polygon and its derived fields. It records what it was
    verified against: ``master_sha256``, the master, and ``image`` with ``image_sha256``, the
    picture actually segmented and shown to Claude (the graded copy when there is one), so a
    later change of either shows. After a failure the existing hotspot (perhaps verified
    earlier, perhaps drawn by hand) stays exactly as it was, together with its ``source`` and
    mask; the scene is only flagged for manual work, and the recorded hashes go with the
    verification they belonged to. The rejected outlines are kept for review as
    assets/review/<scene>/hotspot-<n>[-<reading>].png.
    """
    final = result.final
    text = result.verification_text()
    if final is None or not final.polygon:
        return {
            "verified": False,
            "verification": text,
            **dict.fromkeys(PROVENANCE_FIELDS),
            "needs_manual_hotspot": True,
        }
    polygon = [
        (round(float(x), POLYGON_DECIMALS), round(float(y), POLYGON_DECIMALS))
        for x, y in final.polygon
    ]
    x0, y0, x1, y1 = geometry.bbox(polygon)
    cx, cy = geometry.centroid(polygon)
    score = final.chosen.score if final.chosen else None
    return {
        "polygon": [[x, y] for x, y in polygon],
        "source": source,
        "verified": True,
        "verification": text,
        "verified_master_sha256": master_sha256,
        "verified_image": image,
        "verified_image_sha256": image_sha256,
        "score": round(float(score), 4) if score is not None else None,
        "needs_manual_hotspot": False,
        "bbox": [round(v, DERIVED_DECIMALS) for v in (x0, y0, x1, y1)],
        "centroid": [round(cx, DERIVED_DECIMALS), round(cy, DERIVED_DECIMALS)],
        "area": round(geometry.area(polygon), DERIVED_DECIMALS),
    }


def not_kept_updates(result: HotspotResult, problems: Sequence[Issue]) -> dict[str, Any]:
    """The bridge fields when Claude accepted an outline that breaks the hotspot rules.

    Claude judges whether the region is the object, not whether it suits the game: an outline
    can be exactly the object and still be too small to click (a slender lighthouse at 1.35% of
    the frame). Such an outline is not kept: the polygon, its derived fields and the mask stay
    as they were, and the scene is marked unverified with the reason, like any other failure.
    """
    reasons = "; ".join(issue.message for issue in problems)
    return {
        "verified": False,
        "verification": "Claude accepted the new outline, but it was not kept because it "
        f"breaks the hotspot rules ({reasons}). {result.verification_text()}",
        **dict.fromkeys(PROVENANCE_FIELDS),
        "needs_manual_hotspot": True,
    }


def master_changed(paths: RepoPaths, scene: Scene) -> bool | None:
    """Whether the scene's pictures differ from the ones its hotspot was verified against.

    Compares the master with its recorded hash and, when recorded, the picture that was
    segmented and shown to Claude (normally the graded copy): which file it was and its hash.
    So both a new master from `dsp select` and a re-graded copy count. None when nothing was
    recorded (hotspots verified before hashes were kept, most hand-drawn ones): there is then
    nothing to compare. A missing file counts as changed.
    """
    bridge = scene.bridge
    if bridge.verified_master_sha256 is None and bridge.verified_image_sha256 is None:
        return None
    if bridge.verified_master_sha256 is not None:
        master = paths.master(scene.id)
        if not master.exists() or sha256_file(master) != bridge.verified_master_sha256:
            return True
    if bridge.verified_image_sha256 is not None:
        current = _source_image(paths, scene.id)
        if current is None:
            return True
        if bridge.verified_image is not None and _rel(paths, current) != bridge.verified_image:
            return True
        if sha256_file(current) != bridge.verified_image_sha256:
            return True
    return False


# A fresh checkout writes <id>.graded.png and <id>.png moments apart (git writes the graded copy
# first, so the master looks about 10 ms newer), so only a master written clearly after its
# graded copy counts as newer.
GRADED_STALE_TOLERANCE_S = 10.0


def stale_graded_copy(paths: RepoPaths, scene_id: str) -> str | None:
    """Why the scene's graded copy cannot be trusted to show its master, or None.

    `dsp hotspot` segments the graded copy when there is one. After `dsp select` adopts a new
    master, the graded copy still shows the old picture until `dsp grade` runs, and an outline
    traced on it would be recorded as verified against the new master.
    """
    graded, master = paths.graded(scene_id), paths.master(scene_id)
    if not (graded.exists() and master.exists()):
        return None
    if master.stat().st_mtime <= graded.stat().st_mtime + GRADED_STALE_TOLERANCE_S:
        return None
    return (
        f"{_rel(paths, graded)} is older than {_rel(paths, master)}, so it may still show an "
        "earlier master; run `dsp grade` first so the hotspot is traced on the current picture"
    )


def hotspot_issues(scene: Scene) -> list[Issue]:
    """The hotspot rules of `dsp validate` (polygon shape, area, safe area) for one scene."""
    result = ValidationResult()
    _check_polygon(Manifest.model_construct(scenes=[scene]), result)
    return result.issues


def write_bridge_fields(paths: RepoPaths, scene_id: str, updates: dict[str, Any]) -> bool:
    """Patch content/scenes.yaml in place. Returns False when comments could not be kept."""
    text = paths.scenes_yaml.read_text(encoding="utf-8")
    patched = set_scene_fields(text, scene_id, "bridge", updates, BRIDGE_FIELD_ORDER)
    parse_manifest(yaml.safe_load(patched.text), paths.scenes_yaml)  # never write a bad file
    paths.scenes_yaml.write_text(patched.text, encoding="utf-8")
    return patched.in_place


def _attempt_record(attempt: Attempt) -> dict[str, Any]:
    chosen_reading = attempt.chosen.reading if attempt.chosen else None
    return {
        "number": attempt.number,
        "hint_box": attempt.hint_box,
        "candidates": [
            {
                "label": c.label,
                "score": c.score,
                "bbox": c.bbox,
                "area": round(c.area, DERIVED_DECIMALS),
                "centroid": c.centroid,
                "vertices": len(c.polygon),
                "reading": c.reading.reading if c.reading else None,
                "readings": {r.reading: r.as_record() for r in c.readings},
            }
            for c in attempt.candidates
        ],
        "chosen": attempt.chosen.label if attempt.chosen else None,
        "reading": chosen_reading.reading if chosen_reading else None,
        "points_outside_box": round(chosen_reading.outside, 4) if chosen_reading else None,
        "overlay": overlay_name(attempt) if attempt.polygon else None,
        "polygon": attempt.polygon,
        "verdict": attempt.verdict.model_dump(),
        "segment_cost_usd": round(attempt.segment_cost_usd, 6),
        "verify_cost_usd": round(attempt.verify_cost_usd, 6),
        "segmenter_raw": attempt.segmenter_raw,
        "log_ids": {"segment": attempt.segment_log_id, "verify": attempt.verify_log_id},
    }


# --------------------------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------------------------


@dataclass
class SceneOutcome:
    scene: str
    status: Literal["verified", "manual", "not_kept", "error", "skipped"]
    issues: list[Issue] = field(default_factory=list)


def _rel(paths: RepoPaths, path: Path) -> str:
    return str(path.relative_to(paths.root)) if path.is_relative_to(paths.root) else str(path)


def _source_image(paths: RepoPaths, scene_id: str) -> Path | None:
    for candidate in (paths.graded(scene_id), paths.master(scene_id)):
        if candidate.exists():
            return candidate
    return None


def _process_scene(
    paths: RepoPaths,
    scene: Scene,
    segmenter: SegmentationProvider,
    verifier: Verifier,
    *,
    hint_box: BBox | None,
    dry_run: bool,
    spend: Spend,
    echo: Echo,
) -> SceneOutcome:
    sid = scene.id
    image_path = _source_image(paths, sid)
    if image_path is None:
        echo(f"{sid}: error: no master image at {_rel(paths, paths.master(sid))}")
        return SceneOutcome(sid, "error")
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    concept = scene.bridge.object
    echo(
        f"{sid}: segmenting {_rel(paths, image_path)} ({image.width}x{image.height}) for "
        f"'{concept}' with {segmenter.model}"
    )
    log = None  # a dry run sends nothing, so it has no calls to log
    if not dry_run:
        sent = {"path": _rel(paths, image_path), "sha256": sha256_file(image_path)}
        log = CallLog(paths.generation_log, sid, sent)
    result = extract_hotspot(
        image_path,
        concept,
        segmenter,
        verifier,
        image_size=image.size,
        hint_box=hint_box,
        context=scene.bridge.placement,
        spend=spend,
        log=log,
        echo=echo,
    )

    master = paths.master(sid)
    updates = bridge_updates(
        result,
        segmenter.source,
        master_sha256=sha256_file(master) if master.exists() else None,
        image=_rel(paths, image_path),
        image_sha256=sha256_file(image_path),
    )
    issues = _issues_after(scene, updates)
    rule_errors = [issue for issue in issues if issue.level == "error"]
    not_kept = result.verified and bool(rule_errors)
    if not_kept:
        updates = not_kept_updates(result, rule_errors)
        issues = _issues_after(scene, updates)
    new_polygon = result.polygon if result.verified and not not_kept else None

    review = paths.review_dir(sid)
    review.mkdir(parents=True, exist_ok=True)
    for stale in review.glob("hotspot-*.png"):  # attempts from an earlier run
        stale.unlink()
    for attempt in result.attempts:
        if attempt.polygon:
            render_overlay(image, attempt.polygon).save(review / overlay_name(attempt))
    # hotspot.png always shows the hotspot the manifest holds after this run: the new outline
    # when it was verified and kept, otherwise the one that was there before. Every attempt is
    # one of the hotspot-<n>[-<reading>].png files above.
    overlay = review / "hotspot.png"
    kept = new_polygon or (scene.bridge.polygon if len(scene.bridge.polygon) >= 3 else None)
    if kept:
        render_overlay(image, kept).save(overlay)
        echo(f"  wrote {_rel(paths, overlay)}")
    elif overlay.exists():
        overlay.unlink()  # never leave an overlay from an earlier run looking current

    record = {
        "scene": sid,
        "image": _rel(paths, image_path),
        "concept": concept,
        "segmenter": segmenter.model,
        "verifier": verifier.model,
        "dry_run": dry_run,
        "verified": result.verified,
        "kept": new_polygon is not None,
        "not_kept_because": [issue.message for issue in rule_errors] if not_kept else None,
        "attempts": [_attempt_record(a) for a in result.attempts],
        "updates": updates,
    }
    (review / "hotspot.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    if not_kept:
        for issue in rule_errors:
            echo(f"  {issue.level}: the verified outline breaks a hotspot rule: {issue.message}")
        echo("  not kept: the outline stays as it was and the scene is marked unverified")

    if dry_run:
        echo("  would write to content/scenes.yaml, under bridge (verification is assumed):")
        echo(render_fields(updates, 6).rstrip("\n"))
    else:
        kept_comments = write_bridge_fields(paths, sid, updates)
        echo("  updated content/scenes.yaml")
        if not kept_comments:
            echo("  note: the file's layout forced a full rewrite, so its comments were dropped")
        if new_polygon:
            paths.masks.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rasterize(new_polygon, image.width, image.height)).save(paths.mask(sid))
            echo(f"  wrote {_rel(paths, paths.mask(sid))}")

    for issue in issues:
        echo(f"  {issue.level}: {issue.message}")
    if not issues:
        echo("  hotspot rules: ok")
    status: Literal["verified", "manual", "not_kept"]
    if new_polygon:
        status = "verified"
    else:
        status = "not_kept" if not_kept else "manual"
        echo("  needs a manual hotspot: open tools/annotate/index.html with this master")
    return SceneOutcome(sid, status, issues)


def _issues_after(scene: Scene, updates: dict[str, Any]) -> list[Issue]:
    """The hotspot rule issues the scene would have with ``updates`` applied."""
    bridge = scene.bridge.model_copy(update=updates)
    return hotspot_issues(scene.model_copy(update={"bridge": bridge}))


def run_hotspot_command(
    paths: RepoPaths,
    *,
    scene: str | None,
    all_scenes: bool,
    box: Sequence[float] | None,
    dry_run: bool,
    force: bool = False,
    echo: Echo = print,
    segmenter: SegmentationProvider | None = None,
    verifier: Verifier | None = None,
) -> int:
    """`dsp hotspot`. Returns the exit code: 0 when every scene verified cleanly, 1 otherwise,
    2 for invalid options. ``segmenter`` and ``verifier`` replace the real providers in tests.

    With ``--all``, scenes that are already verified are skipped unless ``force`` is set, so a
    re-run only pays for the scenes that still need work; hand-drawn hotspots are always
    skipped. A scene named with ``--scene`` is always processed.
    """
    if bool(scene) == bool(all_scenes):
        echo("error: pass exactly one of --scene <id> or --all")
        return 2
    hint_box: BBox | None = None
    if box is not None:
        if all_scenes:
            echo("error: --box steers a single scene; use it with --scene")
            return 2
        try:
            hint_box = check_box(box)
        except ValueError as error:
            echo(f"error: --box: {error}")
            return 2
    try:
        manifest = load_manifest(paths.scenes_yaml)
    except ManifestError as error:
        echo(f"error: {error}")
        return 1
    try:
        name = segmenter_name_from_env()
        cap = max_spend_from_env()
    except ValueError as error:
        echo(f"error: {error}")
        return 2

    if scene:
        try:
            targets = [manifest.scene(scene)]
        except KeyError:
            known = ", ".join(s.id for s in manifest.scenes)
            echo(f"error: no scene '{scene}' in content/scenes.yaml (scenes: {known})")
            return 1
        if targets[0].bridge.source == "manual":
            echo(f"{scene}: replacing a hand-drawn hotspot because it was asked for by name")
        elif master_changed(paths, targets[0]):
            echo(f"{scene}: its master or graded copy has changed since the hotspot was verified")
    else:
        targets = []
        for item in manifest.scenes:
            changed = master_changed(paths, item)
            if item.bridge.source == "manual":
                echo(f"{item.id}: kept its hand-drawn hotspot (use --scene {item.id} to redo it)")
                if changed:
                    echo(
                        "  warning: its master or graded copy has changed since then; redraw it "
                        "to match"
                    )
            elif item.bridge.verified and changed:
                echo(
                    f"{item.id}: its master or graded copy has changed since the hotspot was "
                    "verified, so it counts as unverified"
                )
                targets.append(item)
            elif item.bridge.verified and not force:
                unknown = "" if changed is False else " (no master hash recorded to compare)"
                echo(
                    f"{item.id}: already verified, skipped (use --force, or --scene {item.id}, "
                    f"to redo it){unknown}"
                )
            else:
                targets.append(item)

    real = make_segmenter(name)
    if dry_run:
        echo(
            "dry run: nothing is sent and content/scenes.yaml and assets/masks are not changed; "
            "review files are still written to assets/review/ (git-ignored)"
        )
        verifier = verifier or DryRunVerifier(echo)
    else:
        segmenter = segmenter or real
        verifier = verifier or ClaudeVerifier()
        per_scene = ATTEMPTS * (segmenter.call_estimate_usd + verifier.call_estimate_usd)
        estimate = per_scene * len(targets)
        echo(
            f"estimated spend: up to ${estimate:.2f} for {len(targets)} scene(s) "
            f"(cap ${cap:.2f}, from DSP_MAX_SPEND_USD)"
        )
        if estimate > cap:
            echo("error: the estimate exceeds the cap; raise DSP_MAX_SPEND_USD or do fewer scenes")
            return 1
    spend = Spend(cap_usd=cap)

    outcomes: list[SceneOutcome] = []
    for target in targets:
        problem = stale_graded_copy(paths, target.id)
        if problem:
            echo(f"{target.id}: error: {problem}")
            outcomes.append(SceneOutcome(target.id, "error"))
            continue
        scene_segmenter: SegmentationProvider
        if dry_run:
            scene_segmenter = segmenter or MaskFileSegmenter(
                paths.mask(target.id), list(target.bridge.polygon), real.source
            )
            image_path = _source_image(paths, target.id)
            if image_path is not None:
                request = real.request(image_path, target.bridge.object, hint_box, elide=True)
                echo(f"{target.id}: would send to {real.model}:")
                echo(_indent(json.dumps(request, indent=2), 4))
        else:
            assert segmenter is not None
            scene_segmenter = segmenter
        try:
            outcomes.append(
                _process_scene(
                    paths,
                    target,
                    scene_segmenter,
                    verifier,
                    hint_box=hint_box,
                    dry_run=dry_run,
                    spend=spend,
                    echo=echo,
                )
            )
        except SpendLimitExceeded as error:
            echo(f"{target.id}: stopped: {error}")
            outcomes.append(SceneOutcome(target.id, "error"))
            break
        except Exception as error:  # one failing scene must not stop an --all run
            echo(f"{target.id}: error: {type(error).__name__}: {error}")
            outcomes.append(SceneOutcome(target.id, "error"))

    verified = [o.scene for o in outcomes if o.status == "verified"]
    manual = [o.scene for o in outcomes if o.status == "manual"]
    failed = [o.scene for o in outcomes if o.status == "error"]
    broken = [o.scene for o in outcomes if any(i.level == "error" for i in o.issues)]
    not_kept = [o.scene for o in outcomes if o.status == "not_kept"]
    echo(
        f"done: {len(verified)} verified, {len(manual)} need a manual hotspot, "
        f"{len(failed)} failed, {len(broken)} break the hotspot rules, {len(not_kept)} verified "
        "but not kept because they break the hotspot rules"
    )
    if not dry_run:
        echo(
            f"spent: ${spend.spent_usd:.4f} (from the token usage the APIs reported, or the "
            "estimate for an answer that reported none)"
        )
        echo(genlog.spend_summary(paths.generation_log))
    if dry_run:
        return 1 if failed or broken or not_kept else 0
    return 1 if manual or failed or broken or not_kept else 0

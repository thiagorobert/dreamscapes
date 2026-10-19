"""`dsp hotspot`: turn each scene's bridge object into its clickable polygon.

For one scene (plan section 6.2):

1. Segment the bridge object in the graded master (the master when no graded copy exists) with
   a `SegmentationProvider`: Gemini by default, SAM 3 on fal with
   ``DSP_SEGMENTATION_PROVIDER=sam3``.
2. When several instances come back, keep the largest one whose centroid is inside the safe area.
3. Clean the outline (`clean_polygon`): rasterise it, keep the largest connected component, grow
   it by about 0.5% of the image width so the edges are forgiving, and simplify it to at most 40
   vertices.
4. Ask Claude whether the tinted region is exactly the object and nothing else
   (`verify_with_claude`). On "no", segment once more with the bounding box Claude suggests
   written into the prompt; on a second "no", mark the scene ``needs_manual_hotspot`` so a person
   draws it with tools/annotate/index.html.
5. Write the polygon, its derived geometry and the verdict back to content/scenes.yaml (only the
   hotspot fields; comments and everything else are kept), save the overlay to
   assets/review/<scene>/hotspot.png and the mask to assets/masks/<scene>.png, and report the
   hotspot rules that `dsp validate` enforces.

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
  The real-API smoke test settled it: the points are 0-1000 across the whole image, in [x, y]
  order (a telephone box with box_2d [409, 569, 746, 665] came back with mask x in 571-664 and
  y in 409-746). `detect_mask_frame` checks this per entry, so a model that switches to
  box-relative points is still read correctly.
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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import cv2
import numpy as np
import yaml
from PIL import Image
from pydantic import BaseModel, Field, ValidationError, field_validator

from . import geometry
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
MaskFrame = Literal["box", "image"]

# The model's field order, used to place hotspot fields that a scene does not have yet.
BRIDGE_FIELD_ORDER = tuple(Bridge.model_fields)


# --------------------------------------------------------------------------------------------
# Candidates and the provider protocol
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One instance of the object that a segmenter found."""

    polygon: Polygon
    bbox: BBox
    label: str
    score: float | None = None

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
    model: str
    call_estimate_usd: float
    last_cost_usd: float
    """Cost of the most recent `segment` call, from the reported usage."""

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


def detect_mask_frame(box_2d: Sequence[float], mask: Sequence[Sequence[float]]) -> MaskFrame:
    """Whether mask points are 0-1000 across the box or across the whole image.

    Read as whole-image coordinates, a box-relative outline spreads over most of 0-1000 wherever
    the box is, so it does not fit inside the box. An image-relative outline sits inside its box
    and fills most of it. The two readings only coincide when the box covers nearly the whole
    image, where either answer gives the same polygon.
    """
    x0, y0, x1, y1 = (v * 1000 for v in _box_from_1000(box_2d))
    xs = [p[0] for p in mask]
    ys = [p[1] for p in mask]
    tolerance = 20  # 2% of the image
    inside = (
        min(xs) >= x0 - tolerance
        and max(xs) <= x1 + tolerance
        and min(ys) >= y0 - tolerance
        and max(ys) <= y1 + tolerance
    )
    fills = (max(xs) - min(xs)) >= 0.5 * (x1 - x0) and (max(ys) - min(ys)) >= 0.5 * (y1 - y0)
    return "image" if inside and fills else "box"


def gemini_mask_to_polygon(
    box_2d: Sequence[float],
    mask: Sequence[Sequence[float]],
    frame: MaskFrame | Literal["auto"] = "auto",
) -> Polygon:
    """Convert a Gemini mask polygon (0-1000) to normalised image coordinates.

    ``frame="box"`` reads the points as 0-1000 across the bounding box, ``"image"`` as 0-1000
    across the image, and ``"auto"`` decides per entry with `detect_mask_frame`.
    """
    x0, y0, x1, y1 = _box_from_1000(box_2d)
    if frame == "auto":
        frame = detect_mask_frame(box_2d, mask)
    polygon: Polygon = []
    for px, py in mask:
        u, v = min(max(px / 1000.0, 0.0), 1.0), min(max(py / 1000.0, 0.0), 1.0)
        if frame == "box":
            polygon.append((x0 + u * (x1 - x0), y0 + v * (y1 - y0)))
        else:
            polygon.append((u, v))
    return polygon


def gemini_candidate(entry: _LenientMask, frame: MaskFrame | Literal["auto"] = "auto"):
    """A candidate from one response entry; a missing mask falls back to the box itself."""
    bbox = _box_from_1000(entry.box_2d)
    if len(entry.mask) >= 3:
        polygon = gemini_mask_to_polygon(entry.box_2d, entry.mask, frame)
        if geometry.simplicity_problem(polygon) is not None:
            # A crossing outline has its vertices in the wrong order, and filling it leaves
            # holes inside the object (seen in the smoke test: a C shape that missed the body
            # of the telephone box). The points still mark the object's extent, so use their
            # convex hull, which is always simple.
            polygon = convex_hull(polygon)
    else:
        x0, y0, x1, y1 = bbox
        polygon = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if geometry.area(polygon) <= 0:
        return None
    return Candidate(polygon=polygon, bbox=bbox, label=entry.label)


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
    call_estimate_usd = GEMINI_CALL_ESTIMATE_USD

    def __init__(
        self,
        client: Any | None = None,
        model: str = GEMINI_MODEL,
        frame: MaskFrame | Literal["auto"] = "auto",
    ):
        self._client = client
        self.model = model
        self.frame = frame
        self.thinking_level = GEMINI_THINKING_LEVEL
        self.last_cost_usd = 0.0
        self.last_raw: str | None = None

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
        self.last_raw = interaction.output_text or ""
        self.last_cost_usd = gemini_cost(getattr(interaction, "usage", None))
        candidates = (gemini_candidate(e, self.frame) for e in parse_gemini_masks(self.last_raw))
        return [c for c in candidates if c is not None]


def _rejects_thinking_level(error: Exception) -> bool:
    return getattr(error, "status_code", None) == 400 and "thinking level" in str(error).lower()


def _stronger_thinking_level(level: str) -> str | None:
    if level not in GEMINI_THINKING_LEVELS:
        return None
    index = GEMINI_THINKING_LEVELS.index(level) + 1
    return GEMINI_THINKING_LEVELS[index] if index < len(GEMINI_THINKING_LEVELS) else None


def gemini_cost(usage: Any) -> float:
    if usage is None:
        return 0.0
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
    model = SAM3_APP
    call_estimate_usd = SAM3_CALL_ESTIMATE_USD

    def __init__(self, client: Any | None = None, fetch: Callable[[str], bytes] | None = None):
        self._client = client
        self._fetch = fetch
        self.last_cost_usd = SAM3_CALL_ESTIMATE_USD
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
        result = self.client.subscribe(
            SAM3_APP, arguments=self.request(image_path, concept, hint_box)
        )
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

    model = "fake segmenter"
    call_estimate_usd = 0.0

    def __init__(self, responses: Sequence[Sequence[Candidate]], source: HotspotSource = "gemini"):
        if not responses:
            raise ValueError("FakeSegmenter needs at least one response")
        self.responses = [list(r) for r in responses]
        self.source = source
        self.calls: list[tuple[Path, str, BBox | None]] = []
        self.last_cost_usd = 0.0

    def segment(
        self, image_path: Path, concept: str, hint_box: BBox | None = None
    ) -> list[Candidate]:
        self.calls.append((Path(image_path), concept, hint_box))
        return list(self.responses[min(len(self.calls), len(self.responses)) - 1])


class MaskFileSegmenter:
    """Dry-run stand-in: outlines the scene's committed mask instead of calling a model.

    Placeholder scenes ship a mask of their glowing shape, which is exactly what a segmenter
    should find. Without a mask file it returns the polygon already in the manifest.
    """

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
    model: str
    call_estimate_usd: float
    last_cost_usd: float

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


def claude_cost(usage: Any) -> float:
    if usage is None:
        return 0.0
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

    model = CLAUDE_MODEL
    call_estimate_usd = CLAUDE_CALL_ESTIMATE_USD

    def __init__(self, client: Any | None = None, effort: str = CLAUDE_EFFORT):
        self._client = client
        self.effort = effort
        self.last_cost_usd = 0.0
        self.last_model: str | None = None
        self.last_request_id: str | None = None

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
        response = self.client.beta.messages.create(
            **self.request(image_path, polygon, concept, context)
        )
        self.last_cost_usd = claude_cost(getattr(response, "usage", None))
        self.last_model = getattr(response, "model", None)
        self.last_request_id = getattr(response, "_request_id", None)
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

    model = "fake verifier"
    call_estimate_usd = 0.0

    def __init__(self, verdicts: Sequence[Verdict]):
        if not verdicts:
            raise ValueError("FakeVerifier needs at least one verdict")
        self.verdicts = list(verdicts)
        self.calls: list[tuple[Path, Polygon, str, str | None]] = []
        self.last_cost_usd = 0.0

    def verify(self, image_path, polygon, concept, context=None) -> Verdict:
        self.calls.append((Path(image_path), [tuple(p) for p in polygon], concept, context))
        return self.verdicts[min(len(self.calls), len(self.verdicts)) - 1]


class DryRunVerifier:
    """Prints the request Claude would get and assumes "yes" so the rest of the run shows."""

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


@dataclass
class HotspotResult:
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return bool(self.attempts) and self.attempts[-1].verdict.matches

    @property
    def final(self) -> Attempt | None:
        """The attempt whose polygon is kept: the verified one, else the last with a polygon."""
        if self.verified:
            return self.attempts[-1]
        with_polygon = [a for a in self.attempts if a.polygon]
        return with_polygon[-1] if with_polygon else None

    @property
    def polygon(self) -> Polygon | None:
        final = self.final
        return final.polygon if final else None

    def verification_text(self) -> str:
        if self.verified:
            return self.attempts[-1].verdict.reason
        reasons = " ".join(f"Attempt {a.number}: {a.verdict.reason}" for a in self.attempts)
        return f"Not verified after {len(self.attempts)} attempts. {reasons}"


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
    echo: Echo = lambda _: None,
) -> HotspotResult:
    """Segment, choose, clean and verify; on a "no", retry once with Claude's suggested box."""
    spend = spend if spend is not None else Spend(cap_usd=float("inf"))
    result = HotspotResult()
    hint = hint_box
    for number in range(1, ATTEMPTS + 1):
        where = f" inside box {_fmt_box(hint)}" if hint else ""
        spend.reserve(segmenter.call_estimate_usd, "segmentation")
        candidates = segmenter.segment(image_path, concept, hint_box=hint)
        segment_cost = segmenter.last_cost_usd
        spend.add(segment_cost)
        chosen = choose_candidate(candidates)
        echo(f"  attempt {number}{where}: {len(candidates)} candidate(s)")
        polygon: Polygon | None = None
        verify_cost = 0.0
        if chosen is None:
            verdict = Verdict(matches=False, reason=f"The segmenter found no {_concept(concept)}.")
        else:
            echo(
                f"    chose '{chosen.label}': {chosen.area:.2%} of the frame, centroid "
                f"{_fmt_point(chosen.centroid)}"
            )
            polygon = clean_polygon(chosen.polygon, image_size) or None
            if polygon is None:
                verdict = Verdict(matches=False, reason="The chosen outline was empty.")
            else:
                echo(
                    f"    cleaned: {len(polygon)} vertices, {geometry.area(polygon):.2%} of the "
                    f"frame, centroid {_fmt_point(geometry.centroid(polygon))}"
                )
                spend.reserve(verifier.call_estimate_usd, "verification")
                try:
                    verdict = verifier.verify(image_path, polygon, concept, context)
                except VerificationError as error:
                    verdict = Verdict(matches=False, reason=f"Verification failed: {error}.")
                verify_cost = verifier.last_cost_usd
                spend.add(verify_cost)
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
            )
        )
        if verdict.matches:
            break
        if verdict.suggested_box is not None:
            hint = expand_box(verdict.suggested_box, SUGGESTION_MARGIN)
        else:
            hint = hint_box
    return result


def _fmt_point(point: Sequence[float]) -> str:
    return f"({point[0]:.3f}, {point[1]:.3f})"


def _fmt_box(box: Sequence[float]) -> str:
    return "(" + ", ".join(f"{v:.3f}" for v in box) + ")"


def _indent(text: str, spaces: int) -> str:
    return "\n".join((" " * spaces + line) if line else line for line in text.splitlines())


# --------------------------------------------------------------------------------------------
# Writing results
# --------------------------------------------------------------------------------------------


def bridge_updates(result: HotspotResult, source: HotspotSource) -> dict[str, Any]:
    """The hotspot fields of ``bridge`` after a run; None removes a field."""
    final = result.final
    text = result.verification_text()
    if final is None or not final.polygon:
        # Nothing usable was found: keep whatever polygon is there and flag the scene.
        return {"verified": False, "verification": text, "needs_manual_hotspot": True}
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
        "verified": result.verified,
        "verification": text,
        "score": round(float(score), 4) if score is not None else None,
        "needs_manual_hotspot": not result.verified,
        "bbox": [round(v, DERIVED_DECIMALS) for v in (x0, y0, x1, y1)],
        "centroid": [round(cx, DERIVED_DECIMALS), round(cy, DERIVED_DECIMALS)],
        "area": round(geometry.area(polygon), DERIVED_DECIMALS),
    }


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
            }
            for c in attempt.candidates
        ],
        "chosen": attempt.chosen.label if attempt.chosen else None,
        "polygon": attempt.polygon,
        "verdict": attempt.verdict.model_dump(),
        "segment_cost_usd": round(attempt.segment_cost_usd, 6),
        "verify_cost_usd": round(attempt.verify_cost_usd, 6),
        "segmenter_raw": attempt.segmenter_raw,
    }


# --------------------------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------------------------


@dataclass
class SceneOutcome:
    scene: str
    status: Literal["verified", "manual", "error", "skipped"]
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
    result = extract_hotspot(
        image_path,
        concept,
        segmenter,
        verifier,
        image_size=image.size,
        hint_box=hint_box,
        context=scene.bridge.placement,
        spend=spend,
        echo=echo,
    )

    review = paths.review_dir(sid)
    review.mkdir(parents=True, exist_ok=True)
    for stale in review.glob("hotspot-*.png"):  # attempts from an earlier run
        stale.unlink()
    for attempt in result.attempts:
        if attempt.polygon:
            render_overlay(image, attempt.polygon).save(review / f"hotspot-{attempt.number}.png")
    overlay = review / "hotspot.png"
    if result.polygon:
        render_overlay(image, result.polygon).save(overlay)
        echo(f"  wrote {_rel(paths, overlay)}")
    elif overlay.exists():
        overlay.unlink()  # never leave an overlay from an earlier run looking current

    updates = bridge_updates(result, segmenter.source)
    record = {
        "scene": sid,
        "image": _rel(paths, image_path),
        "concept": concept,
        "segmenter": segmenter.model,
        "verifier": verifier.model,
        "dry_run": dry_run,
        "verified": result.verified,
        "attempts": [_attempt_record(a) for a in result.attempts],
        "updates": updates,
    }
    (review / "hotspot.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    new_bridge = scene.bridge.model_copy(update={k: v for k, v in updates.items()})
    issues = hotspot_issues(scene.model_copy(update={"bridge": new_bridge}))

    if dry_run:
        echo("  would write to content/scenes.yaml, under bridge (verification is assumed):")
        echo(render_fields(updates, 6).rstrip("\n"))
    else:
        kept_comments = write_bridge_fields(paths, sid, updates)
        echo("  updated content/scenes.yaml")
        if not kept_comments:
            echo("  note: the file's layout forced a full rewrite, so its comments were dropped")
        if result.polygon:
            paths.masks.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rasterize(result.polygon, image.width, image.height)).save(
                paths.mask(sid)
            )
            echo(f"  wrote {_rel(paths, paths.mask(sid))}")

    for issue in issues:
        echo(f"  {issue.level}: {issue.message}")
    if not issues:
        echo("  hotspot rules: ok")
    if result.verified:
        status: Literal["verified", "manual"] = "verified"
    else:
        status = "manual"
        echo("  needs a manual hotspot: open tools/annotate/index.html with this master")
    return SceneOutcome(sid, status, issues)


def run_hotspot_command(
    paths: RepoPaths,
    *,
    scene: str | None,
    all_scenes: bool,
    box: Sequence[float] | None,
    dry_run: bool,
    echo: Echo = print,
    segmenter: SegmentationProvider | None = None,
    verifier: Verifier | None = None,
) -> int:
    """`dsp hotspot`. Returns the exit code: 0 when every scene verified cleanly, 1 otherwise,
    2 for invalid options. ``segmenter`` and ``verifier`` replace the real providers in tests.
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
    else:
        targets = []
        for item in manifest.scenes:
            if item.bridge.source == "manual":
                echo(f"{item.id}: kept its hand-drawn hotspot (use --scene {item.id} to redo it)")
            else:
                targets.append(item)

    real = make_segmenter(name)
    if dry_run:
        echo("dry run: nothing is sent and content/scenes.yaml is not changed")
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
    echo(
        f"done: {len(verified)} verified, {len(manual)} need a manual hotspot, "
        f"{len(failed)} failed, {len(broken)} break the hotspot rules"
    )
    if not dry_run:
        echo(f"spent: ${spend.spent_usd:.4f} (from the token usage the APIs reported)")
    if dry_run:
        return 1 if failed or broken else 0
    return 1 if manual or failed or broken else 0

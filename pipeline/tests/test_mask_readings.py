"""Regression tests for reading Gemini mask points, built from real segmentation responses.

tests/fixtures/gemini_segmentation_responses.json holds every Gemini response from the first
`dsp hotspot` run on the ten real scenes (one or two attempts each), reduced to box_2d, mask
and label. On five scenes (rooftop-court, balloon-salt-flats, endless-library,
overgrown-greenhouse, mountain-road-phonebox) the mask points came back as [y, x] while box_2d
kept its documented [ymin, xmin, ymax, xmax] order; read as [x, y], every outline collapsed into
a sliver and Claude rejected it. The other five came back as documented. Two of the [y, x]
scenes also have masks that switch order part-way round the outline.

Gemini has not been seen to send box-relative points, so those cases are made from the real
responses by re-expressing their points across their own box.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from dreamscapes_pipeline import geometry
from dreamscapes_pipeline.hotspots import (
    GeminiSegmenter,
    choose_candidate,
    choose_mask_reading,
    gemini_candidate,
    gemini_mask_to_polygon,
    parse_gemini_masks,
    reading_report,
    score_mask_readings,
)

FIXTURE = Path(__file__).parent / "fixtures" / "gemini_segmentation_responses.json"
RESPONSES = json.loads(FIXTURE.read_text(encoding="utf-8"))

FLAGGED = (
    "rooftop-court",
    "balloon-salt-flats",
    "endless-library",
    "overgrown-greenhouse",
    "mountain-road-phonebox",
)

# (scene, attempt index, reading that must win, share of its points inside the box)
EXPECTED = [
    ("balloon-salt-flats", 0, "yx-image", 1.0),
    ("balloon-salt-flats", 1, "yx-image", 1.0),
    ("cliff-cottage", 0, "xy-image", 1.0),
    ("drowned-street", 0, "xy-image", 1.0),
    ("endless-library", 0, "yx-image", 0.58),  # switches from [x, y] to [y, x] part-way
    ("endless-library", 1, "yx-image", 1.0),
    ("fog-ship-deck", 0, "xy-image", 1.0),
    ("fog-ship-deck", 1, "xy-image", 1.0),
    ("lighthouse-gallery", 0, "xy-image", 1.0),
    ("mountain-road-phonebox", 0, "yx-image", 1.0),
    ("overgrown-greenhouse", 0, "yx-image", 0.92),  # starts as [x, y], then [y, x]
    ("overgrown-greenhouse", 1, "yx-image", 0.97),  # the same, for a shorter stretch
    ("prison-yard", 0, "yx-image", 1.0),
    ("prison-yard", 1, "xy-image", 1.0),
    ("rooftop-court", 0, "yx-image", 1.0),
    ("rooftop-court", 1, "yx-image", 1.0),
]


def entry(scene: str, attempt: int, index: int = 0):
    return parse_gemini_masks(json.dumps(RESPONSES[scene]["responses"][attempt]))[index]


def box_of(item) -> tuple[float, float, float, float]:
    ymin, xmin, ymax, xmax = (v / 1000 for v in item.box_2d)
    return xmin, ymin, xmax, ymax


def reading_of(box_2d, mask) -> str:
    return choose_mask_reading(score_mask_readings(box_2d, mask)).reading


def same_points(actual, expected, tol=1e-9) -> bool:
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    return a.shape == b.shape and bool(np.allclose(a, b, atol=tol, rtol=0))


def share_inside(polygon, box, tol=0.01) -> float:
    x0, y0, x1, y1 = box
    inside = [x0 - tol <= x <= x1 + tol and y0 - tol <= y <= y1 + tol for x, y in polygon]
    return sum(inside) / len(inside)


def test_the_fixture_holds_every_saved_response():
    assert len(RESPONSES) == 10
    pairs = {(s, i) for s, i, _, _ in EXPECTED} | {("mountain-road-phonebox", 1)}
    assert pairs == {(s, i) for s, d in RESPONSES.items() for i in range(len(d["responses"]))}
    assert all(len(r["boxes"]) == 1 for d in RESPONSES.values() for r in d["responses"])


@pytest.mark.parametrize(
    ("scene", "attempt", "reading", "inside"),
    EXPECTED,
    ids=[f"{s}-{i + 1}" for s, i, _, _ in EXPECTED],
)
def test_each_saved_response_is_read_the_way_that_fits_its_box(scene, attempt, reading, inside):
    item = entry(scene, attempt)
    candidate = gemini_candidate(item)
    assert candidate is not None
    assert candidate.reading.reading == reading
    assert candidate.reading.inside == pytest.approx(inside, abs=0.01)
    assert [r.reading for r in candidate.readings] == [
        "xy-image",
        "yx-image",
        "xy-box",
        "yx-box",
    ]
    # The winner is clear, not decided by the tie rule: it scores at least twice the runner-up
    # (0.57 against 0.28 for the worst mixed-order mask, above 0.9 against below 0.06 for most).
    runner_up = max(r.total for r in candidate.readings if r.reading != reading)
    assert candidate.reading.total > 1.9 * runner_up
    assert candidate.reading.total - runner_up > 0.1
    # Warn exactly when even the best reading leaves points outside the box.
    warned = any(line.startswith("warning:") for line in reading_report(candidate))
    assert warned == (inside < 1.0)


@pytest.mark.parametrize(
    ("scene", "attempt"),
    [
        ("rooftop-court", 0),
        ("rooftop-court", 1),
        ("balloon-salt-flats", 0),
        ("balloon-salt-flats", 1),
        ("endless-library", 1),
        ("mountain-road-phonebox", 0),
    ],
)
def test_the_flagged_scenes_now_trace_the_object_instead_of_a_sliver(scene, attempt):
    # Masks that are entirely [y, x]. The old reader found points outside the box when reading
    # them as [x, y], decided they must be box-relative and squeezed them into a sliver.
    item = entry(scene, attempt)
    box = box_of(item)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    fixed = gemini_candidate(item)
    old = gemini_candidate(item, "xy-box")
    assert fixed.reading.reading == "yx-image"
    assert share_inside(fixed.polygon, box) == 1.0
    assert geometry.area(fixed.polygon) > 0.3 * box_area
    assert geometry.area(old.polygon) < 0.05 * box_area


def test_a_degenerate_mask_gives_no_candidate():
    # The phone box's second answer is one straight line of points: no reading has an area.
    item = entry("mountain-road-phonebox", 1)
    assert {p[0] for p in item.mask} == {641}
    assert all(r.total == 0 for r in score_mask_readings(item.box_2d, item.mask))
    assert gemini_candidate(item) is None


def test_mixed_order_masks_warn_and_still_take_the_majority_order():
    item = entry("overgrown-greenhouse", 0)
    candidate = gemini_candidate(item)
    lines = reading_report(candidate)
    assert lines[0].startswith("read the mask points as [y, x] across the image (yx-image): 92%")
    assert lines[1].startswith("warning: even the best reading leaves 8% of the mask points")


def to_box_relative(item, order: str) -> list[list[float]]:
    """Re-express a whole-image mask as 0-1000 across its own box, in ``order``."""
    x0, y0, x1, y1 = box_of(item)
    polygon = gemini_mask_to_polygon(item.box_2d, item.mask)  # the reading that fits
    relative = [[(x - x0) / (x1 - x0) * 1000, (y - y0) / (y1 - y0) * 1000] for x, y in polygon]
    return relative if order == "xy" else [[b, a] for a, b in relative]


@pytest.mark.parametrize(
    ("scene", "attempt"),
    [("cliff-cottage", 0), ("drowned-street", 0), ("lighthouse-gallery", 0), ("rooftop-court", 0)],
)
def test_box_relative_masks_are_recognised(scene, attempt):
    item = entry(scene, attempt)
    original = gemini_mask_to_polygon(item.box_2d, item.mask)
    relative = to_box_relative(item, "xy")
    scores = score_mask_readings(item.box_2d, relative)
    best = max(scores, key=lambda s: s.total)
    assert best.reading in ("xy-box", "yx-box")
    # Equal, except that a box-relative point cannot leave its box (cliff-cottage overshoots by 2)
    assert same_points(gemini_mask_to_polygon(item.box_2d, relative), original, tol=0.0025)


def test_box_relative_order_cannot_be_told_apart_so_the_tie_goes_to_x_y():
    # Within its own 0-1000 frame a box is square, so both orders fit it equally well; only
    # the image could tell them apart. The documented order wins, and forcing the other works.
    item = entry("rooftop-court", 0)
    original = gemini_mask_to_polygon(item.box_2d, item.mask)
    swapped = to_box_relative(item, "yx")
    scores = {s.reading: s.total for s in score_mask_readings(item.box_2d, swapped)}
    assert scores["xy-box"] == pytest.approx(scores["yx-box"])
    assert reading_of(item.box_2d, swapped) == "xy-box"
    assert same_points(gemini_mask_to_polygon(item.box_2d, swapped, "yx-box"), original)


def test_the_gemini_segmenter_reads_a_saved_response(tmp_path):
    image = tmp_path / "scene.png"
    Image.new("RGB", (2048, 1152), (40, 50, 70)).save(image)

    class Interactions:
        def create(self, **_kwargs):
            usage = SimpleNamespace(
                total_input_tokens=1500, total_output_tokens=400, total_thought_tokens=0
            )
            text = json.dumps(RESPONSES["rooftop-court"]["responses"][1])
            return SimpleNamespace(output_text=text, usage=usage, id="interaction-1")

    segmenter = GeminiSegmenter(client=SimpleNamespace(interactions=Interactions()))
    candidates = segmenter.segment(image, "hot-air balloon")
    chosen = choose_candidate(candidates)
    assert chosen.reading.reading == "yx-image"
    x0, y0, x1, y1 = chosen.bbox
    assert chosen.centroid[0] == pytest.approx((x0 + x1) / 2, abs=0.02)
    assert chosen.centroid[1] == pytest.approx((y0 + y1) / 2, abs=0.04)
    assert share_inside(chosen.polygon, chosen.bbox) == 1.0
    forced = GeminiSegmenter(
        client=SimpleNamespace(interactions=Interactions()), reading="xy-image"
    ).segment(image, "hot-air balloon")
    assert forced[0].reading.reading == "xy-image"
    assert share_inside(forced[0].polygon, chosen.bbox) < 0.5

from __future__ import annotations

import base64
import io
import json
import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image

from dreamscapes_pipeline import genlog, geometry, hotspots
from dreamscapes_pipeline.hotspots import (
    CallLog,
    Candidate,
    ClaudeVerifier,
    FakeSegmenter,
    FakeVerifier,
    GeminiSegmenter,
    Sam3Segmenter,
    Spend,
    SpendLimitExceeded,
    Verdict,
    VerificationError,
    box_to_1000,
    bridge_updates,
    check_box,
    choose_candidate,
    choose_mask_reading,
    clean_polygon,
    convex_hull,
    expand_box,
    extract_hotspot,
    gemini_candidate,
    gemini_mask_to_polygon,
    mask_to_polygon,
    parse_gemini_masks,
    parse_verdict,
    rasterize,
    render_closeup,
    render_overlay,
    score_mask_readings,
)

W, H = 2048, 1152


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Building a real API client in these tests is a bug: fail loudly instead."""

    def injected_only(self):
        if self._client is None:
            raise AssertionError("a test tried to build a real API client")
        return self._client

    for provider in (GeminiSegmenter, ClaudeVerifier, Sam3Segmenter):
        monkeypatch.setattr(provider, "client", property(injected_only))


def disk_mask(cx: float, cy: float, radius_px: float) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask, (round(cx * W), round(cy * H)), round(radius_px), 255, -1)
    return mask


def box_polygon(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def same_points(actual, expected, tol=1e-9) -> bool:
    """pytest.approx does not compare lists of points, so compare them as arrays."""
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    return a.shape == b.shape and bool(np.allclose(a, b, atol=tol, rtol=0))


def assert_valid(polygon, max_vertices=40):
    assert 3 <= len(polygon) <= max_vertices
    assert geometry.simplicity_problem(polygon) is None
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in polygon)
    assert all(type(x) is float and type(y) is float for x, y in polygon)


# --- polygonisation of raster masks ----------------------------------------------------------


def test_circle_mask_becomes_a_round_polygon_with_few_vertices():
    radius = 120
    polygon = mask_to_polygon(disk_mask(0.5, 0.5, radius), dilate_fraction=0)
    assert_valid(polygon)
    expected = math.pi * radius**2 / (W * H)
    assert geometry.area(polygon) == pytest.approx(expected, rel=0.03)
    assert geometry.centroid(polygon) == pytest.approx((0.5, 0.5), abs=0.002)


def test_dilation_grows_the_outline_by_half_a_percent_of_the_width():
    radius = 120
    grown = mask_to_polygon(disk_mask(0.5, 0.5, radius))  # default dilation
    margin = 0.005 * W
    expected = math.pi * (radius + margin) ** 2 / (W * H)
    assert geometry.area(grown) == pytest.approx(expected, rel=0.03)


def test_l_shape_keeps_its_notch():
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[300:800, 600:800] = 255  # vertical arm
    mask[600:800, 600:1200] = 255  # horizontal arm
    polygon = mask_to_polygon(mask, dilate_fraction=0)
    assert_valid(polygon)
    assert len(polygon) <= 8
    assert geometry.point_in_polygon((700 / W, 400 / H), polygon)  # top of the vertical arm
    assert geometry.point_in_polygon((1100 / W, 700 / H), polygon)  # end of the horizontal arm
    assert not geometry.point_in_polygon((1000 / W, 450 / H), polygon)  # the notch
    expected = (200 * 500 + 400 * 200) / (W * H)
    assert geometry.area(polygon) == pytest.approx(expected, rel=0.02)


def test_two_blobs_keep_only_the_larger_one():
    mask = disk_mask(0.3, 0.5, 60) | disk_mask(0.7, 0.5, 140)
    polygon = mask_to_polygon(mask, dilate_fraction=0)
    assert_valid(polygon)
    assert geometry.centroid(polygon) == pytest.approx((0.7, 0.5), abs=0.003)
    assert not geometry.point_in_polygon((0.3, 0.5), polygon)


def test_complex_outline_is_simplified_to_the_vertex_budget():
    star = np.zeros((H, W), dtype=np.uint8)
    points = []
    for k in range(60):  # a 30-pointed star has 60 corners
        r = 300 if k % 2 == 0 else 220
        a = 2 * math.pi * k / 60
        points.append([1024 + r * math.cos(a), 576 + r * math.sin(a)])
    cv2.fillPoly(star, [np.array(points, dtype=np.int32)], 255)
    polygon = mask_to_polygon(star)
    assert_valid(polygon)


@pytest.mark.parametrize("kind", ["bool", "rgb", "rgba"])
def test_masks_in_other_formats(kind):
    gray = disk_mask(0.4, 0.6, 80)
    if kind == "bool":
        mask = gray > 0
    elif kind == "rgb":
        mask = np.stack([gray] * 3, axis=2)
    else:  # white everywhere, the object marked by alpha only
        mask = np.dstack([np.full_like(gray, 255)] * 3 + [gray])
    polygon = mask_to_polygon(mask, dilate_fraction=0)
    assert geometry.centroid(polygon) == pytest.approx((0.4, 0.6), abs=0.003)


def test_empty_mask_gives_no_polygon():
    assert mask_to_polygon(np.zeros((H, W), dtype=np.uint8)) == []


# --- clean_polygon --------------------------------------------------------------------------


def test_clean_polygon_fixes_a_crossing_outline():
    bow_tie = [(0.4, 0.4), (0.6, 0.6), (0.6, 0.4), (0.4, 0.6)]
    assert geometry.simplicity_problem(bow_tie) is not None
    assert_valid(clean_polygon(bow_tie))


def test_clean_polygon_reduces_a_dense_outline_and_grows_it():
    dense = [
        (0.5 + 0.08 * math.cos(2 * math.pi * k / 300), 0.5 + 0.12 * math.sin(2 * math.pi * k / 300))
        for k in range(300)
    ]
    cleaned = clean_polygon(dense, (W, H))
    assert_valid(cleaned)
    assert geometry.area(cleaned) > geometry.area(dense)
    # grown by roughly 10 px all round, not more
    assert geometry.area(cleaned) < geometry.area(dense) * 1.15


def test_clean_polygon_rasterises_at_the_given_size():
    square = box_polygon(0.25, 0.25, 0.75, 0.75)
    cleaned = clean_polygon(square, (400, 300), dilate_fraction=0)
    assert geometry.area(cleaned) == pytest.approx(0.25, rel=0.02)


# --- Gemini coordinates ---------------------------------------------------------------------

BOX = [200, 100, 600, 500]  # [ymin, xmin, ymax, xmax]: x 0.1-0.5, y 0.2-0.6


def reading_of(box_2d, mask) -> str:
    return choose_mask_reading(score_mask_readings(box_2d, mask)).reading


def test_box_relative_mask_maps_into_the_box():
    corners = [[0, 0], [1000, 0], [1000, 1000], [0, 1000]]
    polygon = gemini_mask_to_polygon(BOX, corners, "xy-box")
    assert same_points(polygon, box_polygon(0.1, 0.2, 0.5, 0.6))
    middle = gemini_mask_to_polygon(BOX, [[500, 500]], "xy-box")
    assert same_points(middle, [(0.3, 0.4)])


def test_image_relative_mask_is_just_scaled():
    polygon = gemini_mask_to_polygon(BOX, [[100, 200], [500, 200], [500, 600]], "xy-image")
    assert same_points(polygon, [(0.1, 0.2), (0.5, 0.2), (0.5, 0.6)])
    swapped = gemini_mask_to_polygon(BOX, [[200, 100], [200, 500], [600, 500]], "yx-image")
    assert same_points(swapped, [(0.1, 0.2), (0.5, 0.2), (0.5, 0.6)])


def test_a_slender_outline_in_a_wide_box_stays_in_image_coordinates():
    # The points span well under half of their box, which is normal for a thin object.
    box = [200, 200, 800, 800]
    mask = [[400, 300], [600, 300], [600, 700], [400, 700]]
    assert reading_of(box, mask) == "xy-image"
    assert same_points(
        gemini_mask_to_polygon(box, mask), [(0.4, 0.3), (0.6, 0.3), (0.6, 0.7), (0.4, 0.7)]
    )
    lighthouse = [[480, 150], [520, 150], [530, 900], [470, 900]]
    assert reading_of([100, 100, 950, 900], lighthouse) == "xy-image"


def test_mask_frame_is_detected_from_where_the_points_fall():
    image_points = [[110, 210], [490, 220], [480, 590], [120, 580]]
    box_points = [[20, 30], [980, 10], [960, 990], [10, 970]]
    assert reading_of(BOX, image_points) == "xy-image"
    assert reading_of(BOX, [[y, x] for x, y in image_points]) == "yx-image"
    assert reading_of(BOX, box_points) == "xy-box"
    assert same_points(
        gemini_mask_to_polygon(BOX, box_points), gemini_mask_to_polygon(BOX, box_points, "xy-box")
    )


def test_box_conversion_round_trip_and_swapped_corners():
    assert box_to_1000((0.1, 0.2, 0.5, 0.6)) == BOX
    entry = parse_gemini_masks(json.dumps([{"box_2d": [600, 500, 200, 100], "label": "x"}]))[0]
    candidate = gemini_candidate(entry)
    assert candidate.bbox == pytest.approx((0.1, 0.2, 0.5, 0.6))
    assert geometry.area(candidate.polygon) == pytest.approx(0.16)  # no mask: the box itself


L_SHAPE = [[100, 100], [500, 100], [500, 300], [300, 300], [300, 500], [100, 500]]  # 12%


@pytest.mark.parametrize(
    "mask",
    [
        L_SHAPE[:3] + [[500, 300]] + L_SHAPE[3:],  # a vertex repeated
        L_SHAPE + [[100, 100]],  # explicitly closed
    ],
    ids=["repeated vertex", "closing vertex"],
)
def test_repeated_vertices_do_not_trigger_the_hull(mask):
    entry = parse_gemini_masks(json.dumps([{"box_2d": [100, 100, 500, 500], "mask": mask}]))[0]
    candidate = gemini_candidate(entry)
    assert len(candidate.polygon) == 6
    assert geometry.area(candidate.polygon) == pytest.approx(0.12)
    assert not geometry.point_in_polygon((0.4, 0.4), candidate.polygon)  # the notch stays open


def test_crossing_gemini_outline_falls_back_to_its_hull():
    # The real smoke-test response: the closing edge runs back over another edge.
    mask = [[571, 409], [582, 412], [603, 417], [641, 418], [653, 423], [664, 436], [664, 467]]
    mask += [[660, 469], [576, 469], [571, 731], [664, 731], [664, 746], [571, 746]]
    entry = parse_gemini_masks(
        json.dumps({"boxes": [{"box_2d": [409, 569, 746, 665], "mask": mask}]})
    )
    candidate = gemini_candidate(entry[0])
    assert geometry.simplicity_problem(candidate.polygon) is None
    assert geometry.point_in_polygon((0.62, 0.6), candidate.polygon)  # the body is inside


def test_parse_gemini_masks_accepts_the_shapes_seen_in_practice():
    entry = {"box_2d": [1, 2, 3, 4], "mask": [[2, 1], [4, 1], [4, 3]], "label": "hoop"}
    for text in (
        json.dumps({"boxes": [entry]}),
        json.dumps([entry]),
        "```json\n" + json.dumps([entry]) + "\n```",
        json.dumps([entry, {"box_2d": [1, 2]}, "junk"]),  # malformed entries are skipped
    ):
        parsed = parse_gemini_masks(text)
        assert len(parsed) == 1 and parsed[0].label == "hoop"


def test_convex_hull():
    hull = convex_hull([(0, 0), (1, 0), (0.5, 0.2), (1, 1), (0, 1), (0.5, 0.5)])
    assert sorted(hull) == [(0, 0), (0, 1), (1, 0), (1, 1)]


# --- choosing among candidates ---------------------------------------------------------------


def cand(x0, y0, x1, y1, label="obj", score=None):
    return Candidate(box_polygon(x0, y0, x1, y1), (x0, y0, x1, y1), label, score)


def test_prefers_the_largest_candidate_inside_the_safe_area():
    outside_big = cand(0.0, 0.3, 0.18, 0.7, "edge")  # centroid x 0.09: outside
    inside_small = cand(0.4, 0.4, 0.5, 0.5, "small")
    inside_large = cand(0.55, 0.3, 0.75, 0.6, "large")
    assert choose_candidate([outside_big, inside_small, inside_large]).label == "large"
    assert choose_candidate([outside_big, inside_small]).label == "small"


def test_falls_back_to_the_largest_when_nothing_is_in_the_safe_area():
    left = cand(0.0, 0.3, 0.1, 0.5, "left")
    right = cand(0.85, 0.3, 1.0, 0.6, "right")
    assert choose_candidate([left, right]).label == "right"
    assert choose_candidate([]) is None
    assert (
        choose_candidate([Candidate([(0.1, 0.1), (0.2, 0.2)], (0.1, 0.1, 0.2, 0.2), "x")]) is None
    )


def test_box_helpers():
    assert expand_box((0.4, 0.4, 0.6, 0.6), 0.5) == pytest.approx((0.3, 0.3, 0.7, 0.7))
    assert expand_box((0.0, 0.9, 0.2, 1.0), 0.5) == pytest.approx((0.0, 0.85, 0.3, 1.0))
    assert check_box([0.1, 0.2, 0.3, 0.4]) == (0.1, 0.2, 0.3, 0.4)
    for bad in ([0.3, 0.2, 0.1, 0.4], [0.1, 0.2, 1.3, 0.4], [0.1, 0.2, 0.3]):
        with pytest.raises(ValueError):
            check_box(bad)


# --- the Gemini provider, with a fake client --------------------------------------------------


class FakeInteractions:
    def __init__(self, outputs, errors=()):
        self.outputs = list(outputs)
        self.errors = list(errors)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        usage = SimpleNamespace(
            total_input_tokens=1200, total_output_tokens=300, total_thought_tokens=0
        )
        return SimpleNamespace(
            output_text=self.outputs.pop(0), usage=usage, id=f"interaction-{len(self.calls)}"
        )


@pytest.fixture
def scene_image(tmp_path):
    path = tmp_path / "scene.png"
    Image.new("RGB", (W, H), (40, 50, 70)).save(path)
    return path


def test_gemini_request_follows_the_documented_shape(scene_image):
    response = {
        "boxes": [{"box_2d": BOX, "mask": [[110, 210], [490, 210], [490, 590]], "label": "hoop"}]
    }
    interactions = FakeInteractions([json.dumps(response)])
    segmenter = GeminiSegmenter(client=SimpleNamespace(interactions=interactions))
    candidates = segmenter.segment(
        scene_image, "the basketball hoop", hint_box=(0.1, 0.2, 0.5, 0.6)
    )
    (call,) = interactions.calls
    assert call["model"] == "gemini-3.8-flash"
    assert call["generation_config"] == {"thinking_level": "low"}
    assert call["response_format"]["mime_type"] == "application/json"
    assert set(call["response_format"]["schema"]["$defs"]["GeminiMask"]["properties"]) == {
        "box_2d",
        "mask",
        "label",
    }
    text, image = call["input"]
    assert text["text"].startswith("Give the segmentation masks for the basketball hoop.")
    assert "[200, 100, 600, 500]" in text["text"]  # the hint box, in Gemini's own convention
    assert image["mime_type"] == "image/jpeg"
    assert Image.open(io.BytesIO(base64.b64decode(image["data"]))).size == (W, H)
    assert len(candidates) == 1
    assert same_points(candidates[0].polygon, [(0.11, 0.21), (0.49, 0.21), (0.49, 0.59)])
    assert segmenter.last_cost_usd == pytest.approx((1200 * 0.75 + 300 * 3.75) / 1e6)


def test_gemini_steps_up_the_thinking_level_when_one_is_refused(scene_image):
    refused = RuntimeError("Thinking level THINKING_LEVEL_LOW is not supported for this model.")
    refused.status_code = 400
    interactions = FakeInteractions([json.dumps({"boxes": []})], errors=[refused])
    segmenter = GeminiSegmenter(client=SimpleNamespace(interactions=interactions))
    assert segmenter.segment(scene_image, "door") == []
    levels = [c["generation_config"]["thinking_level"] for c in interactions.calls]
    assert levels == ["low", "medium"]


def test_gemini_does_not_retry_other_errors(scene_image):
    other = RuntimeError("quota exhausted")
    other.status_code = 429
    interactions = FakeInteractions([], errors=[other])
    segmenter = GeminiSegmenter(client=SimpleNamespace(interactions=interactions))
    with pytest.raises(RuntimeError, match="quota"):
        segmenter.segment(scene_image, "door")
    assert len(interactions.calls) == 1


# --- the SAM 3 provider, with a fake client ---------------------------------------------------


def data_url(mask: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(mask).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_sam3_masks_become_candidates(scene_image):
    small, large = disk_mask(0.3, 0.5, 50), disk_mask(0.6, 0.5, 120)
    calls = []

    class FakeFal:
        def subscribe(self, app, arguments):
            calls.append((app, arguments))
            return {
                "masks": [{"url": data_url(small)}, {"url": data_url(large)}],
                "scores": [0.4, 0.93],
                "boxes": [[0.3, 0.5, 0.05, 0.09], [0.6, 0.5, 0.12, 0.21]],
            }

    segmenter = Sam3Segmenter(client=FakeFal())
    candidates = segmenter.segment(scene_image, "a lantern", hint_box=(0.5, 0.4, 0.7, 0.6))
    ((app, arguments),) = calls
    assert app == "fal-ai/sam-3/image"
    assert arguments["prompt"] == "lantern"
    assert arguments["image_url"].startswith("data:image/jpeg;base64,")
    assert arguments["box_prompts"] == [{"x_min": 1024, "y_min": 461, "x_max": 1434, "y_max": 691}]
    best = choose_candidate(candidates)
    assert best.score == pytest.approx(0.93)
    assert best.centroid == pytest.approx((0.6, 0.5), abs=0.003)
    assert best.bbox == pytest.approx((0.54, 0.395, 0.66, 0.605))


# --- verification with Claude, with a fake client ----------------------------------------------


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def claude_response(payload, stop_reason="end_turn", **extra):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=3000, output_tokens=500),
        model="claude-opus-5-5",
        **extra,
    )


def fake_claude(*responses):
    messages = FakeMessages(responses)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def test_verification_request_uses_the_documented_defaults(scene_image):
    client, messages = fake_claude(
        claude_response({"matches": True, "reason": "The hoop.", "suggested_box": None})
    )
    polygon = box_polygon(0.4, 0.4, 0.5, 0.55)
    verdict = hotspots.verify_with_claude(
        scene_image, polygon, "basketball hoop", context="right of centre", client=client
    )
    assert verdict == Verdict(matches=True, reason="The hoop.", suggested_box=None)
    (call,) = messages.calls
    assert call["model"] == "claude-opus-5-5"
    assert "thinking" not in call  # adaptive thinking: the parameter is left out
    assert call["output_config"]["effort"] == "high"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["output_config"]["format"]["schema"]["required"] == [
        "matches",
        "reason",
        "suggested_box",
    ]
    assert call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    content = call["messages"][0]["content"]
    images = [block for block in content if block["type"] == "image"]
    assert len(images) == 2
    overlay = Image.open(io.BytesIO(base64.b64decode(images[0]["source"]["data"])))
    assert max(overlay.size) == 1568
    question = content[-1]["text"]
    assert "exactly the basketball hoop and nothing else" in question
    assert "right of centre" in question


def test_verdict_parsing():
    no = claude_response(
        {
            "matches": False,
            "reason": "A fence.",
            "suggested_box": {"x0": 0.6, "y0": 0.5, "x1": 0.4, "y1": 0.3},
        }
    )
    assert parse_verdict(no).suggested_box == (0.4, 0.3, 0.6, 0.5)  # corners put in order
    degenerate = claude_response(
        {
            "matches": False,
            "reason": "?",
            "suggested_box": {"x0": 0.5, "y0": 0.5, "x1": 0.5, "y1": 0.7},
        }
    )
    assert parse_verdict(degenerate).suggested_box is None
    refusal = claude_response(
        "", stop_reason="refusal", stop_details=SimpleNamespace(category="cyber")
    )
    verdict = parse_verdict(refusal)
    assert not verdict.matches and "cyber" in verdict.reason
    with pytest.raises(VerificationError):
        parse_verdict(claude_response("not json"))
    with pytest.raises(VerificationError):
        parse_verdict(claude_response('{"matches": tr', stop_reason="max_tokens"))


def test_overlays_keep_the_image_size_and_tint_the_region(scene_image):
    image = Image.open(scene_image)
    polygon = box_polygon(0.4, 0.4, 0.6, 0.6)
    overlay = render_overlay(image, polygon)
    assert overlay.size == (W, H)
    inside = np.asarray(overlay)[576, 1024]
    outside = np.asarray(overlay)[100, 100]
    assert inside[0] > outside[0] + 80 and inside[2] > outside[2] + 60  # magenta tint
    closeup = render_closeup(image, polygon)
    assert max(closeup.size) == 1024


# --- extraction with one retry -----------------------------------------------------------------

GOOD = cand(0.45, 0.4, 0.6, 0.6, "hoop")


def test_first_attempt_verified(scene_image):
    segmenter = FakeSegmenter([[GOOD]])
    verifier = FakeVerifier([Verdict(matches=True, reason="The hoop.")])
    result = extract_hotspot(scene_image, "hoop", segmenter, verifier, image_size=(W, H))
    assert result.verified and len(result.attempts) == 1
    updates = bridge_updates(result, "gemini")
    assert updates["verified"] is True and updates["needs_manual_hotspot"] is False
    assert updates["verification"] == "The hoop."
    assert updates["score"] is None  # Gemini reports no confidence: the field is removed
    assert updates["area"] == pytest.approx(geometry.area(updates["polygon"]), abs=1e-6)
    assert updates["centroid"] == pytest.approx(geometry.centroid(updates["polygon"]), abs=1e-6)
    assert updates["bbox"] == pytest.approx(geometry.bbox(updates["polygon"]), abs=1e-6)


def test_retry_uses_the_box_claude_suggests(scene_image):
    segmenter = FakeSegmenter([[cand(0.2, 0.2, 0.3, 0.3, "fence")], [GOOD]])
    verifier = FakeVerifier(
        [
            Verdict(matches=False, reason="A fence.", suggested_box=(0.45, 0.4, 0.6, 0.6)),
            Verdict(matches=True, reason="The hoop."),
        ]
    )
    result = extract_hotspot(scene_image, "hoop", segmenter, verifier, image_size=(W, H))
    assert result.verified and len(result.attempts) == 2
    assert segmenter.calls[0][2] is None
    assert segmenter.calls[1][2] == pytest.approx(expand_box((0.45, 0.4, 0.6, 0.6), 0.15))
    assert geometry.point_in_polygon((0.5, 0.5), result.polygon)


FLAG_ONLY = {"verified", "verification", "needs_manual_hotspot"}


def test_a_no_without_a_new_box_goes_straight_to_manual_work(scene_image):
    # Without a suggestion the retry would send the very same request, so there is none.
    segmenter = FakeSegmenter([[cand(0.2, 0.2, 0.3, 0.3)]])
    verifier = FakeVerifier([Verdict(matches=False, reason="Wrong object.")])
    result = extract_hotspot(
        scene_image, "hoop", segmenter, verifier, image_size=(W, H), hint_box=(0.4, 0.4, 0.6, 0.6)
    )
    assert not result.verified and len(result.attempts) == 1
    assert [call[2] for call in segmenter.calls] == [(0.4, 0.4, 0.6, 0.6)]
    updates = bridge_updates(result, "gemini")
    assert set(updates) == FLAG_ONLY  # a rejected outline never replaces the hotspot
    assert updates["needs_manual_hotspot"] is True and updates["verified"] is False
    assert updates["verification"].startswith("Not verified after 1 attempt. Attempt 1:")


def test_a_suggestion_equal_to_the_current_hint_is_not_retried(scene_image):
    suggestion = (0.45, 0.4, 0.6, 0.6)
    hint = expand_box(suggestion, 0.15)  # what a retry with this suggestion would send
    segmenter = FakeSegmenter([[cand(0.2, 0.2, 0.3, 0.3)]])
    verifier = FakeVerifier([Verdict(matches=False, reason="No.", suggested_box=suggestion)])
    result = extract_hotspot(
        scene_image, "hoop", segmenter, verifier, image_size=(W, H), hint_box=hint
    )
    assert len(result.attempts) == 1 and len(segmenter.calls) == 1


def test_two_failures_flag_the_scene_without_touching_its_hotspot(scene_image):
    court = cand(0.2, 0.3, 0.8, 0.82, "basketball court")  # 31% of the frame
    segmenter = FakeSegmenter([[court]])
    verifier = FakeVerifier(
        [Verdict(matches=False, reason="The whole court.", suggested_box=(0.5, 0.3, 0.6, 0.5))]
    )
    result = extract_hotspot(scene_image, "hoop", segmenter, verifier, image_size=(W, H))
    assert not result.verified and len(result.attempts) == 2
    assert result.polygon is None and result.final is None
    assert all(a.polygon for a in result.attempts)  # still kept for the review overlays
    updates = bridge_updates(result, "gemini")
    assert set(updates) == FLAG_ONLY
    assert updates["verification"].startswith("Not verified after 2 attempts.")


def test_nothing_found_keeps_the_existing_polygon(scene_image):
    segmenter = FakeSegmenter([[]])
    verifier = FakeVerifier([Verdict(matches=True, reason="unused")])
    result = extract_hotspot(scene_image, "hoop", segmenter, verifier, image_size=(W, H))
    assert verifier.calls == []
    assert bridge_updates(result, "gemini") == {
        "verified": False,
        "verification": result.verification_text(),
        "needs_manual_hotspot": True,
    }


def test_spend_guard_stops_before_a_call_that_could_exceed_the_cap(scene_image):
    segmenter = FakeSegmenter([[GOOD]])
    segmenter.call_estimate_usd = 0.05
    verifier = FakeVerifier([Verdict(matches=True, reason="ok")])
    with pytest.raises(SpendLimitExceeded):
        extract_hotspot(
            scene_image, "hoop", segmenter, verifier, image_size=(W, H), spend=Spend(cap_usd=0.01)
        )
    assert segmenter.calls == []
    spend = Spend(cap_usd=1.0)
    spend.add(0.98)
    with pytest.raises(SpendLimitExceeded, match="DSP_MAX_SPEND_USD"):
        spend.reserve(0.05, "verification")


def test_updates_are_plain_yaml_data(scene_image):
    segmenter = FakeSegmenter([[cand(0.45, 0.4, 0.6, 0.6, score=np.float64(0.87))]])
    verifier = FakeVerifier([Verdict(matches=True, reason="ok")])
    result = extract_hotspot(scene_image, "hoop", segmenter, verifier, image_size=(W, H))
    updates = bridge_updates(result, "sam3")
    assert updates["score"] == 0.87
    yaml.safe_dump(updates)  # numpy scalars would make this raise


def test_rasterize_matches_the_polygon_area():
    polygon = box_polygon(0.25, 0.25, 0.5, 0.75)
    assert (rasterize(polygon, W, H) > 0).mean() == pytest.approx(0.125, rel=0.01)


# --- one generation-log row per paid call --------------------------------------------------------

HOOP_RESPONSE = json.dumps(
    {"boxes": [{"box_2d": BOX, "mask": [[110, 210], [490, 210], [490, 590]], "label": "hoop"}]}
)
YES = {"matches": True, "reason": "The hoop.", "suggested_box": None}


def call_log(tmp_path, scene_image) -> CallLog:
    return CallLog(
        tmp_path / "logs" / "generations.jsonl",
        "prison-yard",
        {"path": "assets/masters/prison-yard.graded.png", "sha256": "ab" * 32},
    )


def test_each_call_gets_a_generation_log_row(tmp_path, scene_image):
    interactions = FakeInteractions([HOOP_RESPONSE])
    segmenter = GeminiSegmenter(client=SimpleNamespace(interactions=interactions))
    client, _ = fake_claude(claude_response(YES, id="msg_1", _request_id="req_1"))
    log = call_log(tmp_path, scene_image)
    spend = Spend(cap_usd=1.0)
    result = extract_hotspot(
        scene_image,
        "the basketball hoop",
        segmenter,
        ClaudeVerifier(client),
        image_size=(W, H),
        spend=spend,
        log=log,
    )
    segment, verify = genlog.read_rows(log.path)
    common = {"purpose": "hotspot", "scene": "prison-yard", "attempt": 1, "status": "ok"}
    assert (
        segment.items()
        >= (
            common
            | {
                "kind": "segment",
                "provider": "gemini",
                "model": "gemini-3.8-flash",
                "concept": "basketball hoop",
                "hint_box": None,
                "image": log.image,
                "error": None,
                "response_id": "interaction-1",
                "usage": {
                    "total_input_tokens": 1200,
                    "total_output_tokens": 300,
                    "total_thought_tokens": 0,
                },
                "cost_estimate_usd": 0.01,
            }
        ).items()
    )
    assert segment["cost_usd"] == pytest.approx((1200 * 0.75 + 300 * 3.75) / 1e6)
    assert (
        verify.items()
        >= (
            common
            | {
                "kind": "verify",
                "provider": "anthropic",
                "model": "claude-opus-5-5",
                "served_model": "claude-opus-5-5",
                "request_id": "req_1",
                "response_id": "msg_1",
                "usage": {"input_tokens": 3000, "output_tokens": 500},
                "matches": True,
                "cost_estimate_usd": 0.09,
            }
        ).items()
    )
    assert verify["cost_usd"] == pytest.approx((3000 * 4 + 500 * 20) / 1e6)
    for row in (segment, verify):
        assert {"id", "timestamp", "duration_s"} <= row.keys()
    assert genlog.total_spend([segment, verify]) == pytest.approx(spend.spent_usd)
    (attempt,) = result.attempts
    assert (attempt.segment_log_id, attempt.verify_log_id) == (segment["id"], verify["id"])


def test_a_retry_logs_four_calls(tmp_path, scene_image):
    segmenter = FakeSegmenter([[cand(0.2, 0.2, 0.3, 0.3, "fence")], [GOOD]], cost_usd=0.002)
    verifier = FakeVerifier(
        [
            Verdict(matches=False, reason="A fence.", suggested_box=(0.45, 0.4, 0.6, 0.6)),
            Verdict(matches=True, reason="The hoop."),
        ],
        cost_usd=0.02,
    )
    log = call_log(tmp_path, scene_image)
    extract_hotspot(scene_image, "hoop", segmenter, verifier, image_size=(W, H), log=log)
    rows = genlog.read_rows(log.path)
    assert [(r["kind"], r["attempt"]) for r in rows] == [
        ("segment", 1),
        ("verify", 1),
        ("segment", 2),
        ("verify", 2),
    ]
    assert [r["matches"] for r in rows if r["kind"] == "verify"] == [False, True]
    assert rows[2]["hint_box"] == pytest.approx(list(expand_box((0.45, 0.4, 0.6, 0.6), 0.15)))
    assert genlog.total_spend(rows) == pytest.approx(0.044)


def test_an_unreadable_gemini_answer_is_logged_and_charged(tmp_path, scene_image):
    interactions = FakeInteractions(["Here is the mask you asked for."])
    segmenter = GeminiSegmenter(client=SimpleNamespace(interactions=interactions))
    log = call_log(tmp_path, scene_image)
    spend = Spend(cap_usd=1.0)
    verifier = FakeVerifier([Verdict(matches=True, reason="unused")])
    with pytest.raises(json.JSONDecodeError):
        extract_hotspot(
            scene_image, "hoop", segmenter, verifier, image_size=(W, H), spend=spend, log=log
        )
    (row,) = genlog.read_rows(log.path)
    assert row["status"] == "error" and row["error"].startswith("JSONDecodeError")
    assert row["cost_usd"] == pytest.approx(spend.spent_usd) and spend.spent_usd > 0


def test_a_call_that_fails_before_answering_has_no_cost(tmp_path, scene_image):
    down = RuntimeError("service unavailable")
    down.status_code = 503
    # The earlier successful call must not lend its cost or ids to the failed one.
    interactions = FakeInteractions([HOOP_RESPONSE], errors=[])
    segmenter = GeminiSegmenter(client=SimpleNamespace(interactions=interactions))
    segmenter.segment(scene_image, "hoop")
    interactions.errors.append(down)
    log = call_log(tmp_path, scene_image)
    verifier = FakeVerifier([Verdict(matches=True, reason="unused")])
    with pytest.raises(RuntimeError, match="unavailable"):
        extract_hotspot(scene_image, "hoop", segmenter, verifier, image_size=(W, H), log=log)
    (row,) = genlog.read_rows(log.path)
    assert row["status"] == "error"
    assert row["cost_usd"] is None and row["response_id"] is None and row["usage"] is None


def test_an_unreadable_verdict_is_logged_as_an_error_with_its_cost(tmp_path, scene_image):
    client, _ = fake_claude(claude_response("not json"))
    log = call_log(tmp_path, scene_image)
    result = extract_hotspot(
        scene_image,
        "hoop",
        FakeSegmenter([[GOOD]]),
        ClaudeVerifier(client),
        image_size=(W, H),
        log=log,
    )
    assert not result.verified
    segment, verify = genlog.read_rows(log.path)
    assert verify["status"] == "error" and verify["error"].startswith("VerificationError")
    assert verify["matches"] is None
    assert verify["cost_usd"] == pytest.approx((3000 * 4 + 500 * 20) / 1e6)

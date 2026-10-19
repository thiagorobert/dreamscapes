from __future__ import annotations

import pytest
import yaml

from dreamscapes_pipeline.hotspots import BRIDGE_FIELD_ORDER
from dreamscapes_pipeline.yaml_patch import PatchError, render_fields, set_scene_fields

HAND_WRITTEN = """\
# The loop. Edited by hand; keep these notes.
version: 1
start: first
style_anchor: first
scenes:
  - id: first
    title: First   # shown in the tab
    alt: Alt one.
    scene: >-
      A long description that is folded over two lines so the block scalar ends on a line of
      its own.
    bridge:
      object: lantern          # noun phrase; also the segmentation prompt
      placement: small, centre
      polygon: []              # filled by dsp hotspot
      verified: false

      # an empty line and a comment between fields
      needs_manual_hotspot: true
    next: second
  - id: second
    title: Second
    alt: Alt two.
    scene: A pond.
    bridge:
      object: paper boat
      placement: small, left
      polygon:
        - [0.4, 0.4]
        - [0.6, 0.4]
        - [0.6, 0.6]
      source: manual
      verified: true
    next: first
"""

UPDATES = {
    "polygon": [[0.45, 0.45], [0.55, 0.45], [0.5, 0.55]],
    "source": "gemini",
    "verified": True,
    "verification": "The region is the lantern hanging from the hook, with a small margin around "
    "it and nothing else.",
    "score": None,
    "needs_manual_hotspot": False,
    "bbox": [0.45, 0.45, 0.55, 0.55],
    "centroid": [0.5, 0.483333],
    "area": 0.005,
}


def patch(text=HAND_WRITTEN, scene="first", updates=UPDATES):
    return set_scene_fields(text, scene, "bridge", updates, BRIDGE_FIELD_ORDER)


def test_only_the_owned_lines_change():
    result = patch()
    assert result.in_place
    before, after = HAND_WRITTEN.splitlines(), result.text.splitlines()
    # Everything outside the first scene's hotspot fields is byte-for-byte the same.
    first_polygon = before.index("      polygon: []              # filled by dsp hotspot")
    assert after[:first_polygon] == before[:first_polygon]
    tail = before.index("    next: second")
    assert after[after.index("    next: second") :] == before[tail:]
    for comment in (
        "# The loop. Edited by hand; keep these notes.",
        "    title: First   # shown in the tab",
        "      object: lantern          # noun phrase; also the segmentation prompt",
        "      # an empty line and a comment between fields",
    ):
        assert comment in after


def test_data_and_key_order_after_the_patch():
    data = yaml.safe_load(patch().text)
    bridge = data["scenes"][0]["bridge"]
    for key, value in UPDATES.items():
        if value is None:
            assert key not in bridge
        else:
            assert bridge[key] == value
    # Existing keys keep their places; new ones land where the model's field order puts them.
    assert list(bridge) == [
        "object",
        "placement",
        "polygon",
        "source",
        "verified",
        "verification",
        "needs_manual_hotspot",
        "bbox",
        "centroid",
        "area",
    ]
    assert data["scenes"][1] == yaml.safe_load(HAND_WRITTEN)["scenes"][1]
    assert list(data) == ["version", "start", "style_anchor", "scenes"]
    assert list(data["scenes"][0]) == ["id", "title", "alt", "scene", "bridge", "next"]


def test_patching_twice_is_stable():
    once = patch().text
    assert patch(once).text == once


def test_none_removes_a_field_and_unknown_order_keys_go_last():
    result = set_scene_fields(
        HAND_WRITTEN, "second", "bridge", {"source": None, "verified": False}, BRIDGE_FIELD_ORDER
    )
    bridge = yaml.safe_load(result.text)["scenes"][1]["bridge"]
    assert "source" not in bridge and bridge["verified"] is False
    assert result.text.count("source: manual") == 0


def test_null_values_elsewhere_are_left_alone():
    text = HAND_WRITTEN.replace(
        "      verified: true\n", "      verified: true\n      score: null\n"
    )
    result = set_scene_fields(text, "second", "bridge", {"verified": False}, BRIDGE_FIELD_ORDER)
    assert result.in_place
    assert "score: null" in result.text


def test_flow_style_section_falls_back_to_a_full_rewrite():
    text = """\
version: 1
start: a
style_anchor: a
scenes:
  - id: a
    title: A
    alt: Alt.
    scene: Somewhere.
    bridge: {object: door, placement: centre}  # comments cannot survive this
    next: a
"""
    result = set_scene_fields(text, "a", "bridge", {"verified": True}, BRIDGE_FIELD_ORDER)
    assert not result.in_place
    data = yaml.safe_load(result.text)
    assert data["scenes"][0]["bridge"] == {
        "object": "door",
        "placement": "centre",
        "verified": True,
    }
    assert list(data["scenes"][0]) == ["id", "title", "alt", "scene", "bridge", "next"]


def test_unknown_scene_is_an_error():
    with pytest.raises(PatchError, match="not found"):
        patch(scene="missing")


def test_render_fields_matches_the_written_lines():
    snippet = render_fields(UPDATES, 6)
    assert "score" not in snippet
    # The second scene has no comments between its fields, so the lines appear as one block.
    assert snippet in patch(scene="second").text
    assert snippet.splitlines()[0] == "      polygon:"
    assert snippet.splitlines()[1] == "        - [0.45, 0.45]"

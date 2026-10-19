from __future__ import annotations

import json

import pytest
import yaml
from PIL import Image

from dreamscapes_pipeline.manifest import Transition, load_manifest, save_manifest
from dreamscapes_pipeline.validate import check_manifest, mentions, validate_repo
from tests.conftest import good_manifest, square


def messages(result, level):
    return [str(i) for i in result.issues if i.level == level]


def assert_single_error(result, fragment):
    errors = messages(result, "error")
    assert len(errors) == 1 and fragment in errors[0], errors


def test_good_manifest_has_no_issues():
    assert check_manifest(good_manifest()).issues == []


def test_placeholder_repository_is_completely_clean(placeholder_repo):
    assert validate_repo(placeholder_repo).issues == []


# --- hotspot polygons ------------------------------------------------------------------------


def test_self_intersecting_polygon_is_an_error():
    m = good_manifest()
    m.scenes[0].bridge.polygon = [(0.4, 0.4), (0.6, 0.6), (0.6, 0.4), (0.4, 0.6)]
    assert_single_error(check_manifest(m), "edges 0 and 2 intersect")


def test_polygon_needs_three_vertices():
    m = good_manifest()
    m.scenes[0].bridge.polygon = [(0.4, 0.4), (0.6, 0.6)]
    assert_single_error(check_manifest(m), "at least 3 vertices")


@pytest.mark.parametrize(("half", "shown"), [(0.05, "1.00%"), (0.2, "16.00%")])
def test_polygon_area_must_be_between_limits(half, shown):
    m = good_manifest()
    m.scenes[1].bridge.polygon = square(0.5, 0.5, half)
    result = check_manifest(m)
    assert_single_error(result, f"hotspot covers {shown} of the frame")
    assert result.errors[0].scene == "second-scene"


@pytest.mark.parametrize(
    ("centre", "size"),
    [((0.15, 0.5), (0.08, 0.2)), ((0.85, 0.5), (0.08, 0.2)), ((0.5, 0.05), (0.2, 0.08))]
    + [((0.5, 0.95), (0.2, 0.08))],
)
def test_centroid_must_be_inside_the_safe_area(centre, size):
    (cx, cy), (w, h) = centre, size  # every rectangle covers 1.6%, inside the area limits
    m = good_manifest()
    m.scenes[0].bridge.polygon = [
        (cx - w / 2, cy - h / 2),
        (cx + w / 2, cy - h / 2),
        (cx + w / 2, cy + h / 2),
        (cx - w / 2, cy + h / 2),
    ]
    assert_single_error(check_manifest(m), "outside the safe area")


def test_missing_polygon_is_only_a_warning():
    m = good_manifest()
    m.scenes[2].bridge.polygon = []
    result = check_manifest(m)
    assert result.errors == []
    assert messages(result, "warning") == [
        "warning third-scene: bridge object 'candle' has no polygon yet (run dsp hotspot)"
    ]


def test_manual_hotspot_flag_is_a_warning():
    m = good_manifest()
    m.scenes[0].bridge.needs_manual_hotspot = True
    result = check_manifest(m)
    assert result.errors == [] and "manual annotation" in messages(result, "warning")[0]


def test_stale_stored_geometry_is_a_warning():
    m = good_manifest()
    m.scenes[0].bridge.area = 0.5
    m.scenes[0].bridge.centroid = (0.5, 0.5)  # correct, so not reported
    result = check_manifest(m)
    assert result.errors == []
    assert "stored area no longer match" in messages(result, "warning")[0]


# --- the loop --------------------------------------------------------------------------------


def test_next_must_exist():
    m = good_manifest()
    m.scenes[1].next = "nowhere"
    assert_single_error(check_manifest(m), "next scene 'nowhere' does not exist")


def test_two_separate_loops_are_an_error():
    m = good_manifest()
    m.scenes[0].next = "first-scene"
    m.scenes[2].next = "second-scene"
    assert_single_error(check_manifest(m), "never reaches second-scene, third-scene")


def test_a_branch_into_the_loop_is_an_error():
    m = good_manifest()
    m.scenes[2].next = "second-scene"  # first -> second -> third -> second; first is orphaned
    assert_single_error(check_manifest(m), "must form one loop")


def test_duplicate_ids_are_an_error():
    m = good_manifest()
    m.scenes[2].id = "second-scene"
    errors = messages(check_manifest(m), "error")
    assert any("ids must be unique" in e for e in errors), errors


def test_start_and_style_anchor_must_exist():
    m = good_manifest()
    m.start = "missing-start"
    m.style_anchor = "missing-anchor"
    errors = messages(check_manifest(m), "error")
    assert any("start scene 'missing-start'" in e for e in errors)
    assert any("style_anchor scene 'missing-anchor'" in e for e in errors)


# --- descriptions ----------------------------------------------------------------------------


def test_description_should_mention_previous_bridge_object():
    m = good_manifest()
    m.scenes[1].scene = "A pond at night."
    result = check_manifest(m)
    assert result.errors == []
    assert messages(result, "warning") == [
        "warning second-scene: description does not mention 'lantern', the bridge object of "
        "the previous scene 'first-scene'"
    ]


@pytest.mark.parametrize(
    ("description", "obj", "expected"),
    [
        ("Balloons drift over the flats.", "hot-air balloon", True),
        ("The dark hull of a ship.", "the ship's hull", True),
        ("A red telephone box glows.", "telephone box", True),
        ("A red phone glows.", "telephone box", False),
        # An adjective alone is not enough; the head noun must appear.
        ("A distant light.", "the distant lighthouse", False),
    ],
)
def test_mentions(description, obj, expected):
    assert mentions(description, obj) is expected


# --- files -----------------------------------------------------------------------------------


def test_missing_master_is_an_error(small_repo):
    small_repo.master("second-scene").unlink()
    result = validate_repo(small_repo)
    assert_single_error(result, "master assets/masters/second-scene.png is missing")


def test_master_with_wrong_dimensions_is_an_error(small_repo):
    Image.new("RGB", (1024, 1024)).save(small_repo.master("first-scene"))
    assert_single_error(validate_repo(small_repo), "is 1024x1024, expected 2048x1152")


def test_mask_with_wrong_dimensions_is_an_error(small_repo):
    small_repo.masks.mkdir(parents=True)
    Image.new("L", (2048, 1024)).save(small_repo.mask("first-scene"))
    assert_single_error(validate_repo(small_repo), "mask assets/masks/first-scene.png")


def test_missing_built_manifest_is_a_warning(small_repo):
    result = validate_repo(small_repo)
    assert result.errors == []
    assert "scenes.json does not exist yet" in messages(result, "warning")[0]


def test_missing_transition_clip_is_an_error(small_repo):
    m = load_manifest(small_repo.scenes_yaml)
    m.scenes[0].transition.type = "video"
    m.scenes[0].transition.video = "morphs/first-scene__second-scene.mp4"
    m.scenes[0].transition.duration = 8
    save_manifest(m, small_repo.scenes_yaml)
    assert_single_error(validate_repo(small_repo), "first-scene__second-scene.mp4 is missing")


def test_missing_export_is_an_error(placeholder_repo):
    built = json.loads(placeholder_repo.scenes_json.read_text())
    src = built["scenes"][3]["image"]["sources"]["avif"][1]["src"]
    placeholder_repo.public_file(src).unlink()
    assert_single_error(validate_repo(placeholder_repo), f"export web/public/{src} is missing")


def test_export_with_wrong_dimensions_is_an_error(placeholder_repo):
    built = json.loads(placeholder_repo.scenes_json.read_text())
    src = built["scenes"][0]["image"]["sources"]["webp"][0]["src"]
    Image.new("RGB", (640, 640)).save(placeholder_repo.public_file(src), format="WEBP")
    assert_single_error(validate_repo(placeholder_repo), "is 640x640, expected 640x360")


def test_out_of_date_built_manifest_is_a_warning(placeholder_repo):
    m = load_manifest(placeholder_repo.scenes_yaml)
    m.scenes[4].title = "A new title"
    save_manifest(m, placeholder_repo.scenes_yaml)
    result = validate_repo(placeholder_repo)
    assert result.errors == []
    warning = messages(result, "warning")[0]
    assert "out of date" in warning and "endless-library (title)" in warning


def test_changed_focal_makes_the_built_manifest_out_of_date(placeholder_repo):
    m = load_manifest(placeholder_repo.scenes_yaml)
    m.scenes[2].focal = (0.5, 0.5)
    save_manifest(m, placeholder_repo.scenes_yaml)
    warnings = messages(validate_repo(placeholder_repo), "warning")
    assert len(warnings) == 1 and "rooftop-court (focal)" in warnings[0], warnings


def test_unset_focal_matches_the_centroid_it_resolves_to(placeholder_repo):
    m = load_manifest(placeholder_repo.scenes_yaml)
    assert all(scene.focal is None for scene in m.scenes)
    # With polygons present the default is their centroid; with polygons cleared (as the
    # content package leaves them) it is the built stand-in's centroid. Neither is drift.
    assert validate_repo(placeholder_repo).issues == []
    for scene in m.scenes:
        scene.bridge.polygon = []
    save_manifest(m, placeholder_repo.scenes_yaml)
    warnings = messages(validate_repo(placeholder_repo), "warning")
    assert warnings and all("has no polygon yet" in w for w in warnings), warnings


def _video_repo(placeholder_repo, clip: str, duration: float):
    """Give the first scene a video transition in both manifests, with the clip on disk."""
    m = load_manifest(placeholder_repo.scenes_yaml)
    m.scenes[0].transition = Transition(type="video", video=clip, duration=duration)
    save_manifest(m, placeholder_repo.scenes_yaml)
    path = placeholder_repo.public_file(clip)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not really a video")
    data = json.loads(placeholder_repo.scenes_json.read_text())
    data["scenes"][0]["transition"] = {"type": "video", "video": clip, "duration": duration}
    placeholder_repo.scenes_json.write_text(json.dumps(data))
    return m


def test_matching_video_transition_is_not_drift(placeholder_repo):
    _video_repo(placeholder_repo, "morphs/cliff-cottage__prison-yard.mp4", 8)
    assert validate_repo(placeholder_repo).issues == []


@pytest.mark.parametrize(
    ("clip", "duration"),
    [("morphs/cliff-cottage__prison-yard.v2.mp4", 8), ("morphs/cliff-cottage__prison-yard.mp4", 6)],
)
def test_changed_clip_with_the_same_type_is_drift(placeholder_repo, clip, duration):
    m = _video_repo(placeholder_repo, "morphs/cliff-cottage__prison-yard.mp4", 8)
    m.scenes[0].transition = Transition(type="video", video=clip, duration=duration)
    save_manifest(m, placeholder_repo.scenes_yaml)
    placeholder_repo.public_file(clip).write_bytes(b"also not a video")
    warnings = messages(validate_repo(placeholder_repo), "warning")
    assert len(warnings) == 1 and "cliff-cottage (transition clip)" in warnings[0], warnings


def test_invalid_built_manifest_is_an_error(placeholder_repo):
    data = json.loads(placeholder_repo.scenes_json.read_text())
    del data["scenes"][0]["image"]["lqip"]
    placeholder_repo.scenes_json.write_text(json.dumps(data))
    assert_single_error(validate_repo(placeholder_repo), "image.lqip: Field required")


def test_bridge_list_in_yaml_is_reported(small_repo):
    data = yaml.safe_load(small_repo.scenes_yaml.read_text())
    bridge = data["scenes"][0]["bridge"]
    data["scenes"][0]["bridge"] = [bridge, dict(bridge, object="second object")]
    small_repo.scenes_yaml.write_text(yaml.safe_dump(data))
    errors = messages(validate_repo(small_repo), "error")
    assert errors and "exactly one bridge object" in errors[0], errors


def test_unreadable_yaml_is_an_error(small_repo):
    small_repo.scenes_yaml.write_text("version: [1\n")
    assert_single_error(validate_repo(small_repo), "invalid YAML")

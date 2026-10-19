from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dreamscapes_pipeline.manifest import (
    Manifest,
    ManifestError,
    Transition,
    build_scene,
    dump_built_manifest,
    dump_manifest,
    load_built_manifest,
    load_manifest,
    parse_manifest,
    save_manifest,
)
from dreamscapes_pipeline.placeholders import placeholder_manifest
from dreamscapes_pipeline.schema import render_schema
from tests.conftest import good_manifest, square

REPO_ROOT = Path(__file__).resolve().parents[2]


def raw(manifest: Manifest) -> dict:
    return manifest.model_dump(mode="json", exclude_none=True)


def test_authored_round_trip(tmp_path: Path):
    manifest = placeholder_manifest()
    manifest.scenes[0].scene = "A very long description " * 12 + "that ends here."
    manifest.scenes[1].alt = "Line one.\nLine two."
    manifest.scenes[2].focal = (0.25, 0.75)
    path = tmp_path / "scenes.yaml"
    save_manifest(manifest, path)
    loaded = load_manifest(path)
    assert loaded == manifest
    assert dump_manifest(loaded) == path.read_text()


def test_built_round_trip(placeholder_repo):
    text = placeholder_repo.scenes_json.read_text()
    built = load_built_manifest(placeholder_repo.scenes_json)
    assert dump_built_manifest(built) == text


def test_polygon_may_be_empty_in_the_authored_manifest():
    data = raw(good_manifest())
    data["scenes"][0]["bridge"]["polygon"] = []
    del data["scenes"][1]["bridge"]["polygon"]
    manifest = parse_manifest(data)
    assert manifest.scenes[0].bridge.polygon == []
    assert manifest.scenes[1].bridge.polygon == []


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda d: d["scenes"][0].update(id="First Scene"), "scenes[0] (First Scene).id"),
        (lambda d: d["scenes"][1].update(next="Not_A_Slug"), "scenes[1] (second-scene).next"),
        (
            lambda d: d["scenes"][0]["bridge"].update(polygon=[[0.1, 0.1], [1.2, 0.1], [0.5, 0.5]]),
            "bridge.polygon[1][0]",
        ),
        (lambda d: d["scenes"][0]["bridge"].update(polgon=[]), "polgon"),
        (lambda d: d["scenes"][0].update(bridge=[d["scenes"][0]["bridge"]] * 2), "exactly one"),
        (lambda d: d["scenes"][0].update(transition={"type": "video"}), "video"),
        (lambda d: d.update(scenes=[]), "scenes"),
    ],
)
def test_invalid_manifests_are_rejected_with_a_location(mutate, expected):
    data = raw(good_manifest())
    mutate(data)
    with pytest.raises(ManifestError) as caught:
        parse_manifest(data)
    assert any(expected in problem for problem in caught.value.problems), caught.value.problems


def test_load_reports_yaml_syntax_errors(tmp_path: Path):
    path = tmp_path / "scenes.yaml"
    path.write_text("scenes: [unclosed\n")
    with pytest.raises(ManifestError, match="invalid YAML"):
        load_manifest(path)


def test_saved_yaml_is_readable(tmp_path: Path):
    path = tmp_path / "scenes.yaml"
    save_manifest(good_manifest(), path)
    text = path.read_text()
    assert "- [0.4, 0.4]" in text  # points stay on one line
    assert yaml.safe_load(text)["start"] == "first-scene"


def test_build_scene_derives_geometry_and_focal(placeholder_repo):
    built = load_built_manifest(placeholder_repo.scenes_json)
    image = built.scenes[0].image
    scene = good_manifest().scenes[0]
    scene.bridge.polygon = square(0.4, 0.5, 0.1)
    out = build_scene(scene, image)
    assert out.bridge.bbox == pytest.approx((0.3, 0.4, 0.5, 0.6))
    assert out.bridge.centroid == pytest.approx((0.4, 0.5))
    assert out.bridge.area == pytest.approx(0.04)
    assert out.focal == out.bridge.centroid
    assert out.transition.type == "zoom"

    scene.focal = (0.6, 0.6)
    scene.transition = Transition(type="video", video="morphs/a__b.mp4", duration=8)
    out = build_scene(scene, image)
    assert out.focal == (0.6, 0.6)
    assert out.transition.type == "video"
    assert out.transition.video == "morphs/a__b.mp4"


def test_build_scene_refuses_a_missing_polygon(placeholder_repo):
    image = load_built_manifest(placeholder_repo.scenes_json).scenes[0].image
    scene = good_manifest().scenes[0]
    scene.bridge.polygon = []
    with pytest.raises(ValueError, match="no usable polygon"):
        build_scene(scene, image)


def test_committed_schema_matches_the_models():
    committed = (REPO_ROOT / "schema" / "scenes.schema.json").read_text()
    assert committed == render_schema(), "run `uv run dsp schema` and commit the result"


def test_committed_manifests_load():
    load_manifest(REPO_ROOT / "content" / "scenes.yaml")
    load_built_manifest(REPO_ROOT / "web" / "public" / "scenes.json")

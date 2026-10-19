from __future__ import annotations

import numpy as np
from PIL import Image

from dreamscapes_pipeline import geometry
from dreamscapes_pipeline.layout import EXPORT_WIDTHS, MASTER_HEIGHT, MASTER_WIDTH
from dreamscapes_pipeline.manifest import load_built_manifest, load_manifest, save_manifest
from dreamscapes_pipeline.placeholders import (
    PLACEHOLDER_SCENES,
    is_placeholder_image,
    placeholder_manifest,
    run_placeholders,
)
from dreamscapes_pipeline.validate import validate_repo
from tests.conftest import file_hashes

SLUGS = [
    "cliff-cottage",
    "prison-yard",
    "rooftop-court",
    "balloon-salt-flats",
    "endless-library",
    "overgrown-greenhouse",
    "mountain-road-phonebox",
    "drowned-street",
    "fog-ship-deck",
    "lighthouse-gallery",
]


def test_writes_every_artifact(placeholder_repo):
    paths = placeholder_repo
    manifest = load_manifest(paths.scenes_yaml)
    built = load_built_manifest(paths.scenes_json)
    assert [s.id for s in manifest.scenes] == SLUGS
    assert [s.id for s in built.scenes] == SLUGS
    assert built.start == manifest.start == "cliff-cottage"
    for scene in built.scenes:
        for path in (paths.master(scene.id), paths.graded(scene.id), paths.mask(scene.id)):
            with Image.open(path) as image:
                assert image.size == (MASTER_WIDTH, MASTER_HEIGHT)
        assert is_placeholder_image(paths.master(scene.id))
        for fmt in ("avif", "webp"):
            sources = getattr(scene.image.sources, fmt)
            assert [s.width for s in sources] == sorted(EXPORT_WIDTHS)
            assert all(paths.public_file(s.src).exists() for s in sources)
        assert scene.image.lqip.startswith("data:image/webp;base64,")


def test_mask_matches_the_hotspot(placeholder_repo):
    built = load_built_manifest(placeholder_repo.scenes_json)
    for scene in built.scenes:
        with Image.open(placeholder_repo.mask(scene.id)) as mask:
            coverage = float((np.asarray(mask) > 0).mean())
            cx, cy = scene.bridge.centroid
            assert mask.getpixel((int(cx * MASTER_WIDTH), int(cy * MASTER_HEIGHT))) == 255
        assert abs(coverage - scene.bridge.area) < 0.002, scene.id


def test_every_hotspot_contains_its_centroid():
    # The website's end-to-end tests click the centroid, so it must land on the shape.
    for scene in placeholder_manifest().scenes:
        polygon = scene.bridge.polygon
        assert geometry.point_in_polygon(geometry.centroid(polygon), polygon), scene.id


def test_shapes_are_distinct_and_spread_out():
    centres = {
        tuple(round(c, 2) for c in geometry.centroid(p.design.polygon()))
        for p in PLACEHOLDER_SCENES
    }
    assert len(centres) == len(PLACEHOLDER_SCENES)
    assert len({len(p.design.outline) for p in PLACEHOLDER_SCENES}) >= 7


def test_second_run_changes_nothing(placeholder_repo):
    before = file_hashes(placeholder_repo.root)
    report = run_placeholders(placeholder_repo)
    assert file_hashes(placeholder_repo.root) == before
    assert report.scenes_json == "unchanged" and report.manifest == "kept"


def test_real_masters_are_never_overwritten(placeholder_repo):
    real = placeholder_repo.master("prison-yard")
    Image.new("RGB", (MASTER_WIDTH, MASTER_HEIGHT), (10, 20, 30)).save(real)
    real_bytes = real.read_bytes()
    scenes_json = placeholder_repo.scenes_json.read_bytes()

    report = run_placeholders(placeholder_repo)

    assert real.read_bytes() == real_bytes
    assert report.kept == ["prison-yard"]
    assert report.scenes_json == "skipped"
    assert placeholder_repo.scenes_json.read_bytes() == scenes_json


def test_authored_yaml_is_kept_and_flows_into_scenes_json(placeholder_repo):
    manifest = load_manifest(placeholder_repo.scenes_yaml)
    manifest.scenes[0].title = "An authored title"
    manifest.scenes[0].bridge.polygon = []  # as the content package leaves it
    save_manifest(manifest, placeholder_repo.scenes_yaml)
    yaml_text = placeholder_repo.scenes_yaml.read_text()

    run_placeholders(placeholder_repo)

    assert placeholder_repo.scenes_yaml.read_text() == yaml_text
    built = load_built_manifest(placeholder_repo.scenes_json)
    assert built.scenes[0].title == "An authored title"
    assert len(built.scenes[0].bridge.polygon) == len(PLACEHOLDER_SCENES[0].design.outline)


def test_bracketed_numbers_in_titles_do_not_cause_drift(placeholder_repo):
    manifest = load_manifest(placeholder_repo.scenes_yaml)
    manifest.scenes[0].title = "Room [1,2]"
    manifest.scenes[1].alt = "A yard with lamps [3,4]."
    save_manifest(manifest, placeholder_repo.scenes_yaml)

    run_placeholders(placeholder_repo)

    built = load_built_manifest(placeholder_repo.scenes_json)
    assert built.scenes[0].title == "Room [1,2]"
    assert built.scenes[1].alt == "A yard with lamps [3,4]."
    assert validate_repo(placeholder_repo).issues == []


def test_force_restores_placeholder_content(placeholder_repo):
    manifest = load_manifest(placeholder_repo.scenes_yaml)
    manifest.scenes[0].title = "An authored title"
    save_manifest(manifest, placeholder_repo.scenes_yaml)
    real = placeholder_repo.master("prison-yard")
    Image.new("RGB", (MASTER_WIDTH, MASTER_HEIGHT)).save(real)

    report = run_placeholders(placeholder_repo, force=True)

    assert report.manifest == "written"
    # Restoring the original placeholder content gives back the original scenes.json.
    assert report.scenes_json == "unchanged"
    assert load_manifest(placeholder_repo.scenes_yaml) == placeholder_manifest()
    assert is_placeholder_image(real)


def test_stale_exports_are_removed(placeholder_repo):
    stale = placeholder_repo.export_dir("cliff-cottage") / "cliff-cottage-640.deadbeef.webp"
    stale.write_bytes(b"old")
    unrelated = placeholder_repo.export_dir("cliff-cottage") / "notes.txt"
    unrelated.write_text("keep me")
    run_placeholders(placeholder_repo)
    assert not stale.exists()
    assert unrelated.exists()

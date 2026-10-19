from __future__ import annotations

from PIL import Image

from dreamscapes_pipeline.export import run_export, verify_exports
from dreamscapes_pipeline.layout import MASTER_HEIGHT, MASTER_WIDTH
from dreamscapes_pipeline.manifest import load_built_manifest, load_manifest, save_manifest
from dreamscapes_pipeline.placeholders import PLACEHOLDERS_BY_ID, run_placeholders
from dreamscapes_pipeline.validate import validate_repo
from tests.conftest import PLACEHOLDER_SCENE_COUNT, file_hashes
from tests.fakes import offline, write_master  # noqa: F401


def _graded(paths, scene_id, colour):
    Image.new("RGB", (MASTER_WIDTH, MASTER_HEIGHT), colour).save(paths.graded(scene_id))


def test_export_reproduces_the_placeholder_output_byte_for_byte(placeholder_repo):
    """With authored scenes.yaml polygons still empty (as the content package ships them),
    `dsp export` must compile exactly what `dsp placeholders` wrote."""
    manifest = load_manifest(placeholder_repo.scenes_yaml)
    for scene in manifest.scenes:
        scene.bridge.polygon = []
        scene.title = f"Authored {scene.title}"
    save_manifest(manifest, placeholder_repo.scenes_yaml)
    run_placeholders(placeholder_repo)
    before = file_hashes(placeholder_repo.root)

    report = run_export(placeholder_repo)
    assert report.errors == [] and report.changed == [] and report.scenes_json == "unchanged"
    assert len(report.scenes) == PLACEHOLDER_SCENE_COUNT
    assert file_hashes(placeholder_repo.root) == before
    built = load_built_manifest(placeholder_repo.scenes_json)
    assert built.scenes[0].title.startswith("Authored")
    expected = PLACEHOLDERS_BY_ID["cliff-cottage"].design.polygon()
    assert [tuple(p) for p in built.scenes[0].bridge.polygon] == [tuple(p) for p in expected]


def test_check_reports_drift_and_export_fixes_it_once(small_repo):
    paths = small_repo
    for sid, colour in (("first-scene", (90, 120, 140)), ("second-scene", (140, 90, 120))):
        _graded(paths, sid, colour)
    _graded(paths, "third-scene", (100, 100, 60))

    check = run_export(paths, check=True)
    assert check.errors == [] and check.scenes_json == "would change"
    assert "web/public/scenes.json" in check.changed and len(check.changed) == 25
    assert not paths.scenes_json.exists() and not paths.dreams.exists()

    report = run_export(paths)
    assert report.errors == [] and report.scenes_json == "written"
    assert validate_repo(paths).errors == []
    assert run_export(paths, check=True).changed == []

    old = sorted(p.name for p in paths.export_dir("second-scene").iterdir())
    _graded(paths, "second-scene", (20, 30, 40))
    report = run_export(paths)
    new = sorted(p.name for p in paths.export_dir("second-scene").iterdir())
    assert len(new) == 8 and not set(new) & set(old), "stale variants are removed"
    assert report.scenes_json == "written"
    assert len(report.changed) == 17, "8 written, 8 removed, scenes.json"

    before = file_hashes(paths.root)
    assert run_export(paths).changed == []
    assert file_hashes(paths.root) == before


def test_problems_are_reported_before_anything_is_encoded_or_written(placeholder_repo):
    paths = placeholder_repo
    manifest = load_manifest(paths.scenes_yaml)
    manifest.scene("rooftop-court").bridge.polygon = []
    save_manifest(manifest, paths.scenes_yaml)
    write_master(paths, "rooftop-court", (50, 60, 70))
    _graded(paths, "rooftop-court", (40, 50, 60))
    paths.graded("endless-library").unlink()
    Image.new("RGB", (640, 360)).save(paths.graded("drowned-street"))
    before = file_hashes(paths.root)
    report = run_export(paths)
    assert report.scenes_json == "skipped"
    assert any("rooftop-court: has no hotspot polygon" in e for e in report.errors)
    assert any("endless-library.graded.png is missing" in e for e in report.errors)
    assert any("drowned-street.graded.png is 640x360" in e for e in report.errors)
    assert file_hashes(paths.root) == before


def test_verify_detects_a_file_that_no_longer_matches_its_hash(placeholder_repo):
    built = load_built_manifest(placeholder_repo.scenes_json)
    assert verify_exports(placeholder_repo, built) == []
    target = placeholder_repo.public_file(built.scenes[0].image.sources.webp[0].src)
    target.write_bytes(b"tampered")
    problems = verify_exports(placeholder_repo, built)
    assert len(problems) == 1 and "does not match its hash" in problems[0]
    target.unlink()
    assert "is missing" in verify_exports(placeholder_repo, built)[0]

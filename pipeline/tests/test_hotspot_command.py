from __future__ import annotations

import json

import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

from dreamscapes_pipeline import geometry
from dreamscapes_pipeline.cli import app
from dreamscapes_pipeline.hotspots import (
    BRIDGE_FIELD_ORDER,
    Candidate,
    ClaudeVerifier,
    FakeSegmenter,
    FakeVerifier,
    GeminiSegmenter,
    Sam3Segmenter,
    Verdict,
    run_hotspot_command,
)
from dreamscapes_pipeline.manifest import load_manifest
from dreamscapes_pipeline.validate import validate_repo
from dreamscapes_pipeline.yaml_patch import set_scene_fields

runner = CliRunner()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """No test may build a real API client; the providers are replaced with fakes."""

    def injected_only(self):
        if self._client is None:
            raise AssertionError("a test tried to build a real API client")
        return self._client

    for provider in (GeminiSegmenter, ClaudeVerifier, Sam3Segmenter):
        monkeypatch.setattr(provider, "client", property(injected_only))
    for name in ("DSP_SEGMENTATION_PROVIDER", "DSP_MAX_SPEND_USD"):
        monkeypatch.delenv(name, raising=False)


def hoop() -> Candidate:
    polygon = [(0.56, 0.32), (0.68, 0.32), (0.68, 0.52), (0.56, 0.52)]
    return Candidate(polygon, geometry.bbox(polygon), "basketball hoop")


def run(paths, *, scene="prison-yard", all_scenes=False, box=None, segmenter=None, verifier=None):
    lines: list[str] = []
    code = run_hotspot_command(
        paths,
        scene=None if all_scenes else scene,
        all_scenes=all_scenes,
        box=box,
        dry_run=False,
        echo=lines.append,
        segmenter=segmenter or FakeSegmenter([[hoop()]]),
        verifier=verifier or FakeVerifier([Verdict(matches=True, reason="The hoop.")]),
    )
    return code, "\n".join(lines)


# --- dry run through the CLI --------------------------------------------------------------------


def test_dry_run_shows_the_whole_pipeline_and_changes_nothing(placeholder_repo, monkeypatch):
    monkeypatch.chdir(placeholder_repo.root)
    yaml_before = placeholder_repo.scenes_yaml.read_bytes()
    mask_before = placeholder_repo.mask("prison-yard").read_bytes()
    result = runner.invoke(app, ["hotspot", "--scene", "prison-yard", "--dry-run"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "would send to gemini-3.8-flash" in out
    assert "Give the segmentation masks for the basketball hoop." in out
    assert '"thinking_level": "low"' in out
    assert "would ask Claude to verify it" in out and '"model": "claude-opus-5-5"' in out
    assert "would write to content/scenes.yaml" in out and "source: gemini" in out
    assert "hotspot rules: ok" in out
    assert "done: 1 verified" in out
    assert placeholder_repo.scenes_yaml.read_bytes() == yaml_before
    assert placeholder_repo.mask("prison-yard").read_bytes() == mask_before
    review = placeholder_repo.review_dir("prison-yard")
    with Image.open(review / "hotspot.png") as overlay:
        assert overlay.size == (2048, 1152)
    record = json.loads((review / "hotspot.json").read_text())
    assert record["dry_run"] is True and record["verified"] is True
    # The cleaned outline still covers the placeholder's rectangle, a little grown.
    polygon = record["updates"]["polygon"]
    assert len(polygon) <= 40
    assert 0.032 < record["updates"]["area"] < 0.04


def test_dry_run_with_a_box_hint_shows_it_in_the_prompt(placeholder_repo, monkeypatch):
    monkeypatch.chdir(placeholder_repo.root)
    args = ["hotspot", "--scene", "prison-yard", "--box", "0.5", "0.3", "0.7", "0.55", "--dry-run"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "inside the bounding box [300, 500, 550, 700]" in result.output


def test_dry_run_all_scenes(placeholder_repo, monkeypatch):
    monkeypatch.chdir(placeholder_repo.root)
    result = runner.invoke(app, ["hotspot", "--all", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "done: 10 verified, 0 need a manual hotspot, 0 failed, 0 break the hotspot rules" in (
        result.output
    )


@pytest.mark.parametrize(
    ("args", "code", "message"),
    [
        (["hotspot", "--dry-run"], 2, "exactly one of --scene"),
        (["hotspot", "--scene", "prison-yard", "--all"], 2, "exactly one of --scene"),
        (["hotspot", "--all", "--box", "0.1", "0.1", "0.2", "0.2"], 2, "use it with --scene"),
        (["hotspot", "--scene", "prison-yard", "--box", "0.5", "0.1", "0.2", "0.2"], 2, "x0 < x1"),
        (["hotspot", "--scene", "nowhere", "--dry-run"], 1, "no scene 'nowhere'"),
    ],
)
def test_option_errors(placeholder_repo, monkeypatch, args, code, message):
    monkeypatch.chdir(placeholder_repo.root)
    result = runner.invoke(app, args)
    assert result.exit_code == code, result.output
    assert message in result.output


def test_unknown_provider_is_rejected(placeholder_repo, monkeypatch):
    monkeypatch.chdir(placeholder_repo.root)
    monkeypatch.setenv("DSP_SEGMENTATION_PROVIDER", "crayons")
    result = runner.invoke(app, ["hotspot", "--scene", "prison-yard", "--dry-run"])
    assert result.exit_code == 2
    assert "DSP_SEGMENTATION_PROVIDER must be gemini or sam3" in result.output


def test_sam3_dry_run_shows_the_fal_request(placeholder_repo, monkeypatch):
    monkeypatch.chdir(placeholder_repo.root)
    monkeypatch.setenv("DSP_SEGMENTATION_PROVIDER", "sam3")
    result = runner.invoke(app, ["hotspot", "--scene", "prison-yard", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "would send to fal-ai/sam-3/image" in result.output
    assert "source: sam3" in result.output


# --- real runs with fake providers --------------------------------------------------------------


def test_write_back_touches_only_the_scene_hotspot_fields(placeholder_repo):
    before = placeholder_repo.scenes_yaml.read_text()
    code, out = run(placeholder_repo)
    assert code == 0, out
    after = placeholder_repo.scenes_yaml.read_text()

    old, new = yaml.safe_load(before), yaml.safe_load(after)
    assert list(new) == list(old)
    for i, (a, b) in enumerate(zip(old["scenes"], new["scenes"], strict=True)):
        if a["id"] != "prison-yard":
            assert a == b, f"scene {i} changed"
            continue
        assert list(b) == list(a)  # scene keys keep their order
        assert {k: v for k, v in b.items() if k != "bridge"} == {
            k: v for k, v in a.items() if k != "bridge"
        }
        bridge = b["bridge"]
        assert (
            bridge["object"] == "basketball hoop"
            and bridge["placement"] == a["bridge"]["placement"]
        )
        assert bridge["source"] == "gemini" and bridge["verified"] is True
        assert bridge["verification"] == "The hoop."
        assert bridge["needs_manual_hotspot"] is False
        assert list(bridge) == [k for k in BRIDGE_FIELD_ORDER if k in bridge]
    # Lines outside the prison-yard bridge block are unchanged.
    old_lines, new_lines = before.splitlines(), after.splitlines()
    start = old_lines.index("  - id: prison-yard")
    assert new_lines[:start] == old_lines[:start]
    end = old_lines.index("  - id: rooftop-court")
    assert new_lines[new_lines.index("  - id: rooftop-court") :] == old_lines[end:]

    manifest = load_manifest(placeholder_repo.scenes_yaml)
    polygon = manifest.scene("prison-yard").bridge.polygon
    with Image.open(placeholder_repo.mask("prison-yard")) as mask:
        assert mask.size == (2048, 1152)
        inside = mask.getpixel((round(0.62 * 2048), round(0.42 * 1152)))
        assert inside == 255
    assert (placeholder_repo.review_dir("prison-yard") / "hotspot.png").exists()
    assert len(polygon) <= 40
    result = validate_repo(placeholder_repo)
    assert result.errors == []  # only the expected "scenes.json is out of date" warning remains
    assert all("out of date" in str(w) for w in result.warnings)


def test_comments_in_the_manifest_survive(placeholder_repo):
    text = placeholder_repo.scenes_yaml.read_text()
    text = "# Authored by hand.\n" + text.replace(
        "    title: The yard\n", "    title: The yard  # keep this short\n"
    )
    placeholder_repo.scenes_yaml.write_text(text)
    code, out = run(placeholder_repo)
    assert code == 0, out
    after = placeholder_repo.scenes_yaml.read_text()
    assert after.startswith("# Authored by hand.\n")
    assert "    title: The yard  # keep this short\n" in after
    assert "comments were dropped" not in out


def test_two_failures_flag_the_scene_and_exit_1(placeholder_repo):
    verifier = FakeVerifier([Verdict(matches=False, reason="That is a fence.")])
    code, out = run(placeholder_repo, verifier=verifier)
    assert code == 1
    assert "needs a manual hotspot" in out
    bridge = load_manifest(placeholder_repo.scenes_yaml).scene("prison-yard").bridge
    assert bridge.needs_manual_hotspot and not bridge.verified
    assert "That is a fence." in bridge.verification
    warnings = [str(w) for w in validate_repo(placeholder_repo).warnings]
    assert any("needing manual annotation" in w for w in warnings)


def test_all_keeps_hand_drawn_hotspots(placeholder_repo):
    text = placeholder_repo.scenes_yaml.read_text()
    patched = set_scene_fields(
        text, "prison-yard", "bridge", {"source": "manual"}, BRIDGE_FIELD_ORDER
    )
    placeholder_repo.scenes_yaml.write_text(patched.text)
    segmenter = FakeSegmenter([[hoop()]])
    code, out = run(placeholder_repo, all_scenes=True, segmenter=segmenter)
    assert code == 0, out
    assert "prison-yard: kept its hand-drawn hotspot" in out
    assert len(segmenter.calls) == 9
    assert (
        load_manifest(placeholder_repo.scenes_yaml).scene("prison-yard").bridge.source == "manual"
    )


def test_graded_master_is_preferred(placeholder_repo):
    segmenter = FakeSegmenter([[hoop()]])
    run(placeholder_repo, segmenter=segmenter)
    assert segmenter.calls[0][0] == placeholder_repo.graded("prison-yard")
    placeholder_repo.graded("prison-yard").unlink()
    run(placeholder_repo, segmenter=segmenter)
    assert segmenter.calls[-1][0] == placeholder_repo.master("prison-yard")


def test_missing_master_is_an_error(placeholder_repo):
    placeholder_repo.graded("prison-yard").unlink()
    placeholder_repo.master("prison-yard").unlink()
    code, out = run(placeholder_repo)
    assert code == 1 and "no master image" in out


def test_spend_cap_refuses_before_any_call(placeholder_repo, monkeypatch):
    monkeypatch.setenv("DSP_MAX_SPEND_USD", "0.1")
    segmenter = FakeSegmenter([[hoop()]])
    segmenter.call_estimate_usd = 0.01
    verifier = FakeVerifier([Verdict(matches=True, reason="ok")])
    verifier.call_estimate_usd = 0.09
    code, out = run(placeholder_repo, segmenter=segmenter, verifier=verifier)
    assert code == 1
    assert "estimated spend: up to $0.20 for 1 scene(s) (cap $0.10" in out
    assert segmenter.calls == [] and verifier.calls == []

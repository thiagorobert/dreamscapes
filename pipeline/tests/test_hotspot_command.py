from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

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


def run(
    paths,
    *,
    scene="prison-yard",
    all_scenes=False,
    box=None,
    force=False,
    segmenter=None,
    verifier=None,
):
    lines: list[str] = []
    code = run_hotspot_command(
        paths,
        scene=None if all_scenes else scene,
        all_scenes=all_scenes,
        box=box,
        dry_run=False,
        force=force,
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


def court() -> Candidate:
    """A wrong answer: the whole basketball court, about 31% of the frame."""
    polygon = [(0.2, 0.3), (0.8, 0.3), (0.8, 0.82), (0.2, 0.82)]
    return Candidate(polygon, geometry.bbox(polygon), "basketball court")


REJECT_TWICE = [
    Verdict(matches=False, reason="That is the whole court.", suggested_box=(0.56, 0.3, 0.7, 0.5)),
    Verdict(matches=False, reason="Still the whole court."),
]


def test_failed_run_flags_the_scene_and_leaves_its_hotspot_alone(placeholder_repo):
    before = load_manifest(placeholder_repo.scenes_yaml).scene("prison-yard").bridge
    mask_before = placeholder_repo.mask("prison-yard").read_bytes()
    segmenter = FakeSegmenter([[court()]])
    code, out = run(placeholder_repo, segmenter=segmenter, verifier=FakeVerifier(REJECT_TWICE))
    assert code == 1
    assert "needs a manual hotspot" in out
    assert len(segmenter.calls) == 2
    after = load_manifest(placeholder_repo.scenes_yaml).scene("prison-yard").bridge
    assert after.needs_manual_hotspot and not after.verified
    assert "That is the whole court." in after.verification
    assert "Still the whole court." in after.verification
    unchanged = ("polygon", "source", "score", "bbox", "centroid", "area", "object", "placement")
    assert {k: getattr(after, k) for k in unchanged} == {k: getattr(before, k) for k in unchanged}
    assert placeholder_repo.mask("prison-yard").read_bytes() == mask_before
    review = placeholder_repo.review_dir("prison-yard")
    assert (review / "hotspot-1.png").exists() and (review / "hotspot-2.png").exists()
    result = validate_repo(placeholder_repo)
    assert result.errors == []  # what CI runs still passes
    assert any("needing manual annotation" in str(w) for w in result.warnings)


def test_a_verified_hotspot_survives_a_failed_rerun(placeholder_repo):
    code, out = run(placeholder_repo)  # first run: verified
    assert code == 0, out
    good = load_manifest(placeholder_repo.scenes_yaml).scene("prison-yard").bridge
    assert good.verified and good.source == "gemini"
    mask_good = placeholder_repo.mask("prison-yard").read_bytes()

    code, out = run(
        placeholder_repo,
        segmenter=FakeSegmenter([[court()]]),
        verifier=FakeVerifier(REJECT_TWICE),
    )
    assert code == 1
    bad = load_manifest(placeholder_repo.scenes_yaml).scene("prison-yard").bridge
    assert bad.polygon == good.polygon
    assert (bad.source, bad.bbox, bad.centroid, bad.area) == (
        good.source,
        good.bbox,
        good.centroid,
        good.area,
    )
    assert not bad.verified and bad.needs_manual_hotspot
    assert placeholder_repo.mask("prison-yard").read_bytes() == mask_good
    assert validate_repo(placeholder_repo).errors == []


def test_all_skips_verified_scenes_unless_forced(placeholder_repo):
    assert run(placeholder_repo)[0] == 0  # prison-yard is now verified
    segmenter = FakeSegmenter([[hoop()]])
    code, out = run(placeholder_repo, all_scenes=True, segmenter=segmenter)
    assert code == 0, out
    assert "prison-yard: already verified, skipped" in out
    assert len(segmenter.calls) == 9
    assert "estimated spend: up to $0.00 for 9 scene(s)" in out

    segmenter = FakeSegmenter([[hoop()]])
    code, out = run(placeholder_repo, all_scenes=True, force=True, segmenter=segmenter)
    assert code == 0, out
    assert "already verified" not in out
    assert len(segmenter.calls) == 10

    segmenter = FakeSegmenter([[hoop()]])
    assert run(placeholder_repo, segmenter=segmenter)[0] == 0  # named: always redone
    assert len(segmenter.calls) == 1


def test_dry_run_help_mentions_the_review_files():
    result = runner.invoke(app, ["hotspot", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    # Rich styling (forced on under GitHub Actions) adds escape codes and panel borders.
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output).replace("│", " ")
    text = " ".join(plain.split())
    assert "review files are still written to assets/review/" in text
    assert "--force" in text


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


# --- how the mask points were read ---------------------------------------------------------------

SAVED = json.loads(
    (Path(__file__).parent / "fixtures" / "gemini_segmentation_responses.json").read_text()
)


class SavedInteractions:
    """A Gemini client stand-in that answers with saved real responses, in turn."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        usage = SimpleNamespace(
            total_input_tokens=1500, total_output_tokens=400, total_thought_tokens=0
        )
        text = json.dumps(self.responses[min(self.calls, len(self.responses)) - 1])
        return SimpleNamespace(output_text=text, usage=usage, id=f"interaction-{self.calls}")


def saved_gemini(*responses) -> GeminiSegmenter:
    return GeminiSegmenter(client=SimpleNamespace(interactions=SavedInteractions(*responses)))


def test_the_reading_is_recorded_in_the_review_files(placeholder_repo):
    # The prison yard's first real answer came back as [y, x]; its second as [x, y].
    first, second = SAVED["prison-yard"]["responses"]
    verifier = FakeVerifier(
        [
            Verdict(matches=False, reason="Not quite.", suggested_box=(0.6, 0.22, 0.72, 0.47)),
            Verdict(matches=True, reason="The hoop."),
        ]
    )
    code, out = run(placeholder_repo, segmenter=saved_gemini(first, second), verifier=verifier)
    assert "read the mask points as [y, x] across the image (yx-image): 100%" in out
    assert "read the mask points as [x, y] across the image (xy-image): 100%" in out
    assert "warning: even the best reading" not in out
    review = placeholder_repo.review_dir("prison-yard")
    assert (review / "hotspot-1-yx-image.png").exists()
    assert (review / "hotspot-2-xy-image.png").exists()
    assert not (review / "hotspot-1.png").exists()
    record = json.loads((review / "hotspot.json").read_text())
    one, two = record["attempts"]
    assert (one["reading"], two["reading"]) == ("yx-image", "xy-image")
    assert (one["overlay"], two["overlay"]) == ("hotspot-1-yx-image.png", "hotspot-2-xy-image.png")
    assert one["points_outside_box"] == 0
    readings = one["candidates"][0]["readings"]
    assert list(readings) == ["xy-image", "yx-image", "xy-box", "yx-box"]
    assert readings["yx-image"]["inside"] == 1.0
    assert readings["xy-image"]["inside"] < 0.1


def test_a_garbled_mask_is_warned_about(placeholder_repo):
    mixed = SAVED["overgrown-greenhouse"]["responses"][0]
    verifier = FakeVerifier([Verdict(matches=False, reason="It includes a wedge of wall.")])
    code, out = run(placeholder_repo, segmenter=saved_gemini(mixed), verifier=verifier)
    assert code == 1
    assert (
        "warning: even the best reading leaves 8% of the mask points outside the box Gemini gave"
        in out
    )
    record = json.loads((placeholder_repo.review_dir("prison-yard") / "hotspot.json").read_text())
    assert record["attempts"][0]["points_outside_box"] == pytest.approx(0.08, abs=0.005)

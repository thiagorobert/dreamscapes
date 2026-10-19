"""Command-line tests for generate, select, grade, export, review and bakeoff."""

from __future__ import annotations

import pytest
from PIL import Image
from typer.testing import CliRunner

from dreamscapes_pipeline import providers
from dreamscapes_pipeline import select as selection
from dreamscapes_pipeline.cli import app
from dreamscapes_pipeline.layout import MASTER_HEIGHT, MASTER_WIDTH
from dreamscapes_pipeline.manifest import load_manifest
from tests.conftest import file_hashes
from tests.fakes import FakeAnthropicClient, offline, write_style  # noqa: F401
from tests.test_cli import plain_help

runner = CliRunner()


@pytest.fixture
def in_placeholder_repo(placeholder_repo, monkeypatch):
    monkeypatch.chdir(placeholder_repo.root)
    return placeholder_repo


@pytest.fixture
def in_small_repo(small_repo, monkeypatch):
    monkeypatch.chdir(small_repo.root)
    write_style(small_repo)
    return small_repo


def test_help_shows_the_new_options():
    assert "--size" in plain_help(["generate", "--help"])
    assert "--quality" in plain_help(["generate", "--help"])
    assert "--check" in plain_help(["export", "--help"])
    assert "--candidates" in plain_help(["bakeoff", "--help"])


def test_generate_dry_run_prints_the_exact_gemini_request(in_placeholder_repo):
    result = runner.invoke(app, ["generate", "--scene", "cliff-cottage", "--dry-run"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert '"endpoint": "client.interactions.create"' in out
    assert '"model": "gemini-nano-banana-2.1"' in out
    assert '"aspect_ratio": "16:9"' in out and '"image_size": "2K"' in out
    assert "Include a lighthouse" in out
    assert "request 4 of 4" in out and "limit $15.00" in out
    assert "dry run: nothing was sent" in out
    assert not in_placeholder_repo.candidates.exists()


def test_generate_all_puts_the_style_anchor_first(in_placeholder_repo):
    result = runner.invoke(app, ["generate", "--all", "--candidates", "1", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert result.output.index("cliff-cottage cand-1") < result.output.index("prison-yard cand-1")
    assert result.output.count("a real run would refuse") == 9


def test_generate_openai_dry_run_with_size_and_quality(in_placeholder_repo):
    args = ["generate", "--scene", "cliff-cottage", "--candidates", "1", "--dry-run"]
    result = runner.invoke(app, [*args, "--provider", "openai", "--quality", "low", "--size", "1K"])
    assert result.exit_code == 0, result.output
    assert '"endpoint": "client.images.generate"' in result.output
    assert '"size": "1280x720"' in result.output and '"quality": "low"' in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["generate", "--dry-run"], "pass --scene <id> or --all"),
        (["generate", "--all", "--scene", "prison-yard"], "not both"),
        (["generate", "--scene", "nowhere"], "unknown scene 'nowhere'"),
        (["generate", "--scene", "prison-yard", "--quality", "high"], "only applies to"),
        (["generate", "--scene", "prison-yard", "--provider", "flux"], "unknown image provider"),
        (["select"], "pass --scene <id> or --all"),
    ],
)
def test_bad_arguments_fail_clearly(in_placeholder_repo, args, message):
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert message in result.output


def test_spend_limit_from_the_environment(in_placeholder_repo, monkeypatch):
    args = ["generate", "--scene", "cliff-cottage", "--dry-run"]
    monkeypatch.setenv("DSP_MAX_SPEND_USD", "")
    assert "limit $15.00" in runner.invoke(app, args).output
    monkeypatch.setenv("DSP_MAX_SPEND_USD", "0.05")
    assert "a real run would refuse" in runner.invoke(app, args).output
    monkeypatch.setenv("DSP_MAX_SPEND_USD", "plenty")
    result = runner.invoke(app, args)
    assert result.exit_code == 1 and "must be a number" in result.output


def test_real_generate_without_a_key_stops_without_logging(in_small_repo):
    result = runner.invoke(app, ["generate", "--scene", "first-scene", "--candidates", "1"])
    assert result.exit_code == 1
    assert "GEMINI_API_KEY is not set" in result.output
    assert not in_small_repo.generation_log.exists()


def test_generate_then_select_end_to_end(in_small_repo, monkeypatch):
    fake = providers.FakeImageProvider(image_size=(512, 288))
    monkeypatch.setattr(providers, "make_image_provider", lambda *a, **k: fake)
    client = FakeAnthropicClient(best=2)
    monkeypatch.setattr(
        selection, "make_judge", lambda root=None: selection.ClaudeJudge(client=client, root=root)
    )
    result = runner.invoke(app, ["generate", "--scene", "second-scene", "--candidates", "2"])
    assert result.exit_code == 0, result.output
    assert len(fake.calls) == 2

    result = runner.invoke(app, ["select", "--scene", "second-scene"])
    assert result.exit_code == 0, result.output
    assert "selected cand-2" in result.output and "updated content/scenes.yaml" in result.output
    generation = load_manifest(in_small_repo.scenes_yaml).scene("second-scene").generation
    assert generation.selected_candidate == 2 and generation.provider == "fake"
    with Image.open(in_small_repo.master("second-scene")) as master:
        assert master.size == (MASTER_WIDTH, MASTER_HEIGHT)

    result = runner.invoke(app, ["select", "--all", "--dry-run"])
    assert result.exit_code == 1, "first and third scene have no candidates"
    assert "pinned to cand-2" in result.output


def test_grade_preview_on_placeholder_masters(in_placeholder_repo):
    before = file_hashes(in_placeholder_repo.root)
    result = runner.invoke(app, ["grade", "--preview"])
    assert result.exit_code == 0, result.output
    assert "left placeholder masters to dsp placeholders" in result.output
    assert "wrote assets/review/grade.html" in result.output
    after = {k: v for k, v in file_hashes(in_placeholder_repo.root).items() if "review" not in k}
    assert after == before


def test_grade_reports_bad_parameter_files(in_placeholder_repo):
    in_placeholder_repo.grade_toml.write_text("[bloom]\nmix = 3\n")
    result = runner.invoke(app, ["grade"])
    assert result.exit_code == 1 and "bloom.mix" in result.output


def test_export_and_check(in_placeholder_repo):
    before = file_hashes(in_placeholder_repo.root)
    result = runner.invoke(app, ["export", "--check"])
    assert result.exit_code == 0, result.output
    assert "up to date (10 scenes)" in result.output
    result = runner.invoke(app, ["export"])
    assert result.exit_code == 0, result.output
    assert "0 file(s) changed" in result.output and "scenes.json is unchanged" in result.output
    assert file_hashes(in_placeholder_repo.root) == before

    in_placeholder_repo.graded("fog-ship-deck").unlink()
    result = runner.invoke(app, ["export"])
    assert result.exit_code == 1 and "fog-ship-deck.graded.png is missing" in result.output


def test_export_check_fails_on_drift(in_small_repo):
    for scene in ("first-scene", "second-scene", "third-scene"):
        Image.new("RGB", (MASTER_WIDTH, MASTER_HEIGHT), (80, 90, 100)).save(
            in_small_repo.graded(scene)
        )
    result = runner.invoke(app, ["export", "--check"])
    assert result.exit_code == 1 and "web/public/scenes.json" in result.output
    assert runner.invoke(app, ["export"]).exit_code == 0
    assert runner.invoke(app, ["export", "--check"]).exit_code == 0


def test_review_and_bakeoff_dry_run(in_placeholder_repo):
    result = runner.invoke(app, ["review"])
    assert result.exit_code == 0, result.output
    assert "wrote assets/review/index.html" in result.output
    result = runner.invoke(app, ["bakeoff", "--dry-run", "--candidates", "2", "--quality", "low"])
    assert result.exit_code == 0, result.output
    assert "_bakeoff/gemini/cand-2.png" in result.output
    assert "_bakeoff/openai/cand-2.png" in result.output
    assert '"quality": "low"' in result.output
    assert "one Claude scoring call over 4" in result.output

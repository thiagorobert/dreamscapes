from __future__ import annotations

import pytest
from typer.testing import CliRunner

from dreamscapes_pipeline.cli import app

runner = CliRunner()

COMMANDS = [
    "validate",
    "schema",
    "placeholders",
    "generate",
    "select",
    "grade",
    "hotspot",
    "export",
    "review",
    "morph",
    "bakeoff",
]

STUBS = [
    ["generate", "--scene", "cliff-cottage", "--candidates", "6", "--provider", "openai"],
    ["generate", "--all", "--no-reference", "--force", "--dry-run"],
    ["select", "--scene", "cliff-cottage", "--repick"],
    ["select", "--all", "--dry-run"],
    ["grade", "--preview"],
    ["hotspot", "--scene", "prison-yard", "--box", "0.1", "0.2", "0.3", "0.4", "--dry-run"],
    ["hotspot", "--all"],
    ["export"],
    ["review"],
    ["morph", "--edge", "cliff-cottage", "prison-yard", "--dry-run"],
    ["morph", "--all"],
    ["bakeoff", "--dry-run"],
]


def test_help_lists_every_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.output


@pytest.mark.parametrize("args", STUBS, ids=lambda a: " ".join(a))
def test_unimplemented_commands_accept_their_options_and_exit_2(args, small_repo, monkeypatch):
    monkeypatch.chdir(small_repo.root)
    result = runner.invoke(app, args)
    assert result.exit_code == 2, result.output
    assert f"dsp {args[0]}: not implemented yet" in result.output


def test_commands_find_the_root_from_a_subdirectory(placeholder_repo, monkeypatch):
    subdir = placeholder_repo.root / "web" / "public" / "dreams"
    monkeypatch.chdir(subdir)
    result = runner.invoke(app, ["validate"])
    assert result.exit_code == 0, result.output
    assert "0 error(s), 0 warning(s)" in result.output


def test_outside_a_repository_commands_fail_clearly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["validate"])
    assert result.exit_code == 1
    assert "could not find the Dreamscapes repository root" in result.output


def test_validate_exit_codes(small_repo, monkeypatch):
    monkeypatch.chdir(small_repo.root)
    # The small repository has no scenes.json yet: one warning, no errors.
    assert runner.invoke(app, ["validate"]).exit_code == 0
    assert runner.invoke(app, ["validate", "--strict"]).exit_code == 1
    small_repo.master("first-scene").unlink()
    result = runner.invoke(app, ["validate"])
    assert result.exit_code == 1
    assert "1 error(s)" in result.output


def test_schema_writes_and_checks(small_repo, monkeypatch):
    monkeypatch.chdir(small_repo.root)
    assert runner.invoke(app, ["schema", "--check"]).exit_code == 1
    result = runner.invoke(app, ["schema"])
    assert result.exit_code == 0 and "wrote schema/scenes.schema.json" in result.output
    assert runner.invoke(app, ["schema"]).output.startswith("unchanged")
    assert runner.invoke(app, ["schema", "--check"]).exit_code == 0


def test_placeholders_command_reports_what_it_did(placeholder_repo, monkeypatch):
    monkeypatch.chdir(placeholder_repo.root)
    result = runner.invoke(app, ["placeholders"])
    assert result.exit_code == 0, result.output
    assert "kept existing content/scenes.yaml" in result.output
    assert "web/public/scenes.json is unchanged" in result.output


def test_placeholders_destructive_flags(placeholder_repo, monkeypatch):
    monkeypatch.chdir(placeholder_repo.root)
    help_text = runner.invoke(app, ["placeholders", "--help"]).output
    assert "--overwrite-masters" in help_text and "--reset-manifest" in help_text
    # The old catch-all flag is gone so nobody destroys both kinds of work by habit.
    assert runner.invoke(app, ["placeholders", "--force"]).exit_code == 2
    result = runner.invoke(app, ["placeholders", "--reset-manifest", "--overwrite-masters"])
    assert result.exit_code == 0, result.output
    assert "content/scenes.yaml is unchanged" in result.output

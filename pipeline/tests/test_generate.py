from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from dreamscapes_pipeline import genlog
from dreamscapes_pipeline.generate import (
    DRAFT_STYLE_BLOCK,
    REFERENCE_INSTRUCTION,
    StyleBlock,
    StyleBlockError,
    build_prompt,
    existing_candidates,
    load_style_block,
    parse_style_block,
    run_generate,
    seed_for,
    with_definite_article,
    with_indefinite_article,
)
from dreamscapes_pipeline.imaging import sha256_file
from dreamscapes_pipeline.manifest import load_manifest
from dreamscapes_pipeline.providers import FakeImageProvider, GenerationResult, ProviderError
from dreamscapes_pipeline.spend import SpendGuard
from tests.fakes import offline, write_style  # noqa: F401

ROW_FIELDS = {
    "id",
    "timestamp",
    "kind",
    "purpose",
    "provider",
    "model",
    "scene",
    "candidate",
    "prompt",
    "parameters",
    "references",
    "seed",
    "request_id",
    "response_id",
    "revised_prompt",
    "output_path",
    "output_sha256",
    "output_size",
    "cost_estimate_usd",
    "cost_usd",
    "usage",
    "style_block_sha256",
    "status",
    "error",
    "duration_s",
}


# --------------------------------------------------------------------------------------------
# Style block
# --------------------------------------------------------------------------------------------


def test_style_block_is_the_first_paragraph_under_its_heading():
    markdown = (
        "# Dreamscapes style bible\n\nIntro text.\n\n## Style block\n\n"
        "A still, dreamlike scene. Calm, melancholic, inviting.\n\nA second paragraph.\n\n"
        "## Do not\n\n- never this\n"
    )
    assert parse_style_block(markdown) == "A still, dreamlike scene. Calm, melancholic, inviting."


@pytest.mark.parametrize(
    "section",
    [
        "A still scene\nwrapped over\nthree lines.",
        "> A still scene\n> wrapped over\n> three lines.",
        "```\nA still scene\nwrapped over\nthree lines.\n```",
    ],
)
def test_wrapped_quoted_or_fenced_blocks_become_one_line(section):
    markdown = f"## Style Block\n\n{section}\n\n## Grade notes\n\nnot this"
    assert parse_style_block(markdown) == "A still scene wrapped over three lines."


@pytest.mark.parametrize(
    ("markdown", "problem"),
    [
        ("# Style\n\nNo block here.", "no '## Style block' heading"),
        ("## Style block\n\n## Next", "empty"),
    ],
)
def test_missing_or_empty_style_block(markdown, problem):
    with pytest.raises(StyleBlockError, match=problem):
        parse_style_block(markdown)


def test_real_runs_need_style_md_but_dry_runs_fall_back_to_the_draft(small_repo):
    with pytest.raises(StyleBlockError, match="content/style.md is missing"):
        load_style_block(small_repo, allow_draft=False)
    draft = load_style_block(small_repo, allow_draft=True)
    assert draft.text == DRAFT_STYLE_BLOCK and "draft" in draft.source
    write_style(small_repo, "The authored block.")
    block = load_style_block(small_repo, allow_draft=False)
    assert block == StyleBlock("The authored block.", "content/style.md")
    assert len(block.sha256) == 64


# --------------------------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------------------------


def test_articles():
    assert with_indefinite_article("lighthouse") == "a lighthouse"
    assert with_indefinite_article("old bell") == "an old bell"
    assert with_indefinite_article("the door") == "the door"
    assert with_definite_article("ship's hull") == "the ship's hull"
    assert with_definite_article("a paper lantern") == "the paper lantern"


def test_prompt_follows_the_template(small_repo):
    scene = load_manifest(small_repo.scenes_yaml).scene("second-scene")
    scene.bridge.placement = "small, left of centre."
    prompt = build_prompt("STYLE.", scene, with_reference=False)
    assert prompt == (
        "STYLE.\n\nScene: A pond where a lantern floats.\n\n"
        "Include a paper boat: small, left of centre. It must be clearly recognisable but not "
        "the main subject, occupying roughly 3-8% of the frame, located within the central 60% "
        "of the image width. Nothing else in the scene should resemble the paper boat."
    )
    with_ref = build_prompt("STYLE.", scene, with_reference=True)
    assert with_ref == prompt + "\n\n" + REFERENCE_INSTRUCTION


def test_seeds_are_stable_per_slot_and_change_per_attempt():
    assert seed_for("a", 1, 0) == seed_for("a", 1, 0)
    assert len({seed_for("a", 1, 0), seed_for("a", 2, 0), seed_for("a", 1, 1)}) == 3
    assert 0 <= seed_for("a", 1, 0) < 2**31


# --------------------------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------------------------


def _run(paths, scene_ids, provider, lines=None, **kwargs):
    manifest = load_manifest(paths.scenes_yaml)
    kwargs.setdefault("count", 2)
    kwargs.setdefault("guard", SpendGuard(limit=15))
    return run_generate(
        paths,
        manifest,
        scene_ids,
        provider,
        echo=(lines.append if lines is not None else lambda _: None),
        sleep=lambda _: None,
        **kwargs,
    )


def test_generates_candidates_with_the_anchor_reference_and_logs_every_call(small_repo):
    write_style(small_repo)
    provider = FakeImageProvider()
    assert _run(small_repo, ["first-scene", "second-scene"], provider) == 0

    assert [n for n, _ in existing_candidates(small_repo, "first-scene")] == [1, 2]
    assert [n for n, _ in existing_candidates(small_repo, "second-scene")] == [1, 2]
    anchor_calls = [c for c in provider.calls if REFERENCE_INSTRUCTION not in c.prompt]
    other_calls = [c for c in provider.calls if REFERENCE_INSTRUCTION in c.prompt]
    assert len(anchor_calls) == 2 and all(not c.references for c in anchor_calls)
    assert len(other_calls) == 2
    for call in other_calls:
        assert [r.path for r in call.references] == [small_repo.master("first-scene")]
    assert all(c.aspect == "16:9" and c.size == "2K" for c in provider.calls)

    rows = genlog.read_rows(small_repo.generation_log)
    assert len(rows) == 4
    style_hash = load_style_block(small_repo, allow_draft=False).sha256
    for row in rows:
        assert set(row) >= ROW_FIELDS, ROW_FIELDS - set(row)
        assert row["status"] == "ok" and row["kind"] == "image" and row["purpose"] == "generate"
        assert row["style_block_sha256"] == style_hash
        out = small_repo.root / row["output_path"]
        assert sha256_file(out) == row["output_sha256"]
        assert row["output_size"] == [256, 144]
        assert row["seed"] == seed_for(row["scene"], row["candidate"], 0)
    second = [r for r in rows if r["scene"] == "second-scene"]
    assert second[0]["references"] == [
        {
            "path": "assets/masters/first-scene.png",
            "sha256": sha256_file(small_repo.master("first-scene")),
            "bytes": small_repo.master("first-scene").stat().st_size,
        }
    ]
    assert second[0]["prompt"].endswith(REFERENCE_INSTRUCTION)


def test_is_idempotent_and_force_regenerates_with_new_seeds(small_repo):
    write_style(small_repo)
    provider = FakeImageProvider()
    _run(small_repo, ["first-scene"], provider)
    first = sha256_file(small_repo.candidate_dir("first-scene") / "cand-1.png")
    lines: list[str] = []
    assert _run(small_repo, ["first-scene"], provider, lines) == 0
    assert len(provider.calls) == 2
    assert "nothing to generate" in lines
    assert any("already generated" in line for line in lines)

    assert _run(small_repo, ["first-scene"], provider, count=3) == 0
    assert len(provider.calls) == 3, "only the missing third candidate is generated"

    _run(small_repo, ["first-scene"], provider, count=1, force=True)
    rows = genlog.read_rows(small_repo.generation_log)
    forced = rows[-1]
    assert forced["candidate"] == 1 and forced["seed"] == seed_for("first-scene", 1, 1)
    assert sha256_file(small_repo.candidate_dir("first-scene") / "cand-1.png") != first


def test_no_reference_leaves_the_anchor_out(small_repo):
    write_style(small_repo)
    provider = FakeImageProvider()
    _run(small_repo, ["second-scene"], provider, count=1, no_reference=True)
    assert provider.calls[0].references == []
    assert REFERENCE_INSTRUCTION not in provider.calls[0].prompt


def test_refuses_to_use_a_placeholder_anchor(placeholder_repo):
    write_style(placeholder_repo)
    provider = FakeImageProvider()
    lines: list[str] = []
    assert _run(placeholder_repo, ["prison-yard"], provider, lines, count=1) == 1
    assert provider.calls == [] and provider.attempts == 0
    assert any("still a placeholder" in line for line in lines)
    # The anchor scene itself needs no reference, so it can be generated.
    assert _run(placeholder_repo, ["cliff-cottage"], provider, count=1) == 0


def test_dry_run_prints_requests_and_sends_nothing(placeholder_repo):
    provider = FakeImageProvider()
    lines: list[str] = []
    assert (
        _run(placeholder_repo, ["cliff-cottage", "prison-yard"], provider, lines, dry_run=True) == 0
    )
    assert provider.attempts == 0
    assert not placeholder_repo.candidates.exists()
    assert not placeholder_repo.generation_log.exists()
    text = "\n".join(lines)
    assert "draft" in text and "a real run would refuse" in text
    assert "request 4 of 4" in text and "dry run: nothing was sent" in text
    first_json = text[text.index("{") : text.index("\n}") + 2]
    assert json.loads(first_json)["model"] == "fake-image-1"


def test_spend_guard_stops_before_any_call(small_repo):
    write_style(small_repo)
    provider = FakeImageProvider(cost=0.01)
    lines: list[str] = []
    code = _run(
        small_repo, ["first-scene"], provider, lines, count=3, guard=SpendGuard(limit=0.025)
    )
    assert code == 1 and provider.attempts == 0
    assert any("refusing to start" in line for line in lines)


def test_spend_guard_stops_mid_run_when_actual_costs_run_high(small_repo):
    write_style(small_repo)

    class Expensive(FakeImageProvider):
        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            return GenerationResult(**{**result.__dict__, "cost_usd": 0.5})

    provider = Expensive(cost=0.01)
    lines: list[str] = []
    code = _run(small_repo, ["first-scene"], provider, lines, count=3, guard=SpendGuard(limit=0.6))
    assert code == 1 and len(provider.calls) == 2
    assert any("would exceed" in line for line in lines)


def test_retries_then_succeeds(small_repo):
    write_style(small_repo)
    provider = FakeImageProvider(
        failures=[ProviderError("429", retryable=True), ProviderError("503", retryable=True)]
    )
    assert _run(small_repo, ["first-scene"], provider, count=1) == 0
    assert provider.attempts == 3
    rows = genlog.read_rows(small_repo.generation_log)
    assert [r["status"] for r in rows] == ["ok"]


def test_a_failing_scene_is_logged_and_the_run_moves_on(small_repo):
    write_style(small_repo)
    provider = FakeImageProvider(failures=[ProviderError("blocked by safety", retryable=False)])
    lines: list[str] = []
    assert _run(small_repo, ["first-scene", "second-scene"], provider, lines) == 1
    rows = genlog.read_rows(small_repo.generation_log)
    assert rows[0]["status"] == "error" and "blocked by safety" in rows[0]["error"]
    assert [(r["scene"], r["candidate"]) for r in rows[1:]] == [
        ("second-scene", 1),
        ("second-scene", 2),
    ]
    assert existing_candidates(small_repo, "first-scene") == []


def test_non_png_output_is_stored_as_png(small_repo):
    write_style(small_repo)

    class Jpeg(FakeImageProvider):
        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            buffer = io.BytesIO()
            Image.open(io.BytesIO(result.image)).save(buffer, format="JPEG")
            return GenerationResult(**{**result.__dict__, "image": buffer.getvalue()})

    _run(small_repo, ["first-scene"], Jpeg(), count=1)
    with Image.open(small_repo.candidate_dir("first-scene") / "cand-1.png") as image:
        assert image.format == "PNG"


def test_unsupported_size_fails_before_sending(small_repo):
    from dreamscapes_pipeline.providers.gemini import GeminiImageProvider

    write_style(small_repo)
    lines: list[str] = []
    assert _run(small_repo, ["first-scene"], GeminiImageProvider(), lines, size="3K") == 1
    assert any("unsupported size" in line for line in lines)

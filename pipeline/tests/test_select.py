from __future__ import annotations

import base64
import io
import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from PIL import Image

from dreamscapes_pipeline import genlog
from dreamscapes_pipeline.generate import run_generate
from dreamscapes_pipeline.imaging import master_png_from, sha256_file
from dreamscapes_pipeline.layout import MASTER_HEIGHT, MASTER_WIDTH
from dreamscapes_pipeline.manifest import Generation, load_manifest, save_manifest
from dreamscapes_pipeline.providers import FakeImageProvider
from dreamscapes_pipeline.select import (
    FALLBACK_BETA,
    JUDGE_MODEL,
    WEIGHTS,
    ClaudeJudge,
    ScoringError,
    build_job,
    run_select,
    scores_path,
)
from dreamscapes_pipeline.spend import SpendGuard
from tests.fakes import (  # noqa: F401
    FakeAnthropicClient,
    message,
    offline,
    score_entry,
    write_candidates,
    write_style,
)


def _generate(paths, scene_id, count=3):
    write_style(paths)
    manifest = load_manifest(paths.scenes_yaml)
    run_generate(
        paths,
        manifest,
        [scene_id],
        FakeImageProvider(image_size=(320, 180)),
        count=count,
        guard=SpendGuard(),
        echo=lambda _: None,
    )


def _select(paths, scene_ids, client, lines=None, **kwargs):
    manifest = load_manifest(paths.scenes_yaml)
    judge = ClaudeJudge(client=client, root=paths.root)
    kwargs.setdefault("guard", SpendGuard(limit=15))
    return run_select(
        paths,
        manifest,
        scene_ids,
        judge=judge,
        echo=(lines.append if lines is not None else lambda _: None),
        sleep=lambda _: None,
        **kwargs,
    )


def _labels(kwargs):
    return [b["text"] for b in kwargs["messages"][0]["content"] if b["type"] == "text"][:-1]


def test_request_follows_the_claude_api_defaults(small_repo):
    write_style(small_repo)
    write_candidates(small_repo, "second-scene", 2, size=(2048, 1152))
    manifest = load_manifest(small_repo.scenes_yaml)
    job = build_job(small_repo, manifest, manifest.scene("second-scene"), "STYLE")
    judge = ClaudeJudge(root=small_repo.root)
    request = judge.request(job, describe=False)

    assert request["model"] == JUDGE_MODEL == "claude-opus-5-5"
    assert "thinking" not in request, "adaptive thinking is the default; omit the parameter"
    assert request["output_config"]["effort"] == "high"
    schema = request["output_config"]["format"]["schema"]
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert schema["additionalProperties"] is False and set(schema["required"]) == {
        "candidates",
        "advice",
    }
    assert request["betas"] == [FALLBACK_BETA] == ["server-side-fallback-2026-07-01"]
    assert request["fallbacks"] == "default"
    assert request["max_tokens"] == 16_000

    content = request["messages"][0]["content"]
    assert _labels(request) == ["Style anchor:", "Candidate 1:", "Candidate 2:"]
    image = content[1]
    assert image["source"]["media_type"] == "image/jpeg"
    with Image.open(io.BytesIO(base64.b64decode(image["source"]["data"]))) as decoded:
        assert decoded.size == (1536, 864)
    assert "paper boat" in content[-1]["text"] and "STYLE" in content[-1]["text"]

    described = judge.request(job, describe=True)["messages"][0]["content"][1]
    assert described["source"]["data"].startswith("<1536x864 JPEG of assets/masters/first-scene")
    assert 0.02 < judge.estimate_cost(job) < 0.5


def test_the_anchor_scene_is_judged_against_the_style_block(small_repo):
    write_candidates(small_repo, "first-scene", 2)
    manifest = load_manifest(small_repo.scenes_yaml)
    job = build_job(small_repo, manifest, manifest.scene("first-scene"), "STYLE")
    assert job.anchor is None
    request = ClaudeJudge().request(job)
    assert _labels(request) == ["Candidate 1:", "Candidate 2:"]
    assert "no anchor image yet" in request["messages"][0]["content"][-1]["text"]


def test_a_placeholder_anchor_is_not_shown(placeholder_repo):
    write_candidates(placeholder_repo, "prison-yard", 1)
    manifest = load_manifest(placeholder_repo.scenes_yaml)
    job = build_job(placeholder_repo, manifest, manifest.scene("prison-yard"), "STYLE")
    assert job.anchor is None


def test_select_scores_records_and_copies_the_winner(small_repo):
    _generate(small_repo, "second-scene")
    client = FakeAnthropicClient(best=2)
    lines: list[str] = []
    assert _select(small_repo, ["second-scene"], client, lines) == 0
    assert len(client.calls) == 1

    scores = json.loads(scores_path(small_repo, "second-scene").read_text())
    assert scores["winner"] == 2 and scores["model"] == "claude-opus-5-5"
    assert scores["weights"] == WEIGHTS
    assert scores["anchor"]["path"] == "assets/masters/first-scene.png"
    best = next(c for c in scores["candidates"] if c["candidate"] == 2)
    assert best["total"] == 5.0 and best["passed"] and best["fail_reasons"] == []
    assert best["sha256"] == sha256_file(small_repo.candidate_dir("second-scene") / "cand-2.png")

    rows = genlog.read_rows(small_repo.generation_log)
    score_row = rows[-1]
    assert score_row["kind"] == "score" and score_row["status"] == "ok"
    assert score_row["served_model"] == "claude-opus-5-5"
    assert score_row["cost_usd"] == pytest.approx((9000 * 4 + 2000 * 20) / 1e6)
    assert [i["label"] for i in score_row["images"]][0] == "Style anchor"
    image_row = genlog.find_image_row(
        rows,
        "second-scene",
        2,
        sha256_file(small_repo.candidate_dir("second-scene") / "cand-2.png"),
    )

    generation = load_manifest(small_repo.scenes_yaml).scene("second-scene").generation
    assert generation == Generation(
        provider="fake",
        model="fake-image-1",
        selected_candidate=2,
        log_ids=[image_row["id"], score_row["id"]],
    )
    master = small_repo.master("second-scene")
    with Image.open(master) as image:
        assert image.size == (MASTER_WIDTH, MASTER_HEIGHT)
    assert master.read_bytes() == master_png_from(
        small_repo.candidate_dir("second-scene") / "cand-2.png"
    )
    sheet = (small_repo.review_dir("second-scene") / "contact-sheet.html").read_text()
    assert "selected" in sheet and "cand-3" in sheet
    assert any("selected cand-2" in line for line in lines)


def test_a_pinned_choice_is_kept_until_repick(small_repo):
    _generate(small_repo, "second-scene")
    client = FakeAnthropicClient(best=2)
    _select(small_repo, ["second-scene"], client)
    master = small_repo.master("second-scene").read_bytes()

    lines: list[str] = []
    assert _select(small_repo, ["second-scene"], client, lines) == 0
    assert len(client.calls) == 1, "a pinned scene is not re-scored"
    assert any("pinned to cand-2" in line for line in lines)
    assert small_repo.master("second-scene").read_bytes() == master

    rescoring = FakeAnthropicClient(best=3)
    assert _select(small_repo, ["second-scene"], rescoring, repick=True) == 0
    assert len(rescoring.calls) == 1
    generation = load_manifest(small_repo.scenes_yaml).scene("second-scene").generation
    assert generation.selected_candidate == 3


def test_a_person_can_pin_a_candidate_by_editing_the_yaml(small_repo):
    _generate(small_repo, "second-scene")
    manifest = load_manifest(small_repo.scenes_yaml)
    manifest.scene("second-scene").generation = Generation(selected_candidate=3)
    save_manifest(manifest, small_repo.scenes_yaml)
    client = FakeAnthropicClient(best=1)
    assert _select(small_repo, ["second-scene"], client) == 0
    assert client.calls == []
    generation = load_manifest(small_repo.scenes_yaml).scene("second-scene").generation
    assert (generation.selected_candidate, generation.provider) == (3, "fake")

    manifest = load_manifest(small_repo.scenes_yaml)
    manifest.scene("second-scene").generation = Generation(selected_candidate=9)
    save_manifest(manifest, small_repo.scenes_yaml)
    lines: list[str] = []
    assert _select(small_repo, ["second-scene"], client, lines) == 1
    assert any("pinned candidate 9 does not exist" in line for line in lines)


def test_current_scores_are_reused_instead_of_paying_again(small_repo):
    _generate(small_repo, "second-scene")
    client = FakeAnthropicClient(best=2)
    _select(small_repo, ["second-scene"], client)
    manifest = load_manifest(small_repo.scenes_yaml)
    manifest.scene("second-scene").generation = None
    save_manifest(manifest, small_repo.scenes_yaml)
    lines: list[str] = []
    assert _select(small_repo, ["second-scene"], client, lines) == 0
    assert len(client.calls) == 1
    assert any("reusing scores.json" in line for line in lines)

    write_candidates(small_repo, "second-scene", 4)  # new files: the cache no longer applies
    manifest = load_manifest(small_repo.scenes_yaml)
    manifest.scene("second-scene").generation = None
    save_manifest(manifest, small_repo.scenes_yaml)
    assert _select(small_repo, ["second-scene"], client) == 0
    assert len(client.calls) == 2


def test_hard_fails_cannot_win(small_repo):
    _generate(small_repo, "second-scene")
    client = FakeAnthropicClient(
        best=1,
        overrides={1: {"has_people_or_faces": True}, 2: {"bridge_object_present": False}},
    )
    assert _select(small_repo, ["second-scene"], client) == 0
    scores = json.loads(scores_path(small_repo, "second-scene").read_text())
    assert scores["winner"] == 3
    verdicts = {c["candidate"]: c["fail_reasons"] for c in scores["candidates"]}
    assert verdicts == {1: ["people or faces"], 2: ["bridge object missing"], 3: []}


def test_when_nothing_passes_no_master_or_yaml_changes(small_repo):
    _generate(small_repo, "second-scene", count=2)
    yaml_before = small_repo.scenes_yaml.read_text()
    master_before = small_repo.master("second-scene").read_bytes()

    def respond(kwargs):
        entries = [score_entry(1, has_text_or_logos=True), score_entry(2, has_text_or_logos=True)]
        return message(json.dumps({"candidates": entries, "advice": "Move the boat left."}))

    lines: list[str] = []
    assert _select(small_repo, ["second-scene"], FakeAnthropicClient(respond=respond), lines) == 1
    assert small_repo.scenes_yaml.read_text() == yaml_before
    assert small_repo.master("second-scene").read_bytes() == master_before
    assert json.loads(scores_path(small_repo, "second-scene").read_text())["winner"] is None
    assert any("Move the boat left." in line for line in lines)


@pytest.mark.parametrize(
    ("reply", "problem"),
    [
        (message("", stop_reason="refusal", stop_details=SimpleNamespace(category="bio")), "bio"),
        (message('{"candidates": [', stop_reason="max_tokens"), "ran out of output tokens"),
        (message("not json"), "did not validate"),
        (
            message(json.dumps({"candidates": [score_entry(1)], "advice": ""})),
            "expected exactly",
        ),
        (
            message(json.dumps({"candidates": [score_entry(1, style_match=9)], "advice": ""})),
            "did not validate",
        ),
    ],
)
def test_unusable_answers_are_errors(small_repo, reply, problem):
    write_candidates(small_repo, "first-scene", 2)
    manifest = load_manifest(small_repo.scenes_yaml)
    job = build_job(small_repo, manifest, manifest.scene("first-scene"), "STYLE")
    judge = ClaudeJudge(client=FakeAnthropicClient(respond=lambda _: reply))
    with pytest.raises(ScoringError, match=problem) as caught:
        judge.score(job)
    assert not caught.value.retryable


def test_a_failed_scoring_call_is_logged(small_repo):
    _generate(small_repo, "second-scene", count=1)
    refusal = message("", stop_reason="refusal")
    assert (
        _select(small_repo, ["second-scene"], FakeAnthropicClient(respond=lambda _: refusal)) == 1
    )
    row = genlog.read_rows(small_repo.generation_log)[-1]
    assert row["kind"] == "score" and row["status"] == "error" and "declined" in row["error"]


def test_a_fallback_answer_is_read_after_the_fallback_block(small_repo):
    write_candidates(small_repo, "first-scene", 1)
    manifest = load_manifest(small_repo.scenes_yaml)
    job = build_job(small_repo, manifest, manifest.scene("first-scene"), "STYLE")
    text = json.dumps({"candidates": [score_entry(1)], "advice": ""})
    fallback = SimpleNamespace(type="fallback")
    partial = SimpleNamespace(type="text", text='{"cand')
    reply = message(text, model="claude-opus-5", blocks=[partial, fallback])
    result = ClaudeJudge(client=FakeAnthropicClient(respond=lambda _: reply)).score(job)
    assert result.served_model == "claude-opus-5" and result.model == "claude-opus-5-5"
    assert result.cost_usd == pytest.approx((9000 * 5 + 2000 * 25) / 1e6)


def test_rate_limits_are_retried(small_repo):
    _generate(small_repo, "second-scene", count=1)
    request = httpx2.Request("POST", "https://api.anthropic.invalid/v1/messages")
    limited = anthropic.RateLimitError(
        "slow down",
        response=httpx2.Response(429, request=request, headers={"retry-after": "3"}),
        body=None,
    )
    good = FakeAnthropicClient()
    attempts = []

    def respond(kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise limited
        return good._create(**kwargs)

    assert _select(small_repo, ["second-scene"], FakeAnthropicClient(respond=respond)) == 0
    assert len(attempts) == 2


def test_dry_run_and_spend_guard_send_nothing(small_repo):
    _generate(small_repo, "second-scene", count=2)
    client = FakeAnthropicClient()
    lines: list[str] = []
    assert _select(small_repo, ["second-scene"], client, lines, dry_run=True) == 0
    assert client.calls == [] and not scores_path(small_repo, "second-scene").exists()
    assert any('"fallbacks": "default"' in line for line in lines)

    lines.clear()
    assert _select(small_repo, ["second-scene"], client, lines, guard=SpendGuard(limit=0.001)) == 1
    assert client.calls == []
    assert any("refusing to start" in line for line in lines)


def test_missing_candidates_and_missing_key(small_repo):
    write_style(small_repo)
    lines: list[str] = []
    assert _select(small_repo, ["second-scene"], FakeAnthropicClient(), lines) == 1
    assert any("no candidates" in line for line in lines)

    write_candidates(small_repo, "second-scene", 1)
    manifest = load_manifest(small_repo.scenes_yaml)
    lines.clear()
    code = run_select(
        small_repo,
        manifest,
        ["second-scene"],
        judge=ClaudeJudge(),
        guard=SpendGuard(),
        echo=lines.append,
        sleep=lambda _: None,
    )
    assert code == 1 and any("ANTHROPIC_API_KEY is not set" in line for line in lines)

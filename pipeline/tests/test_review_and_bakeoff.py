from __future__ import annotations

import json

from dreamscapes_pipeline import genlog
from dreamscapes_pipeline.bakeoff import bakeoff_candidate, run_bakeoff
from dreamscapes_pipeline.manifest import load_manifest
from dreamscapes_pipeline.placeholders import PLACEHOLDER_SCENES
from dreamscapes_pipeline.providers import FakeImageProvider
from dreamscapes_pipeline.review import loop_order, run_review
from dreamscapes_pipeline.select import ClaudeJudge, run_select
from dreamscapes_pipeline.spend import SpendGuard
from tests.conftest import PLACEHOLDER_SCENE_COUNT
from tests.fakes import FakeAnthropicClient, offline, write_candidates, write_style  # noqa: F401


def test_loop_order_follows_next_from_the_start(placeholder_repo):
    manifest = load_manifest(placeholder_repo.scenes_yaml)
    order = loop_order(manifest)
    # The placeholder table lists the scenes in loop order.
    assert order[0] == manifest.start and order[-1] == PLACEHOLDER_SCENES[-1].id
    assert len(order) == len(set(order)) == PLACEHOLDER_SCENE_COUNT


def test_index_shows_every_master_side_by_side(placeholder_repo):
    report = run_review(placeholder_repo, load_manifest(placeholder_repo.scenes_yaml))
    html = report.index.read_text()
    assert report.contact_sheets == []
    assert html.count("<figure>") == PLACEHOLDER_SCENE_COUNT
    assert html.index("cliff-cottage") < html.index("prison-yard") < html.index("rooftop-court")
    assert "placeholder" in html
    assert (placeholder_repo.review / "thumbs" / "drowned-street.webp").exists()


def test_review_rebuilds_contact_sheets_and_links_overlays(small_repo):
    write_style(small_repo)
    write_candidates(small_repo, "second-scene", 2)
    manifest = load_manifest(small_repo.scenes_yaml)
    run_select(
        small_repo,
        manifest,
        ["second-scene"],
        judge=ClaudeJudge(client=FakeAnthropicClient(best=2), root=small_repo.root),
        guard=SpendGuard(),
        echo=lambda _: None,
    )
    write_candidates(small_repo, "third-scene", 1)
    overlay = small_repo.review_dir("second-scene") / "hotspot.png"
    overlay.write_bytes(b"png")

    report = run_review(small_repo, load_manifest(small_repo.scenes_yaml))
    assert sorted(p.parent.name for p in report.contact_sheets) == ["second-scene", "third-scene"]
    index = report.index.read_text()
    assert "second-scene/contact-sheet.html" in index and "second-scene/hotspot.png" in index
    assert ">5.00<" in index, "the selected candidate's score is listed"
    sheet = (small_repo.review_dir("second-scene") / "contact-sheet.html").read_text()
    assert "../../candidates/second-scene/cand-2.png" in sheet
    assert "hotspot overlay" in sheet and "selected" in sheet
    unscored = (small_repo.review_dir("third-scene") / "contact-sheet.html").read_text()
    assert "not scored yet" in unscored


def _bakeoff(paths, client, lines=None, **kwargs):
    providers = [
        FakeImageProvider(name="gemini", model="fake-gemini", image_size=(400, 225)),
        FakeImageProvider(name="openai", model="fake-openai", image_size=(512, 288)),
    ]
    code = run_bakeoff(
        paths,
        load_manifest(paths.scenes_yaml),
        providers,
        count=2,
        judge=ClaudeJudge(client=client, root=paths.root),
        guard=SpendGuard(),
        echo=(lines.append if lines is not None else lambda _: None),
        sleep=lambda _: None,
        **kwargs,
    )
    return code, providers


def test_bakeoff_generates_grades_and_scores_blind(small_repo):
    write_style(small_repo)
    client = FakeAnthropicClient(best=3)
    code, providers = _bakeoff(small_repo, client)
    assert code == 0
    assert [len(p.calls) for p in providers] == [2, 2]
    prompts = {c.prompt for p in providers for c in p.calls}
    assert len(prompts) == 1, "every provider gets the identical prompt"
    assert all(not c.references for p in providers for c in p.calls)
    assert bakeoff_candidate(small_repo, "openai", 2).exists()

    assert len(client.calls) == 1
    request_text = json.dumps(client.calls[0]["messages"][0]["content"][-1])
    assert "gemini" not in request_text and "openai" not in request_text
    labels = [b["text"] for b in client.calls[0]["messages"][0]["content"] if b["type"] == "text"][
        :-1
    ]
    assert labels == ["Candidate 1:", "Candidate 2:", "Candidate 3:", "Candidate 4:"]

    scores = json.loads((small_repo.review / "bakeoff" / "scores.json").read_text())
    assert {c["provider"] for c in scores["candidates"]} == {"gemini", "openai"}
    winner = next(c for c in scores["candidates"] if c["label"] == 3)
    assert winner["total"] == 5.0
    for entry in scores["candidates"]:
        assert (small_repo.review / entry["graded"]).exists()
    page = (small_repo.review / "bakeoff.html").read_text()
    assert "Mean score" in page and "fake-openai" in page

    rows = genlog.read_rows(small_repo.generation_log)
    assert [r["purpose"] for r in rows] == ["bakeoff"] * 5
    assert rows[-1]["kind"] == "score"


def test_bakeoff_reuses_images_and_scores(small_repo):
    write_style(small_repo)
    client = FakeAnthropicClient()
    _bakeoff(small_repo, client)
    lines: list[str] = []
    code, providers = _bakeoff(small_repo, client, lines)
    assert code == 0
    assert [len(p.calls) for p in providers] == [0, 0]
    assert len(client.calls) == 1
    assert "reusing bake-off scores (images unchanged)" in lines


def test_bakeoff_dry_run_sends_nothing(small_repo):
    client = FakeAnthropicClient()
    lines: list[str] = []
    code, providers = _bakeoff(small_repo, client, lines, dry_run=True)
    assert code == 0 and client.calls == [] and all(p.attempts == 0 for p in providers)
    assert not small_repo.candidates.exists()
    assert any("one Claude scoring call over 4" in line for line in lines)

"""`dsp bakeoff`: generate the first scene with each image provider and compare them (plan 5.2).

The style-anchor scene (scene 1, which has no reference image) is generated ``--candidates``
times by every provider with the identical prompt. Each image is fitted to the master size and
graded exactly as a master would be, then Claude scores all of them in one blind request: the
images are numbered in an order derived from their bytes, so the judge cannot tell which
provider made which. The comparison is written to ``assets/review/bakeoff.html`` and the scores
to ``assets/review/bakeoff/scores.json``.

Images live in ``assets/candidates/_bakeoff/<provider>/cand-<n>.png`` (the leading underscore
keeps the folder from ever colliding with a scene id) and are reused on later runs unless
``--force``; scores are reused while the graded images are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from . import genlog
from .generate import (
    GenerationPlan,
    PlannedImage,
    StyleBlockError,
    build_prompt,
    execute_plan,
    load_style_block,
    print_requests,
    seed_for,
)
from .grade import GradeConfigError, grade_image, load_grade_params
from .imaging import fit_master, png_bytes, sha256_file, write_if_changed
from .layout import RepoPaths
from .manifest import Manifest
from .providers import ASPECT, ImageProvider, MissingCredentials, ProviderError
from .select import (
    WEIGHTS,
    ClaudeJudge,
    ScoringJob,
    fail_reasons,
    load_scores,
    make_judge,
    score_with_log,
    weighted_total,
)
from .spend import SpendGuard, SpendLimitExceeded

Echo = Callable[[str], None]

BAKEOFF_FOLDER = "_bakeoff"


def bakeoff_candidate(paths: RepoPaths, provider: str, number: int) -> Path:
    return paths.candidates / BAKEOFF_FOLDER / provider / f"cand-{number}.png"


def bakeoff_review_dir(paths: RepoPaths) -> Path:
    return paths.review / "bakeoff"


def graded_copy(paths: RepoPaths, provider: str, number: int) -> Path:
    return bakeoff_review_dir(paths) / f"{provider}-cand-{number}.graded.png"


def _blind_order(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(entries, key=lambda e: hashlib.sha256(e["sha256"].encode()).hexdigest())


def run_bakeoff(
    paths: RepoPaths,
    manifest: Manifest,
    providers: list[ImageProvider],
    *,
    count: int = 4,
    force: bool = False,
    dry_run: bool = False,
    judge: ClaudeJudge | None = None,
    guard: SpendGuard | None = None,
    echo: Echo = print,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    scene = manifest.scene(manifest.style_anchor)
    try:
        style = load_style_block(paths, allow_draft=dry_run)
        loaded = load_grade_params(paths)
    except (StyleBlockError, GradeConfigError) as error:
        echo(f"error: {error}")
        return 1
    prompt = build_prompt(style.text, scene, with_reference=False)
    rows = genlog.read_rows(paths.generation_log)
    judge = judge or make_judge(paths.root)
    guard = guard or SpendGuard.from_env()

    plans: list[tuple[ImageProvider, GenerationPlan]] = []
    for provider in providers:
        size = provider.default_size()
        plan = GenerationPlan()
        for number in range(1, count + 1):
            out = bakeoff_candidate(paths, provider.name, number)
            if out.exists() and not force:
                plan.skipped.append(f"{provider.name} cand-{number}")
                continue
            attempt = genlog.image_attempts(rows, scene.id, number, provider.name, "bakeoff")
            seed = seed_for(scene.id, number, attempt)
            plan.items.append(
                PlannedImage(
                    scene_id=scene.id,
                    candidate=number,
                    out_path=out,
                    prompt=prompt,
                    references=[],
                    seed=seed,
                    request=provider.request(prompt, ASPECT, size, [], seed),
                    estimate=provider.estimate_cost(size, 0),
                    purpose="bakeoff",
                )
            )
        plans.append((provider, plan))

    total_images = count * len(providers)
    scoring_estimate = judge_estimate(judge, total_images)
    generation_estimate = sum(plan.estimate for _, plan in plans)
    description = (
        f"{sum(len(p.items) for _, p in plans)} image(s) from "
        f"{', '.join(p.name for p in providers)} and one scoring call"
    )
    echo(f"bake-off on '{scene.id}' with the style block from {style.source}")
    for _, plan in plans:
        if plan.skipped:
            echo(f"already generated, skipped (use --force to redo): {', '.join(plan.skipped)}")

    if dry_run:
        for _provider, plan in plans:
            print_requests(plan, paths, echo)
        echo(
            f"then one Claude scoring call over {total_images} graded image(s) "
            f"(estimated ${scoring_estimate:.3f})"
        )
        try:
            echo(guard.check_plan(generation_estimate + scoring_estimate, description))
        except SpendLimitExceeded as error:
            echo(f"note: a real run would refuse: {error}")
        echo("dry run: nothing was sent")
        return 0

    try:
        echo(guard.check_plan(generation_estimate + scoring_estimate, description))
        failed: list[str] = []
        for provider, plan in plans:
            outcome = execute_plan(
                paths,
                provider,
                plan,
                size=provider.default_size(),
                style=style,
                guard=guard,
                echo=echo,
                sleep=sleep,
            )
            failed.extend(outcome.failed)
    except (SpendLimitExceeded, MissingCredentials) as error:
        echo(f"error: {error}")
        return 1

    # Grade every image exactly as a master would be graded.
    rows = genlog.read_rows(paths.generation_log)
    entries: list[dict[str, Any]] = []
    for provider in providers:
        for number in range(1, count + 1):
            raw = bakeoff_candidate(paths, provider.name, number)
            if not raw.exists():
                continue
            with Image.open(raw) as opened:
                graded = grade_image(fit_master(opened), loaded.params, scene.id)
            target = graded_copy(paths, provider.name, number)
            write_if_changed(target, png_bytes(graded))
            source_sha = sha256_file(raw)
            row = next(
                (
                    r
                    for r in reversed(rows)
                    if r.get("kind") == "image"
                    and r.get("purpose") == "bakeoff"
                    and r.get("provider") == provider.name
                    and r.get("candidate") == number
                    and r.get("output_sha256") == source_sha
                ),
                None,
            )
            entries.append(
                {
                    "provider": provider.name,
                    "model": (row or {}).get("model", provider.model),
                    "candidate": number,
                    "path": raw.relative_to(paths.root).as_posix(),
                    "graded": target.relative_to(paths.review).as_posix(),
                    "sha256": sha256_file(target),
                    "cost_usd": (row or {}).get("cost_usd") or (row or {}).get("cost_estimate_usd"),
                    "log_id": (row or {}).get("id"),
                }
            )
    if not entries:
        echo("error: no bake-off images exist; nothing to compare")
        return 1

    blind = _blind_order(entries)
    for label, entry in enumerate(blind, start=1):
        entry["label"] = label
    scores_file = bakeoff_review_dir(paths) / "scores.json"
    previous = load_scores(scores_file) or {}
    reusable = previous.get("prompt") == prompt and sorted(
        (c["provider"], c["candidate"], c["sha256"]) for c in previous.get("candidates", [])
    ) == sorted((e["provider"], e["candidate"], e["sha256"]) for e in entries)

    document: dict[str, Any] = {
        "scene": scene.id,
        "prompt": prompt,
        "style_block_sha256": style.sha256,
        "judge_model": None,
        "log_id": None,
        "candidates": entries,
    }
    if reusable and all("total" in c for c in previous["candidates"]):
        echo("reusing bake-off scores (images unchanged)")
        document = previous
    else:
        job = ScoringJob(
            scene_id=scene.id,
            scene_text=scene.scene,
            bridge_object=scene.bridge.object,
            placement=scene.bridge.placement,
            style_block=style.text,
            candidates=[(e["label"], paths.review / e["graded"]) for e in blind],
            anchor=None,
            purpose="bakeoff",
        )
        try:
            result, log_id = score_with_log(paths, judge, job, guard, echo=echo, sleep=sleep)
        except (ProviderError, SpendLimitExceeded) as error:
            echo(f"scoring failed: {error}; writing the sheet without scores")
            failed.append("scoring")
        else:
            by_label = {s.candidate: s for s in result.sheet.candidates}
            for entry in entries:
                score = by_label[entry["label"]]
                reasons = fail_reasons(score)
                entry.update(
                    scores={name: getattr(score, name) for name in WEIGHTS},
                    total=weighted_total(score),
                    passed=not reasons,
                    fail_reasons=reasons,
                    notes=score.notes,
                )
            document.update(judge_model=result.served_model, log_id=log_id)

    write_if_changed(scores_file, (json.dumps(document, indent=2) + "\n").encode("utf-8"))
    from .review import write_bakeoff_page

    page = write_bakeoff_page(paths, document)
    echo(f"wrote {page.relative_to(paths.root).as_posix()}")
    for entry in document["candidates"]:
        if "total" in entry:
            verdict = "pass" if entry["passed"] else "fail: " + ", ".join(entry["fail_reasons"])
            echo(
                f"  {entry['provider']} cand-{entry['candidate']}: {entry['total']:.2f} ({verdict})"
            )
    return 1 if failed else 0


def judge_estimate(judge: ClaudeJudge, images: int) -> float:
    """Scoring cost before the images exist, assuming judge-sized 16:9 previews."""
    from .select import (
        EXPECTED_OUTPUT_TOKENS,
        JUDGE_IMAGE_WIDTH,
        OUTPUT_TOKENS_PER_CANDIDATE,
        PRICES,
    )

    per_image = JUDGE_IMAGE_WIDTH * round(JUDGE_IMAGE_WIDTH * 9 / 16) / 750
    tokens_in = images * per_image + 1_500
    tokens_out = EXPECTED_OUTPUT_TOKENS + OUTPUT_TOKENS_PER_CANDIDATE * images
    price_in, price_out = PRICES.get(judge.model, PRICES["claude-opus-5"])
    return round((tokens_in * price_in + tokens_out * price_out) / 1e6, 4)

"""`dsp select`: Claude scores a scene's candidates and the best one becomes the master.

Scoring follows plan section 5.4. All of a scene's candidates go to Claude in one request
(downscaled to JPEG), together with the ungraded style anchor master for every scene except the
anchor itself. Each candidate gets 0-5 on five criteria; the weighted sum ranks them, and a
candidate with legible text or logos, people or faces, or no bridge object fails outright.

Results go to ``assets/review/<scene>/scores.json`` and a contact sheet next to it. Unless the
scene's ``generation.selected_candidate`` is already set in ``content/scenes.yaml`` (a person or
an earlier run pinned it; ``--repick`` overrides), the winner is recorded there along with the
provider, model and generation-log rows, and copied to ``assets/masters/<scene>.png`` at the
master size. A pinned scene is not re-scored: its pinned candidate is copied instead.

Scores are cached: when ``scores.json`` already covers exactly the current candidate files (and
the same anchor), it is reused instead of paying for a new call. ``--repick`` always re-scores.

Claude API usage follows the bundled ``claude-api`` guidance: model ``claude-opus-5-5``,
adaptive thinking (the model's default, so the ``thinking`` parameter is omitted), an explicit
``output_config.effort``, structured output through ``output_config.format`` validated with
pydantic, and the server-side ``fallbacks: "default"`` option (beta
``server-side-fallback-2026-07-01``), which re-runs a request that a safety classifier declined
on the model Anthropic recommends for that case.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import genlog
from .generate import StyleBlockError, existing_candidates, load_style_block
from .imaging import jpeg_preview, master_png_from, sha256_file, write_if_changed
from .layout import RepoPaths
from .manifest import Generation, Manifest, Scene, dump_manifest, save_manifest
from .placeholders import is_placeholder_image
from .providers.base import ProviderError, is_retryable_status, retry_after_seconds
from .retry import with_retries
from .spend import SpendGuard, SpendLimitExceeded

Echo = Callable[[str], None]

JUDGE_MODEL = "claude-opus-5-5"
JUDGE_EFFORT = "high"
MAX_TOKENS = 16_000
FALLBACK_BETA = "server-side-fallback-2026-07-01"
JUDGE_IMAGE_WIDTH = 1536

# US dollars per million tokens (input, output). A fallback model is priced at its own rates.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
}
EXPECTED_OUTPUT_TOKENS = 1_500
OUTPUT_TOKENS_PER_CANDIDATE = 600

WEIGHTS = {
    "style_match": 0.25,
    "bridge_object": 0.25,
    "cleanliness": 0.20,
    "mood": 0.15,
    "composition": 0.15,
}


# --------------------------------------------------------------------------------------------
# Structured output
# --------------------------------------------------------------------------------------------


class CandidateScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: int = Field(description="The candidate number shown above its image.")
    style_match: int = Field(ge=0, le=5, description="0-5: match to the style reference.")
    bridge_object: int = Field(ge=0, le=5, description="0-5: the bridge object as specified.")
    cleanliness: int = Field(
        ge=0, le=5, description="0-5: free of text, logos, people, faces and artefacts."
    )
    mood: int = Field(ge=0, le=5, description="0-5: empty, quiet, dreamlike.")
    composition: int = Field(ge=0, le=5, description="0-5: strength as a landing image.")
    bridge_object_present: bool = Field(description="The bridge object is clearly in the image.")
    has_text_or_logos: bool = Field(
        description="Any legible lettering, sign, logo or watermark appears."
    )
    has_people_or_faces: bool = Field(
        description="Any person, figure, silhouette, statue or face appears."
    )
    notes: str = Field(description="One or two sentences: best, worst, where the object is.")


class ScoreSheet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[CandidateScore]
    advice: str = Field(
        description="If no candidate is acceptable, how to change the placement hint or "
        "description; otherwise an empty string."
    )


def weighted_total(score: CandidateScore) -> float:
    return round(sum(getattr(score, name) * weight for name, weight in WEIGHTS.items()), 4)


def fail_reasons(score: CandidateScore) -> list[str]:
    reasons = []
    if not score.bridge_object_present:
        reasons.append("bridge object missing")
    if score.has_text_or_logos:
        reasons.append("text or logos")
    if score.has_people_or_faces:
        reasons.append("people or faces")
    return reasons


def pick_winner(sheet: ScoreSheet) -> int | None:
    """Highest weighted total among candidates without a hard fail; ties go to the lower number."""
    passing = [s for s in sheet.candidates if not fail_reasons(s)]
    if not passing:
        return None
    return max(passing, key=lambda s: (weighted_total(s), -s.candidate)).candidate


# --------------------------------------------------------------------------------------------
# The judge
# --------------------------------------------------------------------------------------------


SYSTEM_PROMPT = (
    "You are the art director of Dreamscapes, a full-screen web page that shows one dreamlike, "
    "painted scene at a time. In every scene one ordinary object, the bridge object, is "
    "secretly clickable and leads to the next scene, so it must be present and findable but "
    "never the obvious subject. You score candidate images for one scene against a fixed "
    "rubric. Your scores choose the scene's image without a person checking first, so be "
    "strict, consistent across candidates, and literal about what is actually visible."
)


@dataclass(frozen=True)
class JudgeImage:
    label: str
    path: Path
    sha256: str
    jpeg: bytes
    size: tuple[int, int]


@dataclass
class ScoringJob:
    scene_id: str
    scene_text: str
    bridge_object: str
    placement: str
    style_block: str
    candidates: list[tuple[int, Path]]
    anchor: Path | None = None
    purpose: str = "select"
    _images: list[JudgeImage] | None = None

    def images(self) -> list[JudgeImage]:
        """The anchor (if any) and every candidate, downscaled once and cached."""
        if self._images is None:
            from PIL import Image

            images = []
            entries = [("Style anchor", self.anchor)] if self.anchor else []
            entries += [(f"Candidate {n}", path) for n, path in self.candidates]
            for label, path in entries:
                assert path is not None
                jpeg = jpeg_preview(path, JUDGE_IMAGE_WIDTH)
                with Image.open(io.BytesIO(jpeg)) as preview:
                    size = preview.size
                images.append(JudgeImage(label, path, sha256_file(path), jpeg, size))
            self._images = images
        return self._images

    def instructions(self) -> str:
        reference = (
            "the style anchor shown first (palette, lighting, grain and rendering only)"
            if self.anchor
            else "the style block below (there is no anchor image yet for this scene)"
        )
        numbers = ", ".join(str(n) for n, _ in self.candidates)
        obj = self.bridge_object
        return (
            f"Scene id: {self.scene_id}\n"
            f"Scene description given to the generator: {self.scene_text.strip()}\n"
            f"Bridge object: {obj}. Requested placement: {self.placement.strip()}.\n"
            f"Style block given to the generator: {self.style_block.strip()}\n\n"
            "Score every candidate from 0 to 5 on each criterion:\n"
            f"- style_match: how well it matches {reference}. 5 means it would sit in the same "
            "set without anyone noticing a different hand.\n"
            f"- bridge_object: the {obj} is present and clearly recognisable, small (roughly "
            "3-8% of the frame, never the main subject), inside the central 60% of the width, "
            "and nothing else in the scene could be mistaken for it. 0 means it is missing.\n"
            "- cleanliness: no legible text, letters, signs, logos, watermarks, people, faces, "
            "figures or obvious generation artefacts (melted or duplicated objects, broken "
            "perspective).\n"
            "- mood: empty, quiet and dreamlike; melancholic but inviting.\n"
            "- composition: a strong full-screen landing image, readable at a glance, that still "
            "works when a phone crops it around the bridge object.\n"
            "Also answer bridge_object_present, has_text_or_logos and has_people_or_faces "
            "literally. In notes, say in one or two sentences what is best and worst about the "
            f"candidate and where the {obj} is. In advice, if no candidate is acceptable, say how "
            "the placement hint or description should change; otherwise leave it empty.\n"
            f"Return exactly one entry for each of candidates {numbers}, using those numbers."
        )


def _schema() -> dict[str, Any]:
    from anthropic import transform_schema

    return transform_schema(ScoreSheet)


@dataclass
class ScoringResult:
    sheet: ScoreSheet
    model: str
    served_model: str
    request_id: str | None
    response_id: str | None
    usage: dict[str, Any] | None
    cost_estimate_usd: float
    cost_usd: float | None


class ScoringError(ProviderError):
    """Claude's answer could not be used (refusal, truncation or an invalid score sheet)."""


class ClaudeJudge:
    """Scores candidates with Claude. ``client`` may be injected; tests pass a fake."""

    provider = "anthropic"

    def __init__(
        self,
        model: str = JUDGE_MODEL,
        effort: str = JUDGE_EFFORT,
        client: Any = None,
        root: Path | None = None,
    ):
        self.model = model
        self.effort = effort
        self._client = client
        self._root = root

    def _content(self, job: ScoringJob, *, describe: bool) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for image in job.images():
            content.append({"type": "text", "text": f"{image.label}:"})
            if describe:
                shown = image.path
                if self._root is not None and image.path.is_relative_to(self._root):
                    shown = image.path.relative_to(self._root)
                data = (
                    f"<{image.size[0]}x{image.size[1]} JPEG of {shown.as_posix()}, "
                    f"{len(image.jpeg)} bytes, sent as base64>"
                )
            else:
                data = base64.b64encode(image.jpeg).decode("ascii")
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
                }
            )
        content.append({"type": "text", "text": job.instructions()})
        return content

    def request(self, job: ScoringJob, *, describe: bool = True) -> dict[str, Any]:
        """Keyword arguments for ``client.beta.messages.create``."""
        return {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": _schema()},
            },
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": self._content(job, describe=describe)}],
        }

    def estimate_cost(self, job: ScoringJob) -> float:
        image_tokens = sum(w * h / 750 for w, h in (img.size for img in job.images()))
        text_tokens = (len(SYSTEM_PROMPT) + len(job.instructions())) / 3.5
        output = EXPECTED_OUTPUT_TOKENS + OUTPUT_TOKENS_PER_CANDIDATE * len(job.candidates)
        price_in, price_out = PRICES.get(self.model, PRICES["claude-opus-5"])
        return round(((image_tokens + text_tokens) * price_in + output * price_out) / 1e6, 4)

    def _get_client(self) -> Any:
        if self._client is None:
            key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            if not key:
                from .providers.base import MissingCredentials

                raise MissingCredentials("ANTHROPIC_API_KEY is not set; add it to .env")
            import anthropic

            self._client = anthropic.Anthropic(api_key=key, max_retries=0, timeout=600.0)
        return self._client

    def score(self, job: ScoringJob) -> ScoringResult:
        kwargs = self.request(job, describe=False)
        estimate = self.estimate_cost(job)
        client = self._get_client()
        try:
            message = client.beta.messages.create(**kwargs)
        except Exception as error:
            raise _translate(error) from error
        return self._parse(message, job, estimate)

    def _parse(self, message: Any, job: ScoringJob, estimate: float) -> ScoringResult:
        stop = getattr(message, "stop_reason", None)
        if stop == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise ScoringError(f"Claude declined to score (category: {category or 'unknown'})")
        if stop == "max_tokens":
            raise ScoringError("Claude ran out of output tokens before finishing the scores")
        blocks = list(getattr(message, "content", None) or [])
        last_fallback = max(
            (i for i, b in enumerate(blocks) if getattr(b, "type", None) == "fallback"),
            default=-1,
        )
        text = "".join(
            b.text for b in blocks[last_fallback + 1 :] if getattr(b, "type", None) == "text"
        )
        try:
            sheet = ScoreSheet.model_validate_json(text)
        except ValidationError as error:
            raise ScoringError(f"Claude's score sheet did not validate: {error}") from error
        expected = sorted(n for n, _ in job.candidates)
        got = sorted(s.candidate for s in sheet.candidates)
        if got != expected:
            raise ScoringError(f"Claude scored candidates {got}, expected exactly {expected}")
        usage = _usage_dict(getattr(message, "usage", None))
        served = str(getattr(message, "model", None) or self.model)
        return ScoringResult(
            sheet=sheet,
            model=self.model,
            served_model=served,
            request_id=getattr(message, "_request_id", None),
            response_id=getattr(message, "id", None),
            usage=usage,
            cost_estimate_usd=estimate,
            cost_usd=_cost(served, usage),
        )


def _translate(error: Exception) -> ProviderError:
    """Typed SDK errors, most specific first, to `ProviderError`."""
    if isinstance(error, ProviderError):
        return error
    import anthropic

    if isinstance(error, anthropic.RateLimitError):
        return ProviderError(
            f"Claude rate limit: {error.message}",
            retryable=True,
            status_code=429,
            retry_after=retry_after_seconds(getattr(error.response, "headers", None)),
        )
    if isinstance(error, anthropic.APIStatusError):
        return ProviderError(
            f"Claude request failed with HTTP {error.status_code}: {error.message}",
            retryable=is_retryable_status(error.status_code),
            status_code=error.status_code,
            retry_after=retry_after_seconds(getattr(error.response, "headers", None)),
        )
    if isinstance(error, anthropic.APIConnectionError):
        return ProviderError(f"Claude connection problem: {error}", retryable=True)
    return ProviderError(f"Claude request failed: {error}", retryable=False)


def _usage_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json", exclude_none=True)
    return dict(usage) if isinstance(usage, dict) else None


def _cost(model: str, usage: dict[str, Any] | None) -> float | None:
    if not usage or "output_tokens" not in usage:
        return None
    price_in, price_out = PRICES.get(model, PRICES["claude-opus-5"])
    tokens_in = (
        usage.get("input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0) * 1.25
        + usage.get("cache_read_input_tokens", 0) * 0.1
    )
    return round((tokens_in * price_in + usage["output_tokens"] * price_out) / 1e6, 6)


def make_judge(root: Path | None = None) -> ClaudeJudge:
    """The judge every command uses; tests replace this function to inject a fake client."""
    return ClaudeJudge(root=root)


# --------------------------------------------------------------------------------------------
# Scores on disk
# --------------------------------------------------------------------------------------------


def scores_path(paths: RepoPaths, scene_id: str) -> Path:
    return paths.review_dir(scene_id) / "scores.json"


def scores_document(
    paths: RepoPaths, job: ScoringJob, result: ScoringResult, log_id: str
) -> dict[str, Any]:
    by_number = dict(job.candidates)
    images = {img.path: img for img in job.images()}
    entries = []
    for score in sorted(result.sheet.candidates, key=lambda s: s.candidate):
        path = by_number[score.candidate]
        reasons = fail_reasons(score)
        entries.append(
            {
                "candidate": score.candidate,
                "path": path.relative_to(paths.root).as_posix(),
                "sha256": images[path].sha256,
                "scores": {name: getattr(score, name) for name in WEIGHTS},
                "bridge_object_present": score.bridge_object_present,
                "has_text_or_logos": score.has_text_or_logos,
                "has_people_or_faces": score.has_people_or_faces,
                "notes": score.notes,
                "total": weighted_total(score),
                "passed": not reasons,
                "fail_reasons": reasons,
            }
        )
    anchor = None
    if job.anchor is not None:
        anchor = {
            "path": job.anchor.relative_to(paths.root).as_posix(),
            "sha256": images[job.anchor].sha256,
        }
    return {
        "scene": job.scene_id,
        "purpose": job.purpose,
        "model": result.model,
        "served_model": result.served_model,
        "log_id": log_id,
        "timestamp": genlog.utc_now(),
        "weights": WEIGHTS,
        "anchor": anchor,
        "candidates": entries,
        "winner": pick_winner(result.sheet),
        "advice": result.sheet.advice,
    }


def sheet_from_document(document: dict[str, Any]) -> ScoreSheet:
    return ScoreSheet(
        candidates=[
            CandidateScore(
                candidate=c["candidate"],
                **c["scores"],
                bridge_object_present=c["bridge_object_present"],
                has_text_or_logos=c["has_text_or_logos"],
                has_people_or_faces=c["has_people_or_faces"],
                notes=c["notes"],
            )
            for c in document["candidates"]
        ],
        advice=document.get("advice", ""),
    )


def load_scores(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def cached_scores(paths: RepoPaths, job: ScoringJob) -> dict[str, Any] | None:
    """The stored scores when they cover exactly the current candidates and anchor."""
    document = load_scores(scores_path(paths, job.scene_id))
    if not document:
        return None
    try:
        stored = {c["candidate"]: c["sha256"] for c in document["candidates"]}
        current = {n: sha256_file(p) for n, p in job.candidates}
        anchor = document.get("anchor")
        anchor_now = sha256_file(job.anchor) if job.anchor else None
        anchor_then = anchor["sha256"] if anchor else None
        sheet_from_document(document)
    except (KeyError, TypeError, ValidationError):
        return None
    if stored != current or anchor_now != anchor_then:
        return None
    return document


# --------------------------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------------------------


def judge_anchor(paths: RepoPaths, manifest: Manifest, scene_id: str) -> Path | None:
    """The style anchor master to show Claude, when it is real and the scene is not the anchor."""
    if scene_id == manifest.style_anchor:
        return None
    anchor = paths.master(manifest.style_anchor)
    if not anchor.exists() or is_placeholder_image(anchor):
        return None
    return anchor


def build_job(paths: RepoPaths, manifest: Manifest, scene: Scene, style_block: str) -> ScoringJob:
    return ScoringJob(
        scene_id=scene.id,
        scene_text=scene.scene,
        bridge_object=scene.bridge.object,
        placement=scene.bridge.placement,
        style_block=style_block,
        candidates=existing_candidates(paths, scene.id),
        anchor=judge_anchor(paths, manifest, scene.id),
    )


def score_with_log(
    paths: RepoPaths,
    judge: ClaudeJudge,
    job: ScoringJob,
    guard: SpendGuard,
    *,
    echo: Echo = print,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[ScoringResult, str]:
    """One scoring call with retries, the spend guard and a log row; returns the row id."""
    estimate = judge.estimate_cost(job)
    guard.before_call(estimate, f"scoring {job.scene_id}")
    request = judge.request(job, describe=True)
    log_id = genlog.new_row_id()
    row: dict[str, Any] = {
        "id": log_id,
        "timestamp": genlog.utc_now(),
        "kind": "score",
        "purpose": job.purpose,
        "provider": judge.provider,
        "model": judge.model,
        "scene": job.scene_id,
        "prompt": {"system": request["system"], "text": job.instructions()},
        "parameters": {
            k: v for k, v in request.items() if k not in ("system", "messages", "output_config")
        }
        | {"effort": judge.effort},
        "images": [
            {
                "label": img.label,
                "path": img.path.relative_to(paths.root).as_posix()
                if img.path.is_relative_to(paths.root)
                else img.path.as_posix(),
                "sha256": img.sha256,
            }
            for img in job.images()
        ],
        "cost_estimate_usd": estimate,
    }
    started = time.monotonic()
    echo(f"scoring {len(job.candidates)} candidate(s) for {job.scene_id} with {judge.model} ...")
    try:
        result = with_retries(
            lambda: judge.score(job), what=f"scoring {job.scene_id}", echo=echo, sleep=sleep
        )
    except ProviderError as error:
        row.update(
            status="error",
            error=str(error),
            cost_usd=None,
            duration_s=round(time.monotonic() - started, 2),
        )
        genlog.append_row(paths.generation_log, row)
        raise
    row.update(
        status="ok",
        error=None,
        served_model=result.served_model,
        request_id=result.request_id,
        response_id=result.response_id,
        usage=result.usage,
        cost_usd=result.cost_usd,
        duration_s=round(time.monotonic() - started, 2),
    )
    genlog.append_row(paths.generation_log, row)
    cost = result.cost_usd if result.cost_usd is not None else estimate
    guard.record(f"scoring {job.scene_id}", cost)
    return result, log_id


@dataclass
class SelectOutcome:
    selected: dict[str, int] = field(default_factory=dict)
    failed: list[str] = field(default_factory=list)
    manifest_changed: bool = False


def _rel(paths: RepoPaths, path: Path) -> str:
    return path.relative_to(paths.root).as_posix()


def adopt_candidate(
    paths: RepoPaths,
    scene: Scene,
    number: int,
    candidate: Path,
    rows: list[dict[str, Any]],
    *,
    echo: Echo = print,
) -> None:
    """Record the choice in the manifest and copy the candidate to the master at master size."""
    sha = sha256_file(candidate)
    image_row = genlog.find_image_row(rows, scene.id, number, sha)
    log_ids: list[str] = []
    provider = model = None
    if image_row is None:
        echo(f"  note: no generation-log row produced {_rel(paths, candidate)}; provider unknown")
    else:
        log_ids.append(image_row["id"])
        provider, model = image_row.get("provider"), image_row.get("model")
    document = load_scores(scores_path(paths, scene.id))
    scored_this_file = document is not None and any(
        c.get("candidate") == number and c.get("sha256") == sha
        for c in document.get("candidates", [])
    )
    if scored_this_file and document.get("log_id"):
        log_ids.append(document["log_id"])
    scene.generation = Generation(
        provider=provider, model=model, selected_candidate=number, log_ids=log_ids
    )
    if write_if_changed(paths.master(scene.id), master_png_from(candidate)):
        echo(f"  copied cand-{number} to {_rel(paths, paths.master(scene.id))}")
    else:
        echo(f"  {_rel(paths, paths.master(scene.id))} already holds cand-{number}")


def run_select(
    paths: RepoPaths,
    manifest: Manifest,
    scene_ids: list[str],
    *,
    judge: ClaudeJudge | None = None,
    repick: bool = False,
    dry_run: bool = False,
    guard: SpendGuard | None = None,
    echo: Echo = print,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    from .review import write_contact_sheet

    judge = judge or make_judge(paths.root)
    by_id = manifest.by_id()
    before = dump_manifest(manifest)
    rows = genlog.read_rows(paths.generation_log)
    outcome = SelectOutcome()

    try:
        style_block = load_style_block(paths, allow_draft=dry_run).text
    except StyleBlockError as error:
        echo(f"error: {error}")
        return 1

    jobs: list[ScoringJob] = []
    for scene_id in scene_ids:
        scene = by_id[scene_id]
        candidates = existing_candidates(paths, scene_id)
        if not candidates:
            echo(f"{scene_id}: no candidates in {_rel(paths, paths.candidate_dir(scene_id))}")
            outcome.failed.append(scene_id)
            continue
        pinned = scene.generation.selected_candidate if scene.generation else None
        if pinned is not None and not repick:
            available = dict(candidates)
            if pinned not in available:
                echo(f"{scene_id}: pinned candidate {pinned} does not exist; use --repick")
                outcome.failed.append(scene_id)
                continue
            if dry_run:
                echo(f"{scene_id}: pinned to cand-{pinned}; would copy it to the master")
                continue
            echo(f"{scene_id}: pinned to cand-{pinned} (use --repick to score again)")
            adopt_candidate(paths, scene, pinned, available[pinned], rows, echo=echo)
            outcome.selected[scene_id] = pinned
            continue
        jobs.append(build_job(paths, manifest, scene, style_block))

    to_score = [job for job in jobs if repick or cached_scores(paths, job) is None]
    scoring_ids = {job.scene_id for job in to_score}
    estimate = sum(judge.estimate_cost(job) for job in to_score)
    guard = guard or SpendGuard.from_env()
    description = f"scoring {len(to_score)} scene(s) with {judge.model}"

    if dry_run:
        for job in to_score:
            echo(f"request for {job.scene_id} (estimated ${judge.estimate_cost(job):.3f}):")
            echo(json.dumps(judge.request(job, describe=True), indent=2, ensure_ascii=False))
        for job in jobs:
            if job.scene_id not in scoring_ids:
                echo(f"{job.scene_id}: scores.json is current; it would be reused")
        if to_score:
            try:
                echo(guard.check_plan(estimate, description))
            except SpendLimitExceeded as error:
                echo(f"note: a real run would refuse: {error}")
        echo("dry run: nothing was sent")
        return 1 if outcome.failed else 0

    if to_score:
        try:
            echo(guard.check_plan(estimate, description))
        except SpendLimitExceeded as error:
            echo(f"error: {error}")
            return 1

    for job in jobs:
        scene = by_id[job.scene_id]
        document = None if job.scene_id in scoring_ids else cached_scores(paths, job)
        if document is None:
            try:
                result, log_id = score_with_log(paths, judge, job, guard, echo=echo, sleep=sleep)
            except SpendLimitExceeded as error:
                echo(f"error: {error}")
                outcome.failed.append(job.scene_id)
                break
            except ProviderError as error:
                echo(f"{job.scene_id}: scoring failed: {error}")
                outcome.failed.append(job.scene_id)
                continue
            document = scores_document(paths, job, result, log_id)
            target = scores_path(paths, job.scene_id)
            write_if_changed(target, (json.dumps(document, indent=2) + "\n").encode("utf-8"))
        else:
            echo(f"{job.scene_id}: reusing scores.json (candidates unchanged)")
        sheet_path = write_contact_sheet(paths, manifest, job.scene_id)
        echo(f"  contact sheet: {_rel(paths, sheet_path)}")
        for entry in document["candidates"]:
            verdict = "pass" if entry["passed"] else "FAIL: " + ", ".join(entry["fail_reasons"])
            echo(f"  cand-{entry['candidate']}: {entry['total']:.2f} ({verdict})")
        winner = document.get("winner")
        if winner is None:
            echo(f"{job.scene_id}: no candidate passed the rubric. {document.get('advice', '')}")
            outcome.failed.append(job.scene_id)
            continue
        echo(f"{job.scene_id}: selected cand-{winner}")
        adopt_candidate(paths, scene, winner, dict(job.candidates)[winner], rows, echo=echo)
        outcome.selected[job.scene_id] = winner

    if dump_manifest(manifest) != before:
        save_manifest(manifest, paths.scenes_yaml)
        outcome.manifest_changed = True
        echo("updated content/scenes.yaml")
    if outcome.selected:
        echo("next: run `dsp grade` to grade the new masters")
    return 1 if outcome.failed else 0

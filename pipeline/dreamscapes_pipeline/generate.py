"""`dsp generate`: build each scene's prompt and generate candidate images.

The prompt is the style block from ``content/style.md`` followed by the scene description and
the bridge-object instruction (plan section 4.2). Every scene except the style anchor also gets
the ungraded anchor master attached as a reference image, with an instruction to borrow only
its palette and rendering; only the anchor is ever attached, never the previous scene, so
drift cannot accumulate.

Candidates are written to ``assets/candidates/<scene>/cand-<n>.png`` (numbered from 1).
Existing candidates are skipped unless ``--force``, so re-running only fills the gaps. Every
call appends one row to ``pipeline/logs/generations.jsonl`` (see `genlog`), and the spend guard
prints the estimate and enforces ``DSP_MAX_SPEND_USD`` before anything is sent.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import genlog
from .imaging import as_png, sha256_bytes
from .layout import RepoPaths
from .manifest import Manifest, Scene
from .placeholders import is_placeholder_image
from .providers import ASPECT, ImageProvider, MissingCredentials, ProviderError, ReferenceImage
from .retry import with_retries
from .spend import SpendGuard, SpendLimitExceeded

Echo = Callable[[str], None]

STYLE_HEADING = re.compile(r"^#{1,6}\s+style block\s*$", re.IGNORECASE)
ANY_HEADING = re.compile(r"^#{1,6}\s")

# Appendix B of the plan. Used only by --dry-run while content/style.md does not exist yet, so
# the request shape can be inspected; real runs require the authored file.
DRAFT_STYLE_BLOCK = (
    "A still, dreamlike scene rendered in soft painterly realism: not a cartoon, not a "
    "photograph. Diffused light with a faint haze, long gentle shadows, a thin mist in the "
    "distance. Muted palette of dusty teal, slate blue and lavender, with warm amber glow on "
    "every light source. Gentle bloom around lights, fine film grain, slightly lifted blacks, "
    "subtle vignette. Wide establishing shot from eye level with a 35 mm lens and deep depth of "
    "field. The place is empty and quiet, as if everyone has just left. No people, no faces, no "
    "legible text, no signs, no logos, no watermarks. Calm, melancholic, inviting."
)

REFERENCE_INSTRUCTION = (
    "Match the colour palette, lighting, grain and rendering style of the reference image only. "
    "Do not reuse its composition, objects or layout."
)


class StyleBlockError(ValueError):
    """content/style.md is missing or has no usable style block."""


@dataclass(frozen=True)
class StyleBlock:
    text: str
    source: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def parse_style_block(markdown: str) -> str:
    """The first non-empty paragraph under the ``## Style block`` heading.

    Lines of that paragraph are joined with single spaces (Markdown's own reading of a wrapped
    paragraph); blockquote markers and code fences around it are dropped.
    """
    lines = markdown.splitlines()
    start = next((i for i, line in enumerate(lines) if STYLE_HEADING.match(line.strip())), None)
    if start is None:
        raise StyleBlockError("content/style.md has no '## Style block' heading")
    paragraph: list[str] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if ANY_HEADING.match(stripped):
            break
        if stripped.startswith("```"):
            if paragraph:
                break
            continue
        stripped = re.sub(r"^>\s?", "", stripped).strip()
        if not stripped:
            if paragraph:
                break
            continue
        paragraph.append(stripped)
    if not paragraph:
        raise StyleBlockError("the '## Style block' section of content/style.md is empty")
    return " ".join(paragraph)


def load_style_block(paths: RepoPaths, *, allow_draft: bool) -> StyleBlock:
    path = paths.style_md
    if not path.exists():
        if allow_draft:
            return StyleBlock(DRAFT_STYLE_BLOCK, "the plan's draft (content/style.md is missing)")
        raise StyleBlockError(
            "content/style.md is missing; the content package writes it. Real generation needs "
            "the authored style block."
        )
    return StyleBlock(parse_style_block(path.read_text(encoding="utf-8")), "content/style.md")


# --------------------------------------------------------------------------------------------
# Prompt assembly
# --------------------------------------------------------------------------------------------

_ARTICLE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)


def _bare(phrase: str) -> str:
    return _ARTICLE.sub("", phrase.strip())


def with_indefinite_article(phrase: str) -> str:
    if _ARTICLE.match(phrase.strip()):
        return phrase.strip()
    bare = phrase.strip()
    return ("an " if bare[:1].lower() in "aeiou" else "a ") + bare


def with_definite_article(phrase: str) -> str:
    return "the " + _bare(phrase)


def build_prompt(style_block: str, scene: Scene, *, with_reference: bool) -> str:
    """The full prompt for one scene, following the template in plan section 4.2."""
    obj = scene.bridge.object
    placement = scene.bridge.placement.strip().rstrip(".")
    bridge = (
        f"Include {with_indefinite_article(obj)}: {placement}. It must be clearly recognisable "
        "but not the main subject, occupying roughly 3-8% of the frame, located within the "
        "central 60% of the image width. Nothing else in the scene should resemble "
        f"{with_definite_article(obj)}."
    )
    parts = [style_block.strip(), f"Scene: {scene.scene.strip()}", bridge]
    if with_reference:
        parts.append(REFERENCE_INSTRUCTION)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------------------------


def candidate_path(paths: RepoPaths, scene_id: str, number: int) -> Path:
    return paths.candidate_dir(scene_id) / f"cand-{number}.png"


_CANDIDATE = re.compile(r"^cand-(\d+)\.png$")


def existing_candidates(paths: RepoPaths, scene_id: str) -> list[tuple[int, Path]]:
    folder = paths.candidate_dir(scene_id)
    if not folder.is_dir():
        return []
    found = []
    for path in folder.iterdir():
        match = _CANDIDATE.match(path.name)
        if match and path.is_file():
            found.append((int(match.group(1)), path))
    return sorted(found)


def seed_for(scene_id: str, candidate: int, attempt: int) -> int:
    """A reproducible seed per slot; a forced regeneration (next attempt) gets a new one."""
    digest = hashlib.sha256(f"{scene_id}:{candidate}:{attempt}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


@dataclass
class PlannedImage:
    scene_id: str
    candidate: int
    out_path: Path
    prompt: str
    references: list[ReferenceImage]
    seed: int
    request: dict[str, Any]
    estimate: float
    purpose: str = "generate"


@dataclass
class GenerationPlan:
    items: list[PlannedImage] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    """Reasons a real run must not start; a dry run prints them as notes."""

    @property
    def estimate(self) -> float:
        return sum(item.estimate for item in self.items)


def anchor_reference(
    paths: RepoPaths, manifest: Manifest, *, require_real: bool
) -> tuple[ReferenceImage | None, str | None]:
    """The ungraded style-anchor master, or why it cannot be used."""
    anchor = paths.master(manifest.style_anchor)
    rel = anchor.relative_to(paths.root).as_posix()
    if not anchor.exists():
        return None, (
            f"the style anchor master {rel} does not exist yet; generate and select "
            f"'{manifest.style_anchor}' first, or pass --no-reference"
        )
    if is_placeholder_image(anchor):
        problem = (
            f"the style anchor master {rel} is still a placeholder; generate and select "
            f"'{manifest.style_anchor}' first, or pass --no-reference"
        )
        return (None if require_real else ReferenceImage.from_path(anchor)), problem
    return ReferenceImage.from_path(anchor), None


def plan_generation(
    paths: RepoPaths,
    manifest: Manifest,
    scene_ids: list[str],
    provider: ImageProvider,
    *,
    count: int,
    size: str,
    style: StyleBlock,
    no_reference: bool = False,
    force: bool = False,
    dry_run: bool = False,
    rows: list[dict[str, Any]] | None = None,
) -> GenerationPlan:
    plan = GenerationPlan()
    rows = rows if rows is not None else genlog.read_rows(paths.generation_log)
    by_id = manifest.by_id()
    for scene_id in scene_ids:
        scene = by_id[scene_id]
        wants_reference = not no_reference and scene_id != manifest.style_anchor
        references: list[ReferenceImage] = []
        if wants_reference:
            ref, problem = anchor_reference(paths, manifest, require_real=not dry_run)
            if problem:
                plan.problems.append(f"{scene_id}: {problem}")
            if ref is not None:
                references.append(ref)
        prompt = build_prompt(style.text, scene, with_reference=wants_reference)
        for number in range(1, count + 1):
            out = candidate_path(paths, scene_id, number)
            if out.exists() and not force:
                plan.skipped.append(f"{scene_id} cand-{number}")
                continue
            attempt = genlog.image_attempts(rows, scene_id, number, provider.name, "generate")
            seed = seed_for(scene_id, number, attempt)
            plan.items.append(
                PlannedImage(
                    scene_id=scene_id,
                    candidate=number,
                    out_path=out,
                    prompt=prompt,
                    references=references,
                    seed=seed,
                    request=provider.request(prompt, ASPECT, size, references, seed),
                    estimate=provider.estimate_cost(size, len(references)),
                )
            )
    return plan


# --------------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------------


def _parameters(request: dict[str, Any], size: str) -> dict[str, Any]:
    """The request minus the prompt and image payloads, which the row stores separately."""
    params = {k: v for k, v in request.items() if k not in ("prompt", "input", "image")}
    params.setdefault("aspect", ASPECT)
    params["requested_size"] = size
    return params


def _references(paths: RepoPaths, refs: list[ReferenceImage]) -> list[dict[str, Any]]:
    out = []
    for ref in refs:
        shown = (
            ref.path.relative_to(paths.root) if ref.path.is_relative_to(paths.root) else ref.path
        )
        out.append({"path": shown.as_posix(), "sha256": ref.sha256, "bytes": len(ref.data)})
    return out


@dataclass
class GenerationOutcome:
    written: list[Path] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    spent: float = 0.0


def execute_plan(
    paths: RepoPaths,
    provider: ImageProvider,
    plan: GenerationPlan,
    *,
    size: str,
    style: StyleBlock,
    guard: SpendGuard,
    echo: Echo = print,
    sleep: Callable[[float], None] = time.sleep,
) -> GenerationOutcome:
    """Send every planned request, write the images and log one row per call.

    A failed scene stops after its first failure (the same prompt would most likely fail again)
    and the run continues with the next scene. A spend-limit stop ends the run.
    """
    outcome = GenerationOutcome()
    failed_scenes: set[str] = set()
    for item in plan.items:
        if item.scene_id in failed_scenes:
            continue
        what = f"{item.scene_id} cand-{item.candidate}"
        guard.before_call(item.estimate, what)
        row: dict[str, Any] = {
            "id": genlog.new_row_id(),
            "timestamp": genlog.utc_now(),
            "kind": "image",
            "purpose": item.purpose,
            "provider": provider.name,
            "model": provider.model,
            "scene": item.scene_id,
            "candidate": item.candidate,
            "prompt": item.prompt,
            "parameters": _parameters(item.request, size),
            "references": _references(paths, item.references),
            "seed": item.seed,
            "style_block_sha256": style.sha256,
            "cost_estimate_usd": item.estimate,
        }
        started = time.monotonic()
        echo(f"generating {what} with {provider.name} {provider.model} ...")
        try:
            result = with_retries(
                lambda item=item: provider.generate(
                    item.prompt, ASPECT, size, item.references, item.seed
                ),
                what=what,
                echo=echo,
                sleep=sleep,
            )
        except MissingCredentials:
            raise  # nothing was sent, so there is nothing to log; the run stops
        except ProviderError as error:
            row.update(
                status="error",
                error=str(error),
                cost_usd=None,
                duration_s=round(time.monotonic() - started, 2),
            )
            genlog.append_row(paths.generation_log, row)
            echo(f"  failed: {error}")
            outcome.failed.append(what)
            failed_scenes.add(item.scene_id)
            continue

        png, dims = as_png(result.image)
        item.out_path.parent.mkdir(parents=True, exist_ok=True)
        item.out_path.write_bytes(png)
        cost = result.cost_usd if result.cost_usd is not None else result.cost_estimate_usd
        row.update(
            model=result.model,
            seed=result.seed,
            status="ok",
            error=None,
            request_id=result.request_id,
            response_id=result.response_id,
            revised_prompt=result.revised_prompt,
            output_path=item.out_path.relative_to(paths.root).as_posix(),
            output_sha256=sha256_bytes(png),
            output_size=list(dims),
            cost_estimate_usd=result.cost_estimate_usd,
            cost_usd=result.cost_usd,
            usage=result.usage,
            text=result.text,
            duration_s=round(time.monotonic() - started, 2),
        )
        genlog.append_row(paths.generation_log, row)
        guard.record(what, cost)
        outcome.spent += cost
        outcome.written.append(item.out_path)
        echo(f"  wrote {row['output_path']} ({dims[0]}x{dims[1]}, ${cost:.3f})")
    return outcome


def print_requests(plan: GenerationPlan, paths: RepoPaths, echo: Echo) -> None:
    for number, item in enumerate(plan.items, start=1):
        rel = item.out_path.relative_to(paths.root).as_posix()
        echo(
            f"request {number} of {len(plan.items)}: {item.scene_id} cand-{item.candidate} -> "
            f"{rel} (estimated ${item.estimate:.3f})"
        )
        echo(json.dumps(item.request, indent=2, ensure_ascii=False))


def run_generate(
    paths: RepoPaths,
    manifest: Manifest,
    scene_ids: list[str],
    provider: ImageProvider,
    *,
    count: int,
    size: str | None = None,
    no_reference: bool = False,
    force: bool = False,
    dry_run: bool = False,
    guard: SpendGuard | None = None,
    echo: Echo = print,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """The whole command; returns the exit code."""
    size = size or provider.default_size()
    try:
        style = load_style_block(paths, allow_draft=dry_run)
        plan = plan_generation(
            paths,
            manifest,
            scene_ids,
            provider,
            count=count,
            size=size,
            style=style,
            no_reference=no_reference,
            force=force,
            dry_run=dry_run,
        )
    except (StyleBlockError, ValueError) as error:
        echo(f"error: {error}")
        return 1

    echo(f"style block from {style.source} (sha256 {style.sha256[:12]})")
    if plan.skipped:
        echo(f"already generated, skipped (use --force to redo): {', '.join(plan.skipped)}")
    for problem in plan.problems:
        echo(("note: a real run would refuse: " if dry_run else "error: ") + problem)
    if plan.problems and not dry_run:
        return 1
    if not plan.items:
        echo("nothing to generate")
        return 0

    description = f"{len(plan.items)} image(s) with {provider.name} {provider.model} at {size}"
    guard = guard or SpendGuard.from_env()
    if dry_run:
        print_requests(plan, paths, echo)
        try:
            echo(guard.check_plan(plan.estimate, description))
        except SpendLimitExceeded as error:
            echo(f"note: a real run would refuse: {error}")
        echo("dry run: nothing was sent")
        return 0

    try:
        echo(guard.check_plan(plan.estimate, description))
        outcome = execute_plan(
            paths, provider, plan, size=size, style=style, guard=guard, echo=echo, sleep=sleep
        )
    except (SpendLimitExceeded, MissingCredentials) as error:
        echo(f"error: {error}")
        return 1
    echo(
        f"generated {len(outcome.written)} image(s), {len(outcome.failed)} failed; "
        f"spent about ${outcome.spent:.2f}"
    )
    return 1 if outcome.failed else 0

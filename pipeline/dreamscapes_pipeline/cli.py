"""The `dsp` command line: `uv run dsp <command>` from anywhere inside the repository.

Every command finds the repository root by walking up from the current directory to the first
directory that contains both ``content/`` and ``pipeline/pyproject.toml``, and resolves all
paths from there.

`validate`, `schema` and `placeholders` work today. The other commands are registered with
their final names and options so scripts and docs can rely on them; the work packages that
implement them replace the stub bodies.
"""

from __future__ import annotations

import contextlib
from typing import Annotated, NoReturn

import typer
from dotenv import load_dotenv

from .layout import RepoPaths, RepoRootNotFound
from .manifest import ManifestError

app = typer.Typer(
    name="dsp",
    help="Dreamscapes offline pipeline: generate, grade, hotspot, export and validate scenes.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

NOT_IMPLEMENTED_EXIT = 2


def _repo() -> RepoPaths:
    try:
        return RepoPaths.discover()
    except RepoRootNotFound as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(1) from error


def _not_implemented(command: str) -> NoReturn:
    _repo()
    typer.echo(f"dsp {command}: not implemented yet", err=True)
    raise typer.Exit(NOT_IMPLEMENTED_EXIT)


# Shared option declarations, so the same flag reads the same everywhere.
SceneOpt = Annotated[str | None, typer.Option("--scene", help="Scene id (slug) to process.")]
AllOpt = Annotated[bool, typer.Option("--all", help="Process every scene in the manifest.")]
DryRunOpt = Annotated[
    bool, typer.Option("--dry-run", help="Print the requests that would be sent; send nothing.")
]


@app.command()
def validate(
    strict: Annotated[
        bool, typer.Option("--strict", help="Treat warnings as errors (exit 1 on any).")
    ] = False,
) -> None:
    """Check the manifest, the hotspot rules, the loop and every referenced file."""
    from .validate import validate_repo

    result = validate_repo(_repo())
    for issue in result.issues:
        typer.echo(str(issue))
    errors, warnings = len(result.errors), len(result.warnings)
    typer.echo(f"{errors} error(s), {warnings} warning(s)")
    if errors or (strict and warnings):
        raise typer.Exit(1)


@app.command()
def schema(
    check: Annotated[
        bool,
        typer.Option(
            "--check", help="Only verify the committed schema is current (exit 1 if not)."
        ),
    ] = False,
) -> None:
    """Write schema/scenes.schema.json from the built-manifest model."""
    from .schema import schema_is_current, write_schema

    paths = _repo()
    rel = paths.schema_json.relative_to(paths.root)
    if check:
        if not schema_is_current(paths):
            typer.echo(f"{rel} is out of date; run `uv run dsp schema`", err=True)
            raise typer.Exit(1)
        typer.echo(f"{rel} is up to date")
        return
    changed = write_schema(paths)
    typer.echo(f"{'wrote' if changed else 'unchanged'} {rel}")


@app.command()
def placeholders(
    overwrite_masters: Annotated[
        bool,
        typer.Option(
            "--overwrite-masters",
            help="Destroys real work: repaint every master in assets/masters (including ones "
            "made by `dsp select`) and its graded copy, mask and exports with placeholders. "
            "Leaves content/scenes.yaml alone.",
        ),
    ] = False,
    reset_manifest: Annotated[
        bool,
        typer.Option(
            "--reset-manifest",
            help="Destroys authored content: replace content/scenes.yaml with the placeholder "
            "loop, losing its titles, descriptions, bridge objects and hotspot polygons. "
            "Leaves real masters alone.",
        ),
    ] = False,
) -> None:
    """Write synthetic masters, masks, exports and scenes.json so the site runs with no keys.

    Without flags it only replaces earlier placeholder output and never touches real work.
    """
    from .placeholders import run_placeholders

    paths = _repo()
    try:
        report = run_placeholders(
            paths, overwrite_masters=overwrite_masters, reset_manifest=reset_manifest
        )
    except ManifestError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(1) from error
    if report.rendered:
        typer.echo(f"placeholder scenes: {', '.join(report.rendered)}")
    if report.kept:
        typer.echo(f"kept real masters: {', '.join(report.kept)}")
    manifest_messages = {
        "written": "wrote content/scenes.yaml",
        "unchanged": "content/scenes.yaml is unchanged",
        "kept": "kept existing content/scenes.yaml",
    }
    typer.echo(manifest_messages.get(report.manifest, ""))
    if report.scenes_json == "written":
        typer.echo("wrote web/public/scenes.json")
    elif report.scenes_json == "unchanged":
        typer.echo("web/public/scenes.json is unchanged")
    for note in report.notes:
        typer.echo(f"note: {note}")


@app.command()
def generate(
    scene: SceneOpt = None,
    all_: AllOpt = False,
    candidates: Annotated[
        int, typer.Option("--candidates", min=1, help="Candidates to generate per scene.")
    ] = 4,
    provider: Annotated[
        str,
        typer.Option(
            "--provider", envvar="DSP_IMAGE_PROVIDER", help="Image provider: gemini or openai."
        ),
    ] = "gemini",
    no_reference: Annotated[
        bool, typer.Option("--no-reference", help="Do not attach the style anchor image.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Regenerate candidates that already exist.")
    ] = False,
    dry_run: DryRunOpt = False,
) -> None:
    """Generate candidate images from the style bible and the scene description."""
    _not_implemented("generate")


@app.command()
def select(
    scene: SceneOpt = None,
    all_: AllOpt = False,
    repick: Annotated[
        bool, typer.Option("--repick", help="Re-score even when a candidate is already pinned.")
    ] = False,
    dry_run: DryRunOpt = False,
) -> None:
    """Have Claude score the candidates, write a contact sheet and copy the winner to masters."""
    _not_implemented("select")


@app.command()
def grade(
    preview: Annotated[
        bool, typer.Option("--preview", help="Also write before/after pairs to the review sheet.")
    ] = False,
) -> None:
    """Apply the deterministic colour grade to every master."""
    _not_implemented("grade")


@app.command()
def hotspot(
    scene: SceneOpt = None,
    all_: AllOpt = False,
    box: Annotated[
        tuple[float, float, float, float] | None,
        typer.Option(
            "--box",
            metavar="X0 Y0 X1 Y1",
            help="Hint box in normalised coordinates to steer segmentation.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="With --all, also redo scenes that are already verified."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print the requests that would be sent and send nothing. content/scenes.yaml "
            "and the masks are left alone; review files are still written to assets/review/ "
            "(git-ignored).",
        ),
    ] = False,
) -> None:
    """Segment the bridge object into a hotspot polygon, verify it and write it to the YAML."""
    from .hotspots import run_hotspot_command

    code = run_hotspot_command(
        _repo(),
        scene=scene,
        all_scenes=all_,
        box=box,
        dry_run=dry_run,
        force=force,
        echo=typer.echo,
    )
    if code:
        raise typer.Exit(code)


@app.command()
def export() -> None:
    """Write AVIF/WebP variants, blurred previews and web/public/scenes.json from graded masters."""
    _not_implemented("export")


@app.command()
def review() -> None:
    """Rebuild every contact sheet and the review index page."""
    _not_implemented("review")


@app.command()
def morph(
    edge: Annotated[
        tuple[str, str] | None,
        typer.Option("--edge", metavar="FROM TO", help="Scene ids of the transition to render."),
    ] = None,
    all_: AllOpt = False,
    dry_run: DryRunOpt = False,
) -> None:
    """Render a first-frame/last-frame morph clip for a transition."""
    _not_implemented("morph")


@app.command()
def bakeoff(dry_run: DryRunOpt = False) -> None:
    """Generate the first scene with each image provider and write a comparison sheet."""
    _not_implemented("bakeoff")


def main() -> None:
    # Load keys before parsing so options backed by environment variables see .env values.
    # A missing root is not an error here; each command reports it itself.
    with contextlib.suppress(RepoRootNotFound):
        load_dotenv(RepoPaths.discover().root / ".env", override=False)
    app()

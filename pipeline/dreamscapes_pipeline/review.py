"""Review pages under ``assets/review/`` (git-ignored), opened straight from disk in a browser.

* ``<scene>/contact-sheet.html``: every candidate with Claude's scores, the winner and the
  candidate pinned in ``content/scenes.yaml`` (written by `dsp select`, rebuilt by `dsp review`).
* ``index.html``: a strip of every graded master side by side in loop order, to check that the
  set looks like one set, plus one row per scene linking to its contact sheet and hotspot
  overlay (written by `dsp review`).
* ``grade.html``: before/after pairs of the colour grade (written by `dsp grade --preview`).
* ``bakeoff.html``: the provider comparison (written by `dsp bakeoff`).

Images are referenced by relative paths, so the pages work as plain files.
"""

from __future__ import annotations

import html
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from PIL import Image

from .imaging import webp_preview, write_if_changed
from .layout import RepoPaths
from .manifest import Manifest
from .placeholders import is_placeholder_image

if TYPE_CHECKING:
    from .grade import GradeReport, LoadedParams

THUMB_WIDTH = 512

STYLE = """
:root { color-scheme: dark; --bg: #14161b; --card: #1d2027; --line: #2c313b;
  --text: #e4e6eb; --muted: #9aa1ad; --pass: #7fc6a4; --fail: #e58a7b; --pick: #e0b56a; }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px 48px; background: var(--bg); color: var(--text);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1400px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 4px; } h2 { font-size: 1.15rem; margin: 32px 0 12px; }
p { margin: 4px 0; } a { color: #9cc3ff; } .muted { color: var(--muted); }
.grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); }
.pairs { display: grid; gap: 16px; grid-template-columns: repeat(auto-fill, minmax(560px, 1fr)); }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
  padding: 10px; }
.card.pick { border-color: var(--pick); box-shadow: 0 0 0 1px var(--pick); }
.card img { width: 100%; height: auto; display: block; border-radius: 4px; background: #000; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.badge { display: inline-block; padding: 0 8px; border-radius: 999px; font-size: 0.8rem;
  border: 1px solid var(--line); margin-right: 4px; }
.pass { color: var(--pass); border-color: var(--pass); }
.fail { color: var(--fail); border-color: var(--fail); }
.pickb { color: var(--pick); border-color: var(--pick); }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line);
  vertical-align: top; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.strip { display: grid; gap: 4px; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); }
.strip figure { margin: 0; } .strip img { width: 100%; display: block; border-radius: 3px; }
.strip figcaption { font-size: 0.75rem; color: var(--muted); white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; }
.thumb { width: 200px; border-radius: 4px; display: block; }
"""


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def url(from_dir: Path, target: Path) -> str:
    return quote(Path(os.path.relpath(target, from_dir)).as_posix())


def page(title: str, body: str) -> str:
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{esc(title)}</title>\n<style>{STYLE}</style>\n</head>\n"
        f"<body><main>\n{body}\n</main></body>\n</html>\n"
    )


def _write(path: Path, text: str) -> Path:
    write_if_changed(path, text.encode("utf-8"))
    return path


def _score_table(entry: dict[str, Any]) -> str:
    cells = "".join(
        f'<tr><td>{esc(name.replace("_", " "))}</td><td class="num">{esc(value)}</td></tr>'
        for name, value in entry.get("scores", {}).items()
    )
    return f"<table>{cells}</table>"


def _verdict(entry: dict[str, Any]) -> str:
    if entry.get("passed"):
        return '<span class="badge pass">pass</span>'
    reasons = ", ".join(entry.get("fail_reasons", []))
    return f'<span class="badge fail">fail: {esc(reasons)}</span>'


# --------------------------------------------------------------------------------------------
# Contact sheets
# --------------------------------------------------------------------------------------------


def contact_sheet_path(paths: RepoPaths, scene_id: str) -> Path:
    return paths.review_dir(scene_id) / "contact-sheet.html"


def write_contact_sheet(paths: RepoPaths, manifest: Manifest, scene_id: str) -> Path:
    from .generate import existing_candidates
    from .select import load_scores, scores_path

    scene = manifest.scene(scene_id)
    target = contact_sheet_path(paths, scene_id)
    here = target.parent
    document = load_scores(scores_path(paths, scene_id)) or {}
    scored = {entry["candidate"]: entry for entry in document.get("candidates", [])}
    pinned = scene.generation.selected_candidate if scene.generation else None
    winner = document.get("winner")

    cards = []
    for number, path in existing_candidates(paths, scene_id):
        entry = scored.get(number)
        badges = []
        if number == pinned:
            badges.append('<span class="badge pickb">selected</span>')
        if number == winner and winner != pinned:
            badges.append('<span class="badge pickb">judge\'s pick</span>')
        if entry:
            badges.append(_verdict(entry))
            detail = (
                f"<p><strong>{entry['total']:.2f}</strong> / 5 weighted</p>"
                f'{_score_table(entry)}<p class="muted">{esc(entry.get("notes"))}</p>'
            )
        else:
            detail = '<p class="muted">not scored yet (run dsp select)</p>'
        link = url(here, path)
        cards.append(
            f'<div class="card{" pick" if number == pinned else ""}">'
            f'<a href="{link}"><img src="{link}" alt="candidate {number}" loading="lazy"></a>'
            f"<p>cand-{number} {''.join(badges)}</p>{detail}</div>"
        )

    anchor_html = ""
    anchor = document.get("anchor")
    if anchor:
        link = url(here, paths.root / anchor["path"])
        anchor_html = (
            "<h2>Style anchor shown to the judge</h2>"
            f'<div class="grid"><div class="card"><img src="{link}" alt="style anchor"></div></div>'
        )
    extras = []
    master = paths.master(scene_id)
    if master.exists():
        extras.append(f'<a href="{url(here, master)}">master</a>')
    if paths.graded(scene_id).exists():
        extras.append(f'<a href="{url(here, paths.graded(scene_id))}">graded master</a>')
    overlay = paths.review_dir(scene_id) / "hotspot.png"
    if overlay.exists():
        extras.append(f'<a href="{url(here, overlay)}">hotspot overlay</a>')
    advice = document.get("advice")
    judged = (
        f"Scored by {esc(document.get('served_model') or document.get('model'))} at "
        f"{esc(document.get('timestamp'))}."
        if document
        else "Not scored yet."
    )
    body = (
        f'<h1>{esc(scene.title)} <span class="muted">({esc(scene_id)})</span></h1>'
        f"<p>{esc(scene.scene)}</p>"
        f"<p>Bridge object: <strong>{esc(scene.bridge.object)}</strong>, "
        f"{esc(scene.bridge.placement)}</p>"
        f'<p class="muted">{judged} {" · ".join(extras)} · <a href="../index.html">index</a></p>'
        + (f"<p>Advice: {esc(advice)}</p>" if advice else "")
        + f'<h2>Candidates</h2><div class="grid">{"".join(cards) or "<p>none</p>"}</div>'
        + anchor_html
    )
    return _write(target, page(f"{scene_id} candidates", body))


# --------------------------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------------------------


def loop_order(manifest: Manifest) -> list[str]:
    """Scene ids following ``next`` from the start; any scene off the loop goes last."""
    by_id = manifest.by_id()
    order: list[str] = []
    current = manifest.start if manifest.start in by_id else manifest.scenes[0].id
    while current in by_id and current not in order:
        order.append(current)
        current = by_id[current].next
    order.extend(s.id for s in manifest.scenes if s.id not in order)
    return order


def _thumbnail(paths: RepoPaths, scene_id: str) -> Path | None:
    source = paths.graded(scene_id)
    if not source.exists():
        source = paths.master(scene_id)
    if not source.exists():
        return None
    target = paths.review / "thumbs" / f"{scene_id}.webp"
    with Image.open(source) as opened:
        write_if_changed(target, webp_preview(opened, THUMB_WIDTH))
    return target


@dataclass
class ReviewReport:
    contact_sheets: list[Path] = field(default_factory=list)
    index: Path | None = None


def run_review(paths: RepoPaths, manifest: Manifest) -> ReviewReport:
    from .generate import existing_candidates
    from .select import load_scores, scores_path

    report = ReviewReport()
    here = paths.review
    strip, rows = [], []
    for scene_id in loop_order(manifest):
        scene = manifest.scene(scene_id)
        has_candidates = bool(existing_candidates(paths, scene_id))
        document = load_scores(scores_path(paths, scene_id))
        sheet = None
        if has_candidates or document:
            sheet = write_contact_sheet(paths, manifest, scene_id)
            report.contact_sheets.append(sheet)
        thumb = _thumbnail(paths, scene_id)
        placeholder = paths.master(scene_id).exists() and is_placeholder_image(
            paths.master(scene_id)
        )
        if thumb is not None:
            strip.append(
                f'<figure><img src="{url(here, thumb)}" alt="{esc(scene.title)}">'
                f"<figcaption>{esc(scene_id)}</figcaption></figure>"
            )
        pinned = scene.generation.selected_candidate if scene.generation else None
        total = ""
        if document and pinned is not None:
            entry = next(
                (c for c in document.get("candidates", []) if c["candidate"] == pinned), None
            )
            total = f"{entry['total']:.2f}" if entry else ""
        links = []
        if sheet is not None:
            links.append(f'<a href="{url(here, sheet)}">contact sheet</a>')
        overlay = paths.review_dir(scene_id) / "hotspot.png"
        if overlay.exists():
            links.append(f'<a href="{url(here, overlay)}">hotspot</a>')
        thumb_html = (
            f'<img class="thumb" src="{url(here, thumb)}" alt="">' if thumb is not None else ""
        )
        rows.append(
            f"<tr><td>{thumb_html}</td><td><strong>{esc(scene.title)}</strong><br>"
            f'<span class="muted">{esc(scene_id)}{" · placeholder" if placeholder else ""}'
            f"</span></td><td>{esc(scene.bridge.object)}</td>"
            f'<td class="num">{esc(pinned if pinned is not None else "")}</td>'
            f'<td class="num">{esc(total)}</td>'
            f'<td class="num">{len(existing_candidates(paths, scene_id))}</td>'
            f"<td>{' · '.join(links)}</td></tr>"
        )
    pages = []
    for name, label in (("bakeoff.html", "provider bake-off"), ("grade.html", "grade preview")):
        if (here / name).exists():
            pages.append(f'<a href="{name}">{label}</a>')
    body = (
        "<h1>Dreamscapes review</h1>"
        f'<p class="muted">{" · ".join(pages) or "no bake-off or grade preview yet"}</p>'
        "<h2>All graded masters, in loop order</h2>"
        f'<div class="strip">{"".join(strip) or "<p>no masters yet</p>"}</div>'
        "<h2>Scenes</h2><table><tr><th></th><th>Scene</th><th>Bridge object</th>"
        '<th class="num">Selected</th><th class="num">Score</th>'
        '<th class="num">Candidates</th><th>Pages</th></tr>'
        f"{''.join(rows)}</table>"
    )
    report.index = _write(here / "index.html", page("Dreamscapes review", body))
    return report


# --------------------------------------------------------------------------------------------
# Grade preview and bake-off
# --------------------------------------------------------------------------------------------


def write_grade_page(
    paths: RepoPaths, manifest: Manifest, loaded: LoadedParams, report: GradeReport
) -> Path:
    here = paths.review
    params = loaded.params.model_dump()
    table = "".join(
        f"<tr><td>{esc(section)}.{esc(key)}</td><td>{esc(value)}</td></tr>"
        for section, values in params.items()
        for key, value in values.items()
    )
    titles = {s.id: s.title for s in manifest.scenes}
    pairs = "".join(
        f'<div class="card"><p>{esc(titles.get(sid, sid))} <span class="muted">({esc(sid)})'
        f'</span></p><div class="pair"><figure style="margin:0"><img src="{quote(before)}" '
        f'alt="before"><figcaption class="muted">before</figcaption></figure>'
        f'<figure style="margin:0"><img src="{quote(after)}" alt="after">'
        f'<figcaption class="muted">after</figcaption></figure></div></div>'
        for sid, before, after in report.previews
    )
    warnings = "".join(f'<p class="fail">{esc(w)}</p>' for w in loaded.warnings)
    body = (
        "<h1>Grade preview</h1>"
        f'<p class="muted">Parameters from {esc(loaded.source)}. Placeholder masters are shown '
        "graded here but their graded files are left to dsp placeholders.</p>"
        f"{warnings}<table>{table}</table><h2>Before and after</h2>"
        f'<div class="pairs">{pairs or "<p>no masters</p>"}</div>'
    )
    return _write(here / "grade.html", page("Grade preview", body))


def write_bakeoff_page(paths: RepoPaths, document: dict[str, Any]) -> Path:
    here = paths.review
    target = here / "bakeoff.html"
    entries = document.get("candidates", [])
    providers: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        providers.setdefault(entry["provider"], []).append(entry)

    summary_rows = []
    for name, group in providers.items():
        scored = [e for e in group if "total" in e]
        mean = sum(e["total"] for e in scored) / len(scored) if scored else None
        passed = sum(1 for e in scored if e.get("passed"))
        cost = sum(e.get("cost_usd") or 0.0 for e in group)
        summary_rows.append(
            f"<tr><td>{esc(name)}</td><td>{esc(group[0].get('model'))}</td>"
            f'<td class="num">{f"{mean:.2f}" if mean is not None else "-"}</td>'
            f'<td class="num">{passed} / {len(group)}</td><td class="num">${cost:.3f}</td></tr>'
        )
    sections = []
    for name, group in providers.items():
        cards = []
        for entry in group:
            graded = here / entry["graded"]
            raw = paths.root / entry["path"]
            badge = _verdict(entry) if "total" in entry else ""
            detail = (
                f"<p><strong>{entry['total']:.2f}</strong> / 5 weighted</p>{_score_table(entry)}"
                f'<p class="muted">{esc(entry.get("notes"))}</p>'
                if "total" in entry
                else '<p class="muted">not scored</p>'
            )
            cards.append(
                f'<div class="card"><a href="{url(here, raw)}"><img src="{url(here, graded)}" '
                f'alt="{esc(name)} {entry["candidate"]}" loading="lazy"></a>'
                f"<p>{esc(name)} {entry['candidate']} {badge} "
                f'<a class="muted" href="{url(here, raw)}">ungraded</a></p>{detail}</div>'
            )
        sections.append(f'<h2>{esc(name)}</h2><div class="grid">{"".join(cards)}</div>')
    body = (
        "<h1>Image provider bake-off</h1>"
        f"<p>Scene <strong>{esc(document.get('scene'))}</strong>, the same prompt for every "
        "image. Images are shown graded; click one for the ungraded original. Claude scored "
        "them blind, without knowing which provider made which.</p>"
        f'<p class="muted">Judge: {esc(document.get("judge_model") or "not scored")}. '
        '<a href="index.html">index</a></p>'
        '<table><tr><th>Provider</th><th>Model</th><th class="num">Mean score</th>'
        '<th class="num">Passed</th><th class="num">Cost</th></tr>'
        f"{''.join(summary_rows)}</table>{''.join(sections)}"
        f'<h2>Prompt</h2><p class="muted" style="white-space: pre-wrap">'
        f"{esc(document.get('prompt'))}</p>"
    )
    return _write(target, page("Provider bake-off", body))

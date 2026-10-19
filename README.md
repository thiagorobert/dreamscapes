# Dreamscapes

A full-screen web page that shows one dreamlike, AI-generated scene at a time. Somewhere in each
scene one object is secretly clickable (the "bridge object"). Clicking it dissolves the current
scene into the next one, which is about that object: the distant lighthouse leads to floodlit
towers, a basketball hoop leads to a rooftop court, and so on. Ten scenes form a closed loop, so
the walk never ends. Everything is generated offline by a Python pipeline and committed; the
site is a static Vite + TypeScript page with no backend.

The architecture and implementation plan lives in
[`docs/plans/2026-10-19-dreamscapes-v1-plan.md`](docs/plans/2026-10-19-dreamscapes-v1-plan.md).

## Repository layout

```
content/          authored content: scenes.yaml (the loop), style.md, grade.toml
pipeline/         Python 3.12 package `dreamscapes_pipeline`, run with `uv run dsp <command>`
assets/
  masters/        2048x1152 master images (<scene>.png) and graded copies (<scene>.graded.png)
  masks/          hotspot masks, one per scene
  candidates/     generated candidates (git-ignored)
  review/         contact sheets and overlays (git-ignored)
web/              the site: Vite + strict TypeScript, managed with pnpm
  public/         scenes.json and the exported images under dreams/<scene>/ (generated)
schema/           scenes.schema.json, exported from the pipeline's manifest models
tools/annotate/   manual hotspot editor
docs/plans/       the plan
```

The committed placeholder scenes (soft gradients, one glowing shape per scene) let the site run
in a fresh clone with no API keys. Real scenes replace them once they are generated.

## Running the site

Needs Node 24 and pnpm (the version is pinned in `web/package.json`).

```sh
cd web
pnpm install
pnpm dev          # local dev server
pnpm build        # production build into web/dist
pnpm preview      # serve the production build
```

Checks: `pnpm lint` (ESLint and a Prettier check), `pnpm typecheck`, `pnpm test` (Vitest), and
`pnpm e2e` (Playwright against the built site, so run `pnpm build` first; the first time, also
run `pnpm exec playwright install chromium`). `pnpm format` rewrites files with Prettier.

GitHub Pages serves the site from `/dreamscapes/`, so the deploy workflow builds with
`GITHUB_PAGES=1 pnpm build`, which sets that base path. Without the variable the site is built
for the root path `/`. To test a Pages build locally:
`GITHUB_PAGES=1 pnpm build && GITHUB_PAGES=1 pnpm e2e`.

## Running the pipeline

Needs [uv](https://docs.astral.sh/uv/); it installs Python 3.12 itself. Copy `.env.example` to
`.env` and fill in the keys for the commands that call paid APIs. Commands find the repository
root by walking up from the current directory to the first directory that holds both `content/`
and `pipeline/pyproject.toml`, so they work from anywhere inside the checkout and never mistake
another project's `content/` folder for this one.

```sh
cd pipeline
uv sync
uv run dsp --help
```

| Command | What it does | Needs keys |
|---|---|---|
| `dsp validate [--strict]` | Checks the manifest, hotspot rules, the loop and every referenced file. Exits 1 on errors (or on warnings with `--strict`). | no |
| `dsp schema [--check]` | Writes `schema/scenes.schema.json` from the pipeline's models; `--check` only verifies it. | no |
| `dsp placeholders [--overwrite-masters] [--reset-manifest]` | Writes synthetic masters, masks, exports and `web/public/scenes.json`. Keeps real masters unless `--overwrite-masters`, and an existing `content/scenes.yaml` unless `--reset-manifest`; each flag destroys that kind of work. | no |
| `dsp generate --scene <id> \| --all [--candidates 4] [--provider gemini\|openai] [--no-reference] [--force] [--size 2K] [--quality high] [--dry-run]` | Builds each prompt from the style block in `content/style.md` and the scene, attaches the ungraded style-anchor master (except for the anchor itself), and writes `assets/candidates/<scene>/cand-<n>.png`. Skips existing candidates unless `--force`. While the anchor's master is still a placeholder, `--all` generates only the anchor and asks you to select it first. Logs every call to `pipeline/logs/generations.jsonl`. | yes |
| `dsp select --scene <id> \| --all [--repick] [--allow-upscale] [--dry-run]` | Claude scores the candidates against the rubric, writes `assets/review/<scene>/scores.json` and a contact sheet, records the winner in `content/scenes.yaml` (unless one is already pinned) and copies it to `assets/masters/<scene>.png`. The anchor is selected first, so the other scenes in the same run are judged against its new master. A winner smaller than 2048x1152 is refused unless `--allow-upscale`. | yes |
| `dsp grade [--preview]` | Applies the colour grade from `content/grade.toml` to every master, writing `<scene>.graded.png`; `--preview` writes before/after pairs to `assets/review/grade.html`. Placeholder masters keep their graded copy. | no |
| `dsp hotspot --scene <id> \| --all [--force] [--box x0 y0 x1 y1] [--dry-run]` | Segments the bridge object into a hotspot polygon, has Claude verify it, and writes it to `content/scenes.yaml`. `--dry-run` needs no keys. | yes |
| `dsp export [--check]` | Writes the AVIF/WebP variants, blurred previews and `web/public/scenes.json` from the graded masters, then verifies every file against its hash; `--check` only reports what would change. | no |
| `dsp review` | Rebuilds every contact sheet and `assets/review/index.html`, which shows all graded masters side by side. | no |
| `dsp morph --edge <a> <b> \| --all` | Renders morph clips between scenes. | yes |
| `dsp bakeoff [--candidates 4] [--force] [--quality high] [--dry-run]` | Generates the first scene with Gemini and OpenAI, grades every image, has Claude score them blind and writes `assets/review/bakeoff.html`. | yes |

Commands that call a paid API print their estimated spend first and refuse to go past
`DSP_MAX_SPEND_USD` (15 US dollars when unset or empty) in one run; `--dry-run` prints the exact
requests without sending anything. Rate limits and server errors are retried with backoff.
`morph` is not implemented yet; it exits with status 2. See section 10 of the
plan for every option.

### Hotspots

`uv run dsp hotspot --scene <id>` segments the scene's bridge object in its graded master (or
the master) with Gemini, cleans the outline (largest piece, grown by 0.5% of the width, at most
40 vertices), and asks Claude whether the tinted region is exactly that object. On a "no" it
retries once with the box Claude suggests; on a second "no" (or a "no" with no new box) it
sets `needs_manual_hotspot`, and `tools/annotate/index.html` is the way to draw it by hand. A
verified outline replaces the hotspot fields under `bridge:` and the mask
`assets/masks/<scene>.png`; a rejected one never does, so a failed run only marks the scene
unverified and keeps its polygon. Comments and everything else in `content/scenes.yaml` stay as
they were. Review overlays go to `assets/review/<scene>/`: `hotspot.png` is the hotspot the
manifest now holds, `hotspot-<n>.png` each attempt, and `hotspot.json` describes them.

`--dry-run` prints the exact Gemini and Claude requests and runs the rest of the pipeline on the
scene's committed mask. It sends nothing and leaves `content/scenes.yaml` and the masks alone,
but still writes the review files under `assets/review/` (git-ignored). `--all` skips scenes
that are already verified (add `--force` to redo them) and hand-drawn hotspots
(`source: manual`); naming a scene with `--scene` always redoes it.
`DSP_SEGMENTATION_PROVIDER=sam3` switches to SAM 3 on fal (needs
`uv sync --extra sam3` and `FAL_KEY`). Each run prints its spend estimate first and refuses to
go past `DSP_MAX_SPEND_USD` (default 15); a scene costs about $0.03, about $0.06 with the retry.

Checks: `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`
(`uv run ruff format .` to fix).

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request and on `main`: the pipeline job lints,
tests, checks the schema and runs `dsp validate`; the web job lints, typechecks, tests, builds
and runs the end-to-end tests. `.github/workflows/deploy.yml` publishes `web/dist` to GitHub
Pages on every push to `main`. Neither workflow calls a paid API.

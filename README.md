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
`pnpm build` also fails when the gzipped JavaScript passes its 30 KB budget, and `pnpm e2e` runs
every test twice: on a 1440x900 desktop and in Playwright's iPhone 13 profile.

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
| `dsp generate --scene <id> \| --all` | Generates candidate images. | yes |
| `dsp select --scene <id> \| --all` | Claude scores candidates and picks the master. | yes |
| `dsp grade [--preview]` | Applies the colour grade to every master. | no |
| `dsp hotspot --scene <id> \| --all` | Segments the bridge object into a hotspot polygon. | yes |
| `dsp export` | Writes the image variants and `scenes.json`. | no |
| `dsp review` | Rebuilds the contact sheets. | no |
| `dsp morph --edge <a> <b> \| --all` | Renders morph clips between scenes. | yes |
| `dsp bakeoff` | Compares image providers on the first scene. | yes |

Only `validate`, `schema` and `placeholders` are implemented so far; the others print
"not implemented yet" and exit with status 2. See section 10 of the plan for every option.

Checks: `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`
(`uv run ruff format .` to fix).

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request and on `main`: the pipeline job lints,
tests, checks the schema and runs `dsp validate`; the web job lints, typechecks, tests, builds
and runs the end-to-end tests. `.github/workflows/deploy.yml` publishes `web/dist` to GitHub
Pages on every push to `main`. Neither workflow calls a paid API.

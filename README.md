# Dreamscapes

A full-screen web page that shows one dreamlike, AI-generated scene at a time. Somewhere in each
scene one object is secretly clickable (the "bridge object"). Clicking it dissolves the current
scene into the next one, which is about that object: the distant lighthouse leads to floodlit
towers, a basketball hoop leads to a rooftop court, and so on. Ten scenes form a closed loop, so
the walk never ends. Everything is generated offline by a Python pipeline and committed; the
site is a static Vite + TypeScript page with no backend.

**Live site:** <https://blog.thiago.pub/dreamscapes/> (GitHub Pages, deployed from `main`).

Scenes change with the CSS dissolve by default. The pre-rendered morph clips are opt-in: the
faint "morph off" switch in the bottom-left corner turns them on ("morph on") and the browser
remembers the choice. A link can choose for the visitor with the `morph` query parameter, which
goes before the hash: <https://blog.thiago.pub/dreamscapes/?morph=1#/prison-yard> turns the clips
on, `?morph=0` turns them off, and either is remembered like a click on the switch. The order is:
the parameter, then the remembered choice, then the default (`DEFAULT_TRANSITION` in
`web/src/preference.ts`). With the dissolve, no clip is downloaded at all. Reduced-motion
settings always get a short crossfade instead, whatever the switch says.

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
  public/         scenes.json, the exported images under dreams/<scene>/ and the morph
                  clips under morphs/ (generated)
schema/           scenes.schema.json, exported from the pipeline's manifest models
tools/annotate/   manual hotspot editor
docs/plans/       the plan
```

The ten generated scenes, their exports and the morph clips are all committed, so the site runs
in a fresh clone with no API keys. `dsp placeholders` can still paint synthetic stand-ins (soft
gradients, one glowing shape per scene) for any scene that has no real master yet.

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

`pnpm walk <url>` checks a deployed copy, for example
`pnpm walk https://blog.thiago.pub/dreamscapes/`. Unlike `pnpm e2e`, which tests a local build,
it drives the deployed site, at 1440x900 and in the iPhone 13 profile. By default it walks with
the dissolve (`?morph=0`) and fails if any morph clip is requested; `--morph`, or `?morph=1` in
the URL it is given, walks with the real clips, and `--morph` wins if the two disagree. The
report says which mode it used and why, and what the page's morph switch showed. Each profile
starts fresh and dismisses the first-visit overlay. It clicks or taps the centroid of every
scene's hotspot polygon, and checks the address, the counter and the transition played at each
step, plus the return to the start after the last scene. Then it checks back, forward, a deep
link and an unknown hash. It prints every console error and failed request, and exits 1 if any
check fails. `--json <file>` saves the report and `--screenshots <dir>` saves one picture per
step. With the dissolve the walk takes under a minute; with the clips it takes about three,
because each clip plays for eight seconds.

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
| `dsp generate --scene <id> \| --all [--candidates 4] [--provider gemini\|openai] [--no-reference] [--force] [--size 2K] [--quality high] [--model id] [--dry-run]` | Builds each prompt from the style block in `content/style.md` and the scene, attaches the ungraded style-anchor master (except for the anchor itself), and writes `assets/candidates/<scene>/cand-<n>.png`. Skips existing candidates unless `--force`. While the anchor's master is still a placeholder, `--all` generates only the anchor and asks you to select it first. Logs every call to `pipeline/logs/generations.jsonl`. | yes |
| `dsp select --scene <id> \| --all [--repick] [--allow-upscale] [--dry-run]` | Claude scores the candidates against the rubric, writes `assets/review/<scene>/scores.json` and a contact sheet, records the winner in `content/scenes.yaml` (unless one is already pinned) and copies it to `assets/masters/<scene>.png`. The anchor is selected first, so the other scenes in the same run are judged against its new master. A winner smaller than 2048x1152 is refused unless `--allow-upscale`. | yes |
| `dsp grade [--preview]` | Applies the colour grade from `content/grade.toml` to every master, writing `<scene>.graded.png`; `--preview` writes before/after pairs to `assets/review/grade.html`. Placeholder masters keep their graded copy. | no |
| `dsp hotspot --scene <id> \| --all [--force] [--box x0 y0 x1 y1] [--dry-run]` | Segments the bridge object into a hotspot polygon, has Claude verify it, and writes it to `content/scenes.yaml`. `--dry-run` needs no keys. | yes |
| `dsp export [--check]` | Writes the AVIF/WebP variants, blurred previews and `web/public/scenes.json` from the graded masters, then verifies every file against its hash; `--check` only reports what would change. | no |
| `dsp review` | Rebuilds every contact sheet and `assets/review/index.html`, which shows all graded masters side by side. | no |
| `dsp morph --edge <a> <b> \| --all [--force] [--reencode] [--keep-stale] [--resolution 1080p] [--duration 8] [--model id] [--dry-run]` | Renders a Veo clip from scene A's graded master to scene B's (B must be A's `next`), strips the audio and re-encodes it with ffmpeg to `web/public/morphs/<a>__<b>.<hash>.mp4`, points A's transition at it and re-exports `scenes.json`. Skips edges whose clip is in place unless `--force`, or unless their graded masters changed since (then it renders again; `--keep-stale` keeps the old clip). `--reencode` rebuilds clips from the saved downloads without a new call. | yes |
| `dsp bakeoff [--candidates 4] [--force] [--quality high] [--dry-run]` | Generates the first scene with Gemini and OpenAI, grades every image, has Claude score them blind and writes `assets/review/bakeoff.html`. | yes |

Commands that call a paid API print their estimated spend first and refuse to go past
`DSP_MAX_SPEND_USD` (15 US dollars when unset or empty) in one run; `--dry-run` prints the exact
requests without sending anything. Rate limits and server errors are retried with backoff.
See section 10 of the plan for every option.

### Hotspots

`uv run dsp hotspot --scene <id>` segments the scene's bridge object in its graded master (or
the master) with Gemini, cleans the outline (largest piece, grown by 0.5% of the width, at most
40 vertices), and asks Claude whether the tinted region is exactly that object. On a "no" it
retries once with the box Claude suggests; on a second "no" (or a "no" with no new box) it
sets `needs_manual_hotspot`, and `tools/annotate/index.html` is the way to draw it by hand. A
verified outline replaces the hotspot fields under `bridge:` and the mask
`assets/masks/<scene>.png`; a rejected one never does, so a failed run only marks the scene
unverified and keeps its polygon. The same goes for an outline Claude accepts that breaks the
hotspot rules of `dsp validate` (too small, too big, off-centre): it is not kept, the scene is
marked unverified with the reason, and the summary line counts it. Comments and everything else
in `content/scenes.yaml` stay as they were. Review overlays go to `assets/review/<scene>/`: `hotspot.png` is the hotspot the
manifest now holds, `hotspot-<n>-<reading>.png` each attempt, and `hotspot.json` describes them.

Gemini does not always send its mask points in the documented `[x, y]` order: on half of the
first real scenes they came back as `[y, x]` while the bounding box kept its order. So every
mask is read all four ways (`xy-image`, `yx-image`, `xy-box`, `yx-box`: either order, 0-1000
across the image or across the box), and the reading whose outline stays inside its own box and
reaches across it wins, ties going to `xy-image`. The chosen reading is printed, named in the
overlay file and recorded in `hotspot.json` with every reading's score; a warning is printed
when even the best reading leaves mask points outside the box (some masks switch order
part-way round, and those outlines are usually garbled).

`--dry-run` prints the exact Gemini and Claude requests and runs the rest of the pipeline on the
scene's committed mask. It sends nothing and leaves `content/scenes.yaml` and the masks alone,
but still writes the review files under `assets/review/` (git-ignored). `--all` skips scenes
that are already verified (add `--force` to redo them) and hand-drawn hotspots
(`source: manual`); naming a scene with `--scene` always redoes it. A verified hotspot records
what it was checked against: the SHA-256 of the master (`verified_master_sha256`) and the
picture actually segmented and shown to Claude, normally the graded copy (`verified_image`,
`verified_image_sha256`). When either has changed since (say `dsp select` adopted another
candidate, or the grade was retuned), `--all` treats the scene as unverified and redoes it. A
scene whose graded copy is older than its master is refused with an error asking for
`dsp grade` first, because the outline would be traced on the old picture.
`DSP_SEGMENTATION_PROVIDER=sam3` switches to SAM 3 on fal (needs
`uv sync --extra sam3` and `FAL_KEY`). Each run prints its spend estimate first and refuses to
go past `DSP_MAX_SPEND_USD` (default 15); a scene costs about $0.03, about $0.06 with the retry.
Every segmentation and verification call appends a row to `pipeline/logs/generations.jsonl`
(`kind` `segment` or `verify`, `purpose` `hotspot`), so hotspot spend is counted with the rest;
a call whose answer reported no token usage has a null `cost_usd` and counts at its estimate.

Known limits of reading the mask points, all because only Gemini's box is used to judge, never
the image:

- A mask that switches order part-way round the outline cannot be read whole; it gets a warning
  and usually a garbled outline that Claude rejects.
- When a near-square box straddles the diagonal x = y, an outline and its mirror image both fit
  it, so `xy-image` and `yx-image` can tie. `xy-image` is used and a warning says to check the
  overlay.
- The two box-relative readings always tie, so a `[y, x]` box-relative mask would be read as
  `[x, y]`. Gemini has not been seen to send box-relative points.

### Morph clips

`uv run dsp morph --all` gives every edge of the loop a pre-rendered transition: Veo 3.1 Fast
(`veo-3.1-fast-generate-preview`) animates from the outgoing scene's graded master (first
frame) to the incoming one's (last frame), at 16:9, 1080p and 8 seconds, with the style block
plus a short "drift toward the bridge object and dissolve into the next scene" prompt. It costs
$0.12 per second of video, $0.96 per clip. Veo always adds a soundtrack, so ffmpeg (needed on
the PATH: `brew install ffmpeg`) drops it and re-encodes the video as H.264 (CRF 23, yuv420p,
`+faststart`); a clip that is over 5 MB at 1080p is scaled to 720p before the quality is
lowered. Every 1080p encode so far came out between 6 and 9 MB, so all nine committed clips are
1280x720 (1.8 to 3.9 MB each). The raw download is kept locally in `assets/candidates/morphs/`,
so a clip whose post-processing failed is finished on the next run without paying again. The
site plays the clips only for visitors who switch them on (see the top of this README), and
falls back to the zoom whenever a clip fails or is slow to load. Reviewing, re-rendering
and rejecting clips are covered in
[How to re-render or reject a morph clip](#how-to-re-render-or-reject-a-morph-clip).

A call that fails after Veo accepted it is still counted in the generation log and against
the spend limit: a video that finished is billed even if its download failed, and one still
rendering when the command gave up may finish and be billed later. Those rows carry the
operation name and `retrievable_until`, the moment Google deletes the video (two days on).

Checks: `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`
(`uv run ruff format .` to fix).

## How to regenerate a scene

Run these from `pipeline/` with the keys in `.env`. Every paid step prints its estimate first,
and `--dry-run` shows the requests without sending them.

1. Optionally improve the prompt first: edit the scene's `scene` description or its
   `bridge.placement` in `content/scenes.yaml` (in v1 the usual reason to regenerate was a
   bridge object that came out under 1.5% of the frame, fixed by asking for it nearer and
   bigger).
2. `uv run dsp generate --scene <id> --force` replaces the scene's candidates in
   `assets/candidates/<id>/` (four by default; `--candidates 6` for more). The style anchor's
   ungraded master is attached as the style reference, as before. Gemini at 2K costs about
   $0.06 per candidate.
3. `uv run dsp select --scene <id> --repick` has Claude score the new candidates (about $0.08)
   and copies the winner to `assets/masters/<id>.png`. `--repick` is required here: the scene
   already has a `generation.selected_candidate` from its first run, and without the flag
   `select` copies that candidate number from the new set without scoring anything. To choose
   by eye instead, look at the candidates, set `generation.selected_candidate` to your pick and
   run `uv run dsp select --scene <id>`, which copies it without a Claude call.
4. `uv run dsp grade` writes the new `<id>.graded.png`. The grade is deterministic, so the other
   scenes' graded copies come out unchanged.
5. `uv run dsp hotspot --scene <id>` traces the bridge object on the new picture, has Claude
   check the outline, and replaces the polygon and the mask. Look at
   `assets/review/<id>/hotspot.png`. If the scene ends up `needs_manual_hotspot`, draw the
   outline with `tools/annotate/index.html` (see its README) and paste it into the YAML.
6. `uv run dsp validate`, then `uv run dsp export`, which writes the new AVIF and WebP files
   under new hashed names, removes the old ones and updates `web/public/scenes.json`.
7. Re-render the two morph clips that touch the scene: the clip arriving from the previous scene
   ends on the old picture, and the clip leaving for the next scene starts on it.
   `uv run dsp morph --all --dry-run` lists them (it compares the graded masters each clip was
   made from with the current ones), and `uv run dsp morph --all` renders them at $0.96 each.
   Review them as described in
   [How to re-render or reject a morph clip](#how-to-re-render-or-reject-a-morph-clip).
   `--keep-stale` keeps the old clips instead, at the price of a visible jump between the clip
   and the new still.
8. `uv run dsp review`, then `cd ../web && pnpm build && pnpm e2e`. Commit
   `content/scenes.yaml`, `assets/masters/`, `assets/masks/`, `web/public/` and
   `pipeline/logs/generations.jsonl`.

Regenerating the style anchor (`style_anchor` in `content/scenes.yaml`, the cliff cottage) also
changes the reference image every other scene was generated from. The other scenes keep their
masters (each has its candidate pinned), so check `assets/review/index.html`, which shows all
the graded masters side by side, and regenerate any scene that no longer fits the set.

## How to add a scene

The loop is the chain of `next` pointers in `content/scenes.yaml`. A new scene goes between two
neighbours: A, and B, which is A's `next` today.

1. **Write the entry** in `content/scenes.yaml`, following the existing ones:
   - `id`: a new slug. `title`, and `alt`, which describes the scene and never says where the
     object is.
   - `scene`: 60-120 words for the image prompt. Show A's bridge object again in some form
     (`dsp validate` warns when the description does not mention it), and name two or three
     distractor objects. No people, faces, legible text, signs or logos.
   - `bridge.object`: a short noun phrase that also works as a segmentation prompt
     ("telephone box").
   - `bridge.placement`: where the object sits in the frame and how big it is.
   - `next: B`. Leave out `transition` (it defaults to the zoom), `polygon` and `generation`;
     the pipeline fills them in.
2. **Respect the placement rules** (plan section 4.4), which `dsp validate` enforces once the
   hotspot exists. There is exactly one bridge object, and it is not the main subject. Its
   outline covers 1.5% to 12% of the frame. Its centroid lies in the central 60% of the width
   (x 0.2-0.8) and the central 80% of the height (y 0.1-0.9). Ask for the object at a size and
   position inside those bounds, such as "medium, just right of centre, mid-distance".
3. **Re-point the loop.** Set A's `next` to the new id. B's description should mention the new
   scene's bridge object (the same warning applies): pick an object B already shows, or edit
   B's description and regenerate B. If A's `transition` is a clip (`type: video`), set it back
   to `type: zoom` (remove its `video` and `duration` lines) and delete A's old clip,
   `web/public/morphs/<A>__<B>.<hash>.mp4`: it morphs into B, not into the new scene.
4. **Placeholders**, if the branch must stay green before the real images exist. Until then
   `dsp validate` fails on the missing master and `dsp export` cannot build `scenes.json`. Add
   a stand-in for the new id to `PLACEHOLDER_SCENES` in
   `pipeline/dreamscapes_pipeline/placeholders.py`. That is a
   `PlaceholderScene(id, title, scene, bridge_object, placement, next, design=Design(...))`
   whose `Design` draws a shape that obeys the rules above. Put the entry between A's and B's,
   and re-point A's entry and the new entry exactly as in step 3. The table must stay in loop
   order with its `next` pointers forming one loop, because `dsp placeholders --reset-manifest`
   rebuilds the manifest from it. The pipeline's test fixtures are also built from this table,
   and the tests take their scene counts and the loop's last scene from it, so they follow
   along. Then run `uv run dsp placeholders` (it paints only scenes that have no real master)
   and `uv run dsp export`. Otherwise keep the work on a branch until step 5 is done.
5. **Run the pipeline** for the new scene, as for regenerating but without `--force` or
   `--repick`: `uv run dsp generate --scene <id>`, `uv run dsp select --scene <id>`,
   `uv run dsp grade`, `uv run dsp hotspot --scene <id>`, `uv run dsp validate` and
   `uv run dsp export`.
6. **Morph clips** for the two new edges: `uv run dsp morph --edge <A> <id>` and
   `uv run dsp morph --edge <id> <B>`. `uv run dsp morph --all` also works. It renders the
   edges that have no clip and were never rejected. It also re-renders any clip whose graded
   masters changed since it was made, such as the clips next to a regenerated B. Like every paid
   command it prints the estimated spend first, so run it with `--dry-run` to see which edges
   it picks. Review the clips as below.
7. `uv run dsp review`, then `cd ../web && pnpm build && pnpm e2e`. The end-to-end tests read
   the loop from `scenes.json`, so they walk the longer loop without changes. Commit the same
   paths as for a regenerated scene.

## How to re-render or reject a morph clip

Every clip needs a look before it ships, because Veo can add motion, people or lettering that
break the mood. In v1 all three takes of mountain-road-phonebox -> drowned-street wrote garbled
lettering on the phone box or the shopfronts, which is why that edge keeps the zoom. After any render, open
`assets/review/morphs/index.html`. Each edge gets a row: the two masters with the clip's first,
middle and last frames between them, an eight-frame contact sheet and the clip itself. It also
says how far the end frames are from the masters (mean absolute difference, 0-255; the shipped
clips score 2.4 to 10).

- **Re-render one clip:** `uv run dsp morph --edge <a> <b> --force` (B must be A's `next`).
  It pays for a new render ($0.96), writes the clip under a new hashed name, removes the old
  file and updates `content/scenes.yaml` and `scenes.json`. Without `--force`, an edge whose
  clip is in place and whose masters have not changed is skipped.
- **After a master changed:** `uv run dsp morph --all` re-renders exactly the clips whose first
  or last graded master changed since they were made, with a warning for each; `--keep-stale`
  keeps them.
- **Re-encode without paying:** `uv run dsp morph --all --reencode` (or
  `--edge <a> <b> --reencode`) rebuilds clips from the saved Veo downloads after a change to
  the ffmpeg settings. Those downloads live in `assets/candidates/morphs/`, which is
  git-ignored, so this only works on the machine that rendered them.
- **Reject a clip:** in `content/scenes.yaml` set the scene's `transition` back to `type: zoom`
  (delete its `video` and `duration` lines), delete the clip from `web/public/morphs/`, and run
  `uv run dsp export`. From then on `--all` leaves that edge alone: it neither re-encodes the
  saved download nor pays for a new render. To try that edge again later, name it:
  `uv run dsp morph --edge <a> <b>`.

Then run `cd ../web && pnpm build && pnpm e2e`, and commit `content/scenes.yaml`,
`web/public/` and `pipeline/logs/generations.jsonl`.

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request and on `main`: the pipeline job lints,
tests, checks the schema and runs `dsp validate`; the web job lints, typechecks, tests, builds
and runs the end-to-end tests. `.github/workflows/deploy.yml` publishes `web/dist` to GitHub
Pages on every push to `main`. Neither workflow calls a paid API.

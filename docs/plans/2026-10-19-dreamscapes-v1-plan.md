# Dreamscapes v1 — architecture and implementation plan

Date: 2026-10-19
Owner: Thiago (product decisions); orchestrated by the main Claude session; implemented by Opus subagents.
Status: approved by Thiago on 2026-10-19 (see §15); v1 shipped the same day (see "Status" below). Section 2 lists every decision taken on Thiago's behalf.

---

## Status (2026-10-19)

**v1 is live at <https://blog.thiago.pub/dreamscapes/>.** The repository is public. The account's user site owns the custom domain, so this project is served under it at `/dreamscapes/`, deployed by `.github/workflows/deploy.yml` on every push to `main`.

What shipped:
- Ten generated scenes in a closed loop (Gemini `gemini-nano-banana-2.1` at 2K, every scene anchored on `cliff-cottage`, picked by Claude, colour-graded), exported as AVIF and WebP at four widths with blurred previews.
- A verified hotspot on every scene. Nine were traced by Gemini segmentation and confirmed by Claude; the bicycle was drawn by hand in `tools/annotate` and confirmed by Claude.
- Nine Veo 3.1 Fast morph clips. The tenth edge keeps the CSS zoom.
- The static site: vanilla TypeScript, 7.6 KB of gzipped JavaScript, unit tests and Playwright e2e at desktop and iPhone 13 sizes. CI runs the pipeline and web checks on both base paths.
- Checked on the live site on 2026-10-19 with `pnpm walk` (in `web/`), at 1440x900 and in the iPhone 13 profile. All ten steps advanced the hash and counter. The nine clip edges each played their clip to 8.0 s, and the zoom edge zoomed. The loop closed at "dream 11". Back, forward, a deep link and an unknown hash behaved, with 0 console errors and 0 failed requests. Lighthouse (12.8 and 13.5) on the live site scored 100 in performance, accessibility, best practices and SEO, on both the desktop and mobile presets. The only sizeable flag is cache lifetime: GitHub Pages serves everything with `max-age=600`, which cannot be changed there.

Merged pull requests (all in `thiagorobert/dreamscapes`):

| PR | Title |
|---|---|
| #1 | Scaffold the repository: manifest models, pipeline CLI, placeholder scenes, web app shell and CI |
| #2 | Write the scene loop, style bible and colour grade |
| #3 | Build the web app: stage, hotspot, transitions, hints, routing and tests |
| #4 | Add hotspot extraction: Gemini segmentation, Claude verification and a manual annotator |
| #5 | Add the generation pipeline: providers, generate, select, grade, export, review and bakeoff |
| #6 | Produce the ten real scenes |
| #7 | Fix Gemini mask point order and verify the five pending hotspots |
| #8 | Add Veo morph clips between scenes (dsp morph) |
| #9 | Finish v1: live-site walk, Lighthouse check, README how-tos and plan status (this section) |

Main deviations from this plan:
- **Gemini's Interactions API instead of `generate_content`**, for both image generation (#5) and segmentation (#4), because that is the shape the current Gemini docs describe. Segmentation uses `gemini-3.8-flash`.
- **Segmentation masks sometimes come back as `[y, x]`**, not the documented `[x, y]`, while `box_2d` keeps its order. That happened on five of the ten scenes. `dsp hotspot` now reads every mask four ways and keeps the reading that fits its own box (#7). Masks that switch order part-way round remain a known limit, which is why the bicycle was drawn by hand.
- **Clips ship at 720p, not 1080p.** Veo renders at 1080p, but every 1080p encode came out at 6-9 MB, over the 5 MB per-clip cap, so ffmpeg scales them to 1280x720 (1.8-3.9 MB each). They are H.264 MP4 only; no WebM is made (#8).
- **One edge keeps the zoom:** mountain-road-phonebox -> drowned-street. All three Veo takes wrote garbled lettering on the phone box or the shopfronts (#8).
- **`transition.duration` is required** whenever `transition.type` is `video` (the example in §8.2 had only the path). `dsp morph` measures it with ffprobe.
- **`bridge` is a single object**, not the list §2 anticipated for branching. A list remains a schema change for later.
- **Extra bridge verification fields:** `verified_master_sha256`, `verified_image` and `verified_image_sha256` record what a hotspot was checked against. `dsp hotspot --all` redoes a scene when its master or graded image changes (#7). The built `scenes.json` and its schema are unchanged.
- Smaller ones: Vite 8 and TypeScript 6 instead of Vite 6 (#1); Pillow 13 encodes AVIF natively, so no AVIF plugin was added (#1).

Spend: **$20.55** across the 129 calls in `pipeline/logs/generations.jsonl`. Every row reports its own `cost_usd`, so no estimate was needed:

| Item | Calls | Cost |
|---|---|---|
| Veo morph clips (ten edges plus five retakes, $0.96 each) | 15 | $14.40 |
| Gemini scene images | 64 | $4.02 |
| Claude candidate scoring | 15 | $1.24 |
| Bake-off: 4 Gemini and 4 OpenAI images, plus 1 Claude scoring call | 9 | $0.55 |
| Claude hotspot verification | 13 | $0.28 |
| Gemini segmentation | 13 | $0.04 |

Two earlier amounts are not in the log, which did not record them at the time: about $1.06 of hotspot calls during #6, before hotspot calls were logged, and about $0.10 of smoke-test calls in #5. That makes about $21.70 in all, against the ≈ $20 that §5.7 estimated for v1 plus the optional clips.

---

## 1. Summary

Dreamscapes is a full-screen web page that shows one dreamlike, AI-generated scene at a time. Somewhere in each scene one object is secretly clickable (the "bridge object"). Clicking it dissolves the current scene into the next one, which is thematically about that object. Ten scenes form a closed loop, so the walk never ends.

Architecture in one paragraph: everything is generated **offline** by a Python pipeline (images, colour grade, hotspot polygons, optional morph videos, a JSON manifest) and committed to the repo. The site itself is a **static** Vite + TypeScript page with no backend, deployed to static hosting by CI. There is no runtime AI, no database, and no server to operate. The pipeline uses Google's Gemini image model as the default generator (with an adapter so OpenAI or FLUX can be swapped in), Gemini's built-in segmentation for hotspot polygons (SAM 3 on fal as an optional sharper alternative), Claude (Opus 5.5) as the judge that picks the best candidate image and verifies hotspots, and optionally Veo 3.1 to pre-render first-frame-to-last-frame morph clips for the transitions.

Three things make the "feel" consistent across images: a fixed style block in every prompt, the first approved image passed to every later generation as a style reference, and a deterministic colour grade applied in the pipeline to every final image.

---

## 2. Decisions on unspecified or ambiguous aspects

Each row is a default chosen so implementation can start. Thiago can overrule any of them by editing this table before the matching work package starts.

| Topic | Decision | Why | Change it if |
|---|---|---|---|
| Runtime infrastructure | None. Static site, assets committed to the repo, offline pipeline. | 10 images and a loop need no server. Zero ops, zero monthly cost, trivially cacheable. | A future "generate the next dream live" mode is wanted (see §9.5). |
| Hosting | GitHub Pages deployed by GitHub Actions from `main`. | Free, already have `gh`, no new accounts. Output is a plain static bundle, so moving to Cloudflare Pages or S3+CloudFront later is a one-file change. | Thiago prefers the company AWS account or a custom domain with Cloudflare. |
| Source control | New **private** GitHub repo, one PR per work package, CI must be green before merge. | Matches Thiago's standing PR workflow (branch from main, CI watch, one fresh-context review). | Thiago wants local-only; then branches are merged locally by the orchestrator. |
| Image generator | `gemini-nano-banana-2.1` at 2K, 16:9, with style-reference images. A short bake-off against OpenAI `gpt-image-2.5-flare` on scene 1 decides the final choice. | Cheapest high-quality option with native 16:9 and up to 14 reference images; one Google key also covers Veo for morph clips. | The bake-off shows OpenAI or FLUX produces a clearly better dreamy look. The provider adapter makes the switch a config change. |
| Segmentation for hotspots | Gemini's built-in segmentation (text prompt → polygon mask, same `GEMINI_API_KEY`) by default; fal's SAM 3 as an optional second provider. | No new account needed and Gemini returns polygons directly. SAM 3 is sharper if ever needed (key from fal.ai/dashboard/keys). | Gemini polygons prove too loose on several scenes; then add `FAL_KEY` and switch `DSP_SEGMENTATION_PROVIDER=sam3`. |
| Master image format | 16:9 landscape, 2048×1152 PNG master, exported to AVIF + WebP at 2048/1536/1024/640 widths plus a tiny blurred placeholder. | Landscape fits most screens; multiple widths keep mobile fast. | Portrait companions are wanted for phones (v2 idea in §3.6). |
| Mobile portrait | The image is shown "cover" with the hotspot centroid as the focal point, and the user can drag/swipe horizontally to pan. | A 16:9 image shows only ~32% of its width on a 9:16 phone. Panning turns the crop into exploration instead of a defect, and the focal point guarantees the hotspot is reachable. | Thiago prefers letterboxing or portrait renders. |
| Hotspot count | Exactly one per scene in v1. Schema allows a list so branching can be added later. | Matches the brief; keeps the authoring and validation simple. | Branching dreams are wanted. |
| Hotspot discoverability | Cursor becomes a pointer over the hotspot (desktop). No glow on hover. A wrong click shows a soft ripple. After 20 s idle or 3 misses the hotspot shimmers once. A first-visit overlay says "Something in each dream leads to the next. Find it." | "Not obvious" but not frustrating. Touch devices have no hover, so the miss ripple and idle shimmer carry the hint there. All thresholds are constants in one file. | Thiago wants it harder (no pointer cursor) or easier (hover glow). |
| Transition | v1: CSS "fall into the object": the current image zooms toward the hotspot and dissolves while the next image settles in. v2 (the "morph" package, in scope for this round): pre-rendered Veo 3.1 morph clip from image A to image B, with the CSS transition as fallback. | CSS version costs nothing, works offline, always matches the images exactly. Morph clips are the higher-wow option but cost ~$1 per edge and need review. | — |
| On-screen chrome | Minimal: a faint "dreamscapes" wordmark top-left, a small "dream N" depth counter bottom-right, nothing else. No visible next/back buttons. | The image is the product. Keyboard and screen-reader users still get a visually hidden "continue to the next dream" control. | Thiago wants a fully bare screen or more UI. |
| URL / deep links | Hash routes by scene slug (`/#/prison-yard`). Browser back returns to the previous dream. Landing with no hash shows scene 1. | Shareable, works on static hosting without rewrite rules. | — |
| Sound | None. Morph videos are muted. | Autoplay with sound is blocked by browsers anyway. | — |
| Analytics / telemetry | None. | Private toy; no need. | — |
| Content constraints | No legible text, no logos, no recognisable people or faces, no gore; empty, quiet scenes. | Generators garble text; faces break the mood; keeps it safe to share. | — |
| Reproducibility | Every generation call is logged (provider, model, full prompt, parameters, seed when the API has one, response ids). Candidates are kept locally; masters and exports are committed. | Image APIs are not deterministic, so the committed output is the source of truth and the log explains how it was made. | — |
| Who picks the best candidate | Claude (Opus 5.5, vision) scores candidates against a rubric and writes a contact sheet; the orchestrator (and Thiago) can override in the manifest. | Lets subagents run the whole pipeline unattended while keeping a human veto. | — |
| Languages | Python 3.12 managed by `uv` for the pipeline; TypeScript (Vite, vanilla DOM, no framework) with `pnpm` for the site. | Thiago's `uv` rule; the site is small enough that a framework adds nothing. | — |
| Browser support | Evergreen desktop browsers, iOS Safari and Android Chrome from the last two years. | AVIF + WebP + `<picture>` fallback covers all of them. | — |

---

## 3. Experience specification

### 3.1 Landing
The page opens on scene 1 (`cliff-cottage`) filling the viewport. A blurred low-resolution placeholder paints instantly and the full image fades in over it. A one-time overlay (dismissed by any click or tap, remembered in `localStorage`) explains the single rule. The wordmark and depth counter fade to ~40% opacity after two seconds.

### 3.2 Exploring
The user moves the pointer or taps around. Over the bridge object the cursor becomes a pointer. Clicking anywhere else triggers a soft, short ripple at the click point and nothing else. After 20 seconds without a successful click, or after 3 misses, the bridge object shimmers once (a 1.2 s soft luminance sweep clipped to the hotspot polygon). The idle timer resets after each shimmer and backs off (20 s, 40 s, 60 s) so it never nags.

### 3.3 Transition
Clicking the bridge object plays the transition (§7) and shows the next scene. The depth counter increments. The hash updates. The next-next image starts preloading as soon as the new scene is visible.

### 3.4 The loop
Scene 10's bridge object leads back to scene 1. The counter keeps counting ("dream 11"), making the loop feel infinite rather than reset.

### 3.5 Keyboard and screen readers
`Tab` reaches a visually hidden "Continue to the next dream" button so keyboard and screen-reader users are never stuck; `Enter` activates it. Each image has `alt` text describing the scene (not where the hotspot is). `prefers-reduced-motion` replaces the zoom with a plain 300 ms crossfade and disables the shimmer and ripple animations (hint still shows as a static highlight for 1 s).

### 3.6 Viewports
The stage element keeps the image's 16:9 ratio and is scaled to cover the viewport. On viewports narrower than 16:9 the stage overflows horizontally; its initial position centres the scene's focal point (default: hotspot centroid), and the user can drag or swipe to pan within bounds. The SVG hotspot overlay lives inside the stage with the same coordinate system, so no coordinate mapping is needed anywhere. v2 idea: generate 9:16 portrait companions per scene with their own polygons.

---

## 4. Content design

### 4.1 Style bible (the fixed part of every prompt)
The full text lives in `content/style.md` and is pasted verbatim into every generation request. Draft in Appendix B. Key properties: painterly realism (not cartoon, not photoreal), soft diffused light with mist, muted dusty-teal/lavender palette with warm amber light sources, gentle bloom, fine film grain, wide establishing shot at eye level, nobody in frame, no text or logos, "the moment after everyone has left".

### 4.2 Prompt template
```
{STYLE_BIBLE}

Scene: {scene_description}

Include {bridge_object_phrase}: {placement_hint}. It must be clearly
recognisable but not the main subject, occupying roughly 3-8% of the frame,
located within the central 60% of the image width. Nothing else in the
scene should resemble {bridge_object_noun}.
```
Modern generators do not take negative prompts, so constraints are phrased positively inside the prompt. When a style reference image is attached the request adds: "Match the colour palette, lighting, grain and rendering style of the reference image only. Do not reuse its composition, objects or layout."

### 4.3 The ten scenes (first draft of the loop)
Each next scene is *about* the previous bridge object. Distractor objects are deliberate.

| # | Slug | Scene (short) | Bridge object (clickable) | Leads to |
|---|---|---|---|---|
| 1 | `cliff-cottage` | Stone cottage on a sea cliff under a storm; waves below; a slender lighthouse glowing faintly on a far headland. Distractors: cottage, a small boat. | the distant lighthouse | 2 |
| 2 | `prison-yard` | A prison at night: floodlight towers washing an empty exercise yard in cold light, cracked basketball court, chain-link fences, fog. Distractors: floodlights, a guard tower. | the basketball hoop | 3 |
| 3 | `rooftop-court` | A rooftop basketball court at blue hour above a sleeping city; one ball resting at the free-throw line; strings of unlit bulbs; far off a lone hot-air balloon drifting between towers. | the hot-air balloon | 4 |
| 4 | `balloon-salt-flats` | A fleet of hot-air balloons hanging motionless over a mirror-flat salt lake at dawn, perfect reflections; on the flat stands a single free-standing wooden door with nothing behind it. | the door | 5 |
| 5 | `endless-library` | An impossibly tall library: ladders, dust in shafts of light, shelves spiralling upward; on a reading table a brass birdcage with its door open. Distractors: a globe, a lamp. | the birdcage | 6 |
| 6 | `overgrown-greenhouse` | A Victorian greenhouse reclaimed by plants, fogged panes, small birds roosting among ferns; leaning against a bench an old rusted bicycle with a wicker basket. | the bicycle | 7 |
| 7 | `mountain-road-phonebox` | A night mountain road; a bicycle leaning on the guardrail; at the hairpin a glowing red telephone box with moths circling its light; the valley lost in mist below. | the telephone box | 8 |
| 8 | `drowned-street` | A submerged city street seen from underwater: a payphone handset floating on its cord, fish drifting past shop windows, light shafts from above, and up at the surface the dark hull of a passing ship. | the ship's hull | 9 |
| 9 | `fog-ship-deck` | The deck of an old ship becalmed in thick fog at night; the wheel unattended; rigging fading into white; one paper lantern glowing amber in the shrouds. Distractors: the wheel, a bell. | the paper lantern | 10 |
| 10 | `lighthouse-gallery` | Inside the lamp room of a lighthouse at dusk, the great lens glowing; through the glass, across the bay, a tiny stone cottage on a cliff with one window lit. | the cottage | 1 |

Lantern → lighthouse lamp and cottage → scene 1 close the loop with a pleasing symmetry (scene 1's bridge was the lighthouse seen from the cottage; scene 10 is the cottage seen from the lighthouse).

### 4.4 Bridge-object rules (enforced by `dsp validate`)
- Exactly one bridge object per scene; a short noun phrase usable as a segmentation prompt (e.g. "basketball hoop").
- Polygon area between 1.5% and 12% of the frame; centroid inside the central 60% of the width and central 80% of the height.
- The object must not be the dominant subject (area check above handles most cases; Claude's selection rubric handles the rest).
- Scene N+1 must mention the bridge object of scene N in its description (string check, warns only).

---

## 5. Image generation

### 5.1 Provider choice
Researched on 2026-10-19 (sources in Appendix C):

| Provider / model | Why it fits | Caveats |
|---|---|---|
| Google `gemini-nano-banana-2.1` (default) | Native 16:9 and 21:9, 1K/2K/4K output, up to 14 reference images for style, ~$0.034 (1K) / ~$0.050 (2K) per image, same API key as Veo. | Output carries an invisible SynthID watermark (fine). Seed support must be checked at implementation time. |
| Google `gemini-3-pro-image` | Higher-end sibling, same API, ~$0.134 per image. | Use only if the bake-off shows nano-banana lacks detail. |
| OpenAI `gpt-image-2.5-flare` / `-sunburst` (bake-off contender) | Custom sizes (e.g. `2048x1152`, multiples of 16), reference images via the edits endpoint, quality up to `max`. | Token-billed (~$0.05–0.20 per image depending on quality); no seed. |
| Black Forest Labs FLUX.2 / FLUX 3 (optional) | Up to 10 references; open-weight tiers exist for self-hosting. | Endpoint IDs and pricing to verify on docs.bfl.ai; lowest priority. |

Rule: the `pipeline/dreamscapes_pipeline/providers/` package defines one `ImageProvider` protocol (`generate(prompt, aspect, size, references, seed) -> GenerationResult`) and ships `gemini.py` first, `openai_images.py` second, `flux.py` only if someone asks. The provider is selected per run with `--provider` or `DSP_IMAGE_PROVIDER`.

### 5.2 Bake-off (part of the "produce" work package)
Generate scene 1 four times with Gemini and four times with OpenAI using the identical prompt, grade all eight, write a contact sheet to `assets/review/bakeoff.html`, and have Claude score them with the rubric in §5.4. The orchestrator shows Thiago the sheet; Thiago picks the provider (default stays Gemini if they do not answer). Cost under $2.

### 5.3 Keeping the feel consistent (five levers, all used)
1. **Fixed style block** in every prompt (§4.1). Cheapest and most important lever.
2. **Same model, same settings** for every scene (aspect, size, quality). Never mix providers within a set.
3. **Style anchor image**: scene 1 is generated first and approved. Every later scene attaches the *ungraded* anchor master as a reference image with the "palette and rendering only" instruction. Only the anchor is used, never the previous scene, so drift cannot accumulate.
4. **Deterministic grade** applied by the pipeline to every master (§5.5). This guarantees a shared palette, grain and glow even where the generator wanders.
5. **Candidates and a judge**: 4 candidates per scene, scored by Claude against the rubric; the top one becomes the master. Scenes that fail the rubric are regenerated with a tweaked placement hint.

### 5.4 Selection rubric (Claude vision, structured output)
Each candidate gets 0–5 on: style match to the anchor; bridge object present, recognisable and sized/placed per §4.4; no text, logos, faces or obvious artefacts; mood (empty, quiet, dreamy); composition (would make a good landing image). Weighted sum; hard fail on text/faces or missing bridge object. Output is JSON validated by a schema, written to `assets/review/<scene>/scores.json`, and a contact sheet HTML is rendered alongside. Model: `claude-opus-5-5` via the Python `anthropic` SDK, following the `claude-api` skill's defaults (adaptive thinking, explicit effort, structured outputs, server-side fallbacks).

### 5.5 Deterministic grade ("the Dreamscapes look")
Implemented with Pillow + NumPy in `grade.py`, parameters in `content/grade.toml` so they can be tuned once and re-applied to everything:
1. Desaturate to 85%.
2. Lift blacks (0 → 18/255) and soften highlights (gentle shoulder).
3. Split-tone: shadows toward teal, highlights toward amber, low strength.
4. Bloom: luminance > 0.8 blurred with radius 2% of width, screen-blended at 25%.
5. Vignette: soft, 15% at the corners.
6. Film grain: Gaussian noise, σ ≈ 2.5/255, luminance only, fixed RNG seed per scene for reproducibility.
`dsp grade --preview` writes before/after pairs to `assets/review/grade.html` for tuning.

### 5.6 Reproducibility and logging
`pipeline/logs/generations.jsonl` (committed) records every call: timestamp, provider, model, scene, candidate index, full prompt, parameters, reference image hashes, seed if any, provider response id, output file hash, cost estimate. Candidates live in `assets/candidates/` (git-ignored, kept locally); masters in `assets/masters/` (committed); exports in `web/public/dreams/` (committed).

### 5.7 Cost (v1)
| Item | Estimate |
|---|---|
| 10 scenes × 4 candidates × ~$0.05 (Gemini 2K) | ~$2 |
| Bake-off (8 images incl. OpenAI) | ~$2 |
| Regenerations buffer | ~$2 |
| Hotspot segmentation (≤ 60 Gemini calls) | < $0.50 |
| Claude scoring/verification (~60 vision calls) | ~$2–3 |
| Hosting | $0 |
| **Total v1** | **≈ $10** |
| Optional morph clips (10 × 8 s Veo 3.1 Fast at 1080p ≈ $0.96 each) | ≈ $10 |

---

## 6. Hotspots: making one part of the image clickable

### 6.1 Representation
A hotspot is a polygon in **normalised image coordinates** (x, y in 0–1, origin top-left), with derived `bbox` and `centroid`. Normalised coordinates survive every resize and crop. Stored in the manifest; rendered as an SVG `<polygon>` overlay whose `viewBox` is `0 0 1 1` stretched over the stage, fill fully transparent, `pointer-events="all"`, plus a transparent 10-px stroke to enlarge the touch target.

### 6.2 Extraction (automatic)
`dsp hotspot --scene <id>`:
1. Ask Gemini (default provider, same key as generation) for the segmentation of the bridge object: a prompt such as "Give the segmentation mask for the {bridge object}" with a JSON-schema response containing `box_2d` (`[ymin, xmin, ymax, xmax]`, normalised 0–1000) and `mask` (a polygon of `[x, y]` points, 0–1000, relative to the box) and a `label`. Gemini's docs recommend minimal thinking for segmentation. Convert to normalised image coordinates.
2. If several instances come back, prefer the one inside the safe area with the largest area. Optional second provider: SAM 3 on fal (`fal-ai/sam-3/image`, needs `FAL_KEY`) returns raster masks, converted with OpenCV (`findContours` on the largest connected component).
3. Rasterise the polygon to a mask, keep the largest connected component, simplify with `approxPolyDP` to ≤ 40 vertices, dilate by ~0.5% of width so edges are forgiving, normalise.
4. Render a verification overlay (`assets/review/<scene>/hotspot.png`: mask tinted over the image) and ask Claude: "Is the highlighted region exactly the {bridge object} and nothing else?" (structured yes/no + reason). On "no", retry with a prompt that includes the bounding box Claude suggests; on second failure, mark the scene `needs_manual_hotspot`.
5. Write polygon, bbox, centroid, area, score and verification result to `content/scenes.yaml`.

### 6.3 Manual editor (fallback and override)
`tools/annotate/index.html` is a dependency-free local page: open a master image, click to place polygon vertices, drag to adjust, see the area/safe-area checks live, and copy the YAML snippet. Used by Thiago or the orchestrator when the segmenter picks the wrong thing, or to make a hotspot deliberately tighter.

### 6.4 Validation (`dsp validate`)
Polygon is simple (non-self-intersecting), ≥ 3 vertices, area and centroid within §4.4 limits, every scene has exactly one hotspot and a `next` that exists, the `next` graph is a single cycle covering all scenes, every referenced file exists with the expected dimensions.

---

## 7. Transitions: "transforming" one image into the next

### 7.1 v1 — CSS "fall into the object" (always available)
On activation, with the hotspot centroid as `transform-origin` for the outgoing image:
- Outgoing image: scale 1 → 1.6, blur 0 → 10 px, opacity 1 → 0, over 900 ms, ease-in (`cubic-bezier(.7,0,.84,0)`).
- Incoming image (already decoded): scale 1.12 → 1, blur 6 px → 0, opacity 0 → 1 over 900 ms, ease-out, starting 150 ms after the outgoing begins.
- A brief brightness bloom (white overlay 0 → 0.25 → 0, 500 ms) centred on the hotspot sells the "dissolving into light" feel.
- Pointer events are disabled during the transition; the hash and counter update at the end.
- Reduced motion: 300 ms crossfade, no scale or blur.
The incoming image must be fully loaded before the transition starts (preloading in §11.4); if it is not, show a subtle pulse on the hotspot and start when `decode()` resolves.

### 7.2 v2 — pre-rendered morph clips (optional work package "morph")
For each edge A→B, `dsp morph --edge A B` calls Veo 3.1 (`veo-3.1-fast-generate-preview`) with `image` = graded master A, `last_frame` = graded master B, 16:9, 8 s at 1080p (1080p requires 8 s), with a prompt like "slow dreamlike dissolve: the camera drifts toward the {bridge object}, which blooms into {scene B}". Audio is always generated and is stripped with ffmpeg (`-an`), the clip is encoded to H.264 MP4 (and WebM if size allows) and stored in `web/public/morphs/<A>__<B>.mp4`. The manifest edge gets `transition: {type: "video", src, duration}`.

Playback: a muted `<video>` sits above the stage, pre-seeked to 0, its first frame crossfaded in over 120 ms (Veo's first frame may differ slightly from A), plays, and on `ended` the stage already shows B underneath; crossfade the video out over 120 ms. If the video fails to load or is slower than 1 s to become playable, fall back to §7.1. Reduced motion always uses §7.1's crossfade.

Each clip is reviewed (the orchestrator views the first/middle/last frames) because Veo may invent motion that breaks the mood; rejected clips keep the CSS transition.

---

## 8. Storage and data model

### 8.1 Repository layout
```
dreamscapes/
  README.md
  .gitignore  .env.example  .github/workflows/{ci.yml,deploy.yml}
  docs/plans/2026-10-19-dreamscapes-v1-plan.md   (this file)
  content/
    style.md          style bible (fixed prompt block) + grade notes
    grade.toml        grade parameters
    scenes.yaml       the authored loop: prompts, bridge objects, hotspots, next pointers
  pipeline/           Python, uv-managed; `uv run dsp <command>`
    pyproject.toml
    dreamscapes_pipeline/
      cli.py manifest.py providers/{base,gemini,openai_images}.py
      generate.py select.py grade.py hotspots.py export.py morph.py
      review.py validate.py placeholders.py
    tests/
    logs/generations.jsonl
  assets/
    candidates/<scene>/cand-{n}.png      git-ignored
    masters/<scene>.png                  committed (ungraded) + <scene>.graded.png
    masks/<scene>.png                    committed
    review/                              git-ignored contact sheets and overlays
  web/                Vite + TypeScript, pnpm
    index.html  src/{main,stage,hotspot,transition,hints,router,preload,manifest,a11y}.ts
    src/styles.css  src/types.ts  public/scenes.json  public/dreams/<scene>/<scene>-{w}.{avif,webp}
    public/morphs/   tests/ (vitest)  e2e/ (playwright)
  tools/annotate/index.html
  schema/scenes.schema.json              exported from the pydantic models
```

### 8.2 Authored manifest (`content/scenes.yaml`), per scene
```yaml
- id: prison-yard
  title: The yard
  alt: A prison exercise yard at night under floodlights, fog drifting across a cracked basketball court.
  scene: >
    A prison at night ...
  bridge:
    object: basketball hoop          # noun phrase; also the segmentation prompt
    placement: small, right of centre, mid-distance
    polygon: [[0.61,0.42],[0.66,0.41], ...]   # filled by `dsp hotspot`
    source: sam3 | manual
    verified: true
  next: rooftop-court
  focal: [0.63, 0.45]                 # optional; defaults to hotspot centroid
  generation:
    provider: gemini
    model: gemini-nano-banana-2.1
    selected_candidate: 2
    log_ids: [ ... ]                  # rows in generations.jsonl
  transition:
    type: zoom                        # or video
    video: morphs/prison-yard__rooftop-court.mp4
```
Top level also holds `version`, `start: cliff-cottage`, and `style_anchor: cliff-cottage`.

### 8.3 Built manifest (`web/public/scenes.json`)
`dsp export` compiles the YAML into the JSON the site loads: same fields plus `image: {width, height, sources: {avif: [...], webp: [...]}, lqip: "data:image/webp;base64,..."}` and derived `bbox`/`centroid`. The pydantic models in `manifest.py` are the source of truth; `dsp schema` exports `schema/scenes.schema.json`, and a web unit test validates `scenes.json` against it with `ajv`. `web/src/types.ts` mirrors the schema by hand (small enough).

### 8.4 What is committed
Masters, masks, exports, morphs, logs, manifests: yes (≈ 60–120 MB for 10 scenes; acceptable, no LFS needed). Candidates and review sheets: no (git-ignored). Secrets: never (`.env` ignored; `.env.example` lists the variables).

---

## 9. Infrastructure

### 9.1 Runtime
Static files on GitHub Pages (or any static host). Cache headers are the host's defaults; filenames carry a content hash via Vite for JS/CSS, and image/video files are versioned by scene name plus a short hash of their bytes appended at export time, so updates never serve stale assets.

### 9.2 Build and deploy (GitHub Actions)
- `ci.yml` on every PR and on `main`: pipeline job (`uv sync`, `ruff check`, `ruff format --check`, `pytest`), web job (`pnpm install --frozen-lockfile`, `pnpm lint`, `pnpm typecheck`, `pnpm test`, `pnpm build`), e2e job (Playwright against the built site, desktop + mobile viewports), and `uv run dsp validate`.
- `deploy.yml` on push to `main`: build the site and publish `web/dist` to GitHub Pages.
- CI never calls paid APIs. The pipeline's unit tests use fakes; `dsp placeholders` generates synthetic images so the site and e2e tests work in a checkout with no keys.

### 9.3 Secrets and accounts
Local `.env` (never committed; already present in the directory with `GEMINI_API_KEY`, `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`): `GEMINI_API_KEY` (image generation, segmentation and Veo), `ANTHROPIC_API_KEY` (Claude), `OPENAI_API_KEY` for the bake-off, optional `FAL_KEY` only if the SAM 3 provider is enabled. GitHub needs no secrets beyond the default token for Pages.

### 9.4 Local development
`pnpm dev` in `web/` serves the site with the committed assets; `uv run dsp placeholders` fills `web/public` with synthetic scenes (gradient backgrounds, one coloured shape per scene as the hotspot) for anyone without keys. `tools/annotate` is opened directly in the browser.

### 9.5 Future: live generation ("truly infinite")
Not in v1. The design keeps it possible: scenes are a graph in the manifest, not an array, and the site resolves `next` by id. A later mode would add a small edge worker (Cloudflare Worker or Lambda) that, given the current scene's bridge object, generates scene N+1 on demand, runs segmentation and the grade, stores the result in object storage, and returns a manifest fragment; the site would fetch it instead of reading `scenes.json`. That mode needs a cache keyed by (scene, bridge object), rate limiting, cost caps and moderation, which is why it is explicitly deferred.

---

## 10. Pipeline CLI specification (`uv run dsp …`)

| Command | What it does | Needs keys |
|---|---|---|
| `dsp validate` | Runs every check in §6.4 and §4.4; non-zero exit on errors; warnings for soft rules. | no |
| `dsp schema` | Writes `schema/scenes.schema.json` from the pydantic models. | no |
| `dsp placeholders` | Writes synthetic masters, masks, exports and `scenes.json` so the site runs with no keys. | no |
| `dsp generate --scene <id> [--all] [--candidates 4] [--provider gemini] [--no-reference]` | Builds the prompt from style bible + scene, attaches the anchor reference unless disabled or the scene *is* the anchor, writes candidates and log rows. Idempotent: skips existing candidates unless `--force`. | yes |
| `dsp select --scene <id> [--all]` | Claude scores candidates, writes `scores.json` + contact sheet, sets `selected_candidate` unless already pinned (`--repick` to override). Copies the winner to `assets/masters/<id>.png`. | yes |
| `dsp grade [--preview]` | Applies the grade to every master → `<id>.graded.png`; preview writes before/after HTML. | no |
| `dsp hotspot --scene <id> [--all] [--box x0 y0 x1 y1]` | Gemini segmentation (or SAM 3) → polygon, Claude verification, writes YAML, overlay PNG. | yes |
| `dsp export` | Produces AVIF/WebP variants, LQIP, `scenes.json`; verifies hashes. | no |
| `dsp review` | Rebuilds every contact sheet and an index page at `assets/review/index.html`. | no |
| `dsp morph --edge <a> <b> [--all]` | v2: Veo first/last-frame clip, ffmpeg post-processing, manifest update. | yes |
| `dsp bakeoff` | §5.2. | yes |

Shared behaviours: every command prints what it will spend before calling a paid API and refuses to exceed `DSP_MAX_SPEND_USD` (default 15) per run; `--dry-run` prints the requests without sending; providers retry with backoff on 429/5xx; all paths are relative to the repo root found by walking up from the CWD.

---

## 11. Web application specification

### 11.1 Stack
Vite 6, TypeScript strict, vanilla DOM, no runtime dependencies. ESLint + Prettier. Vitest for units, Playwright for e2e. Bundle target: < 30 KB JS gzipped.

### 11.2 Modules
- `manifest.ts`: fetch and validate `scenes.json` (light runtime checks), build a `Map<id, Scene>`.
- `stage.ts`: owns the stage element; computes cover scale and focal-point offset from viewport size; handles drag/swipe panning with bounds; exposes `showScene(scene)`.
- `hotspot.ts`: renders the SVG overlay for the current scene; emits `activate` on click/tap/Enter; exposes `shimmer()` for hints; sets `cursor: pointer`.
- `transition.ts`: implements §7.1 and (if `transition.type === "video"`) §7.2 with fallback; respects reduced motion.
- `hints.ts`: miss ripple, idle timer with back-off, miss counter, first-visit overlay. Constants in `hints.ts` header.
- `router.ts`: hash ↔ scene id; `popstate` handling; initial scene resolution; unknown hash → start scene.
- `preload.ts`: picks the right `srcset` candidate for the viewport, creates an `Image`, awaits `decode()`, keeps a small LRU of decoded images for next and next-next.
- `a11y.ts`: hidden continue button, `aria-live` announcement of the new scene title, focus management.
- `main.ts`: wires everything; `depth` counter in `sessionStorage` so a reload keeps the count.

### 11.3 Stage geometry (the only maths in the app)
Given viewport (vw, vh) and image ratio r = 16/9: if vw/vh ≥ r the stage is vw × vw/r, else vh·r × vh. Offset x = clamp(vw/2 − focal.x·stageW, vw − stageW, 0), same for y. Unit tests cover the four cases (wide, tall, exact, panning bounds).

### 11.4 Preloading order
On showing scene S: ensure S's full image is decoded (LQIP shown meanwhile), then preload next(S), then next(next(S)), then the morph clip for S→next(S) if present (`<link rel="preload" as="video">` is unreliable; use a muted `video.preload = "auto"` element kept off-screen).

### 11.5 Tests
Vitest: stage geometry, point-in-polygon (sanity, SVG handles hits natively), router parsing, hint state machine (fake timers), manifest validation against the exported schema.
Playwright (desktop 1440×900 and iPhone 13 viewport): page loads with scene 1; a click outside the polygon does not change the hash; a click at the polygon centroid changes the hash to `next`; ten activations return to scene 1 with counter 11; back button returns to the previous scene; keyboard Tab + Enter advances; reduced-motion emulation still navigates; placeholders mode runs the whole loop in CI.

---

## 12. Testing and quality gates

- Pipeline: pytest with fake providers (fixtures return deterministic PNGs), golden tests for the grade (hash of output for a fixed input and seed), geometry tests for mask → polygon, manifest round-trip and validation tests, CLI smoke tests via `typer`'s test runner. `ruff` for lint and format.
- Web: §11.5. Lighthouse performance ≥ 90 on the built site (manual check by the orchestrator using the chrome-devtools tools).
- Visual QA by the orchestrator after the "produce" package: view every graded master, every hotspot overlay and every contact sheet; walk the full loop in a real browser at desktop and phone sizes; confirm no scene is trivially solvable or impossible.
- Definition of done for v1: ten real scenes, loop closes, CI green, site deployed, Thiago has walked the loop once.

---

## 13. Work packages and orchestration

### 13.1 Roles
- **Orchestrator**: the main Claude session (this one). Spawns implementers, reviews their results, merges, runs visual QA, talks to Thiago. Does not write product code itself.
- **Implementers**: `Agent` tool, `subagent_type: "general-purpose"`, `model: "opus"`, one per work package, each with `isolation: "worktree"` once the repo has its first commit so parallel packages never touch the same checkout. Each gets the brief from Appendix A plus the pointer to this file.
- **Reviewer**: after each PR's CI is green, exactly one fresh `general-purpose` agent runs `/review` on the PR, per Thiago's standing rules; the orchestrator triages its findings.

### 13.2 Packages and dependencies
```
scaffold ──┬── content ─────────┐
           ├── pipeline ────────┤
           ├── hotspots ────────┼── produce ──┬── qa-and-deploy
           └── web ─────────────┘             └── morph
```
| Package | Branch | Produces | Blocks on |
|---|---|---|---|
| scaffold | `scaffold-repo` | git init, GitHub repo, layout, tooling, CI skeleton, pydantic manifest models + schema, CLI stub, `dsp placeholders`, README | — |
| content | `author-scene-loop` | `content/style.md`, `content/grade.toml`, `content/scenes.yaml` with the ten scenes fully written | scaffold |
| pipeline | `add-generation-pipeline` | providers, generate, select, grade, export, review, validate, logging, spend guard, tests | scaffold |
| hotspots | `add-hotspot-extraction` | `hotspots.py` (Gemini segmentation + polygonisation + Claude verification), `tools/annotate`, tests | scaffold |
| web | `build-web-app` | the site per §11 running against placeholders, unit + e2e tests | scaffold |
| produce | `produce-v1-scenes` | bake-off, anchor, all scenes generated, selected, graded, hotspotted, exported; review sheets | content, pipeline, hotspots, web |
| qa-and-deploy | `deploy-v1` | full-loop e2e on real assets, Lighthouse check, Pages deployment, README usage docs | produce |
| morph | `add-morph-transitions` | Veo clips, video transition path, review | produce |

The four middle packages run **in parallel** as four Opus agents. They only share the contracts fixed by scaffold (manifest models, schema, CLI command names, directory layout), so merges are clean. If one of them needs a contract change it must say so in its report instead of changing the shared file silently.

### 13.3 Orchestrator checklist per package
1. Confirm prerequisites merged; spawn the implementer with its Appendix A brief.
2. On completion: read the report, run the package's acceptance commands locally, look at any produced images.
3. Watch CI to green; fix or send back if red.
4. Spawn the fresh-context reviewer; triage; fix real findings; CI green again.
5. Merge, open the PR once in Thiago's browser, note any contract deviations in this document.
6. For "produce": additionally send Thiago the bake-off sheet, the contact sheets and the full-loop walkthrough and wait for their veto before "qa-and-deploy".

### 13.4 Ground rules for every implementer (include in each brief)
- Work only inside your worktree on your branch; commit in small steps; push and open a PR with a plain-language description. Never force-push, never touch `main`.
- `uv` for all Python; `pnpm` for the web. Run `ruff format` / `prettier` before every commit. Python 3.12 (`uv python install 3.12` if needed; the system Python is 3.9).
- No vague tags in code, comments, commit messages or PR text. Explain reasoning in plain words.
- Never commit secrets. Read keys from `.env` via `python-dotenv` or the shell.
- Never call a paid API from a test. Fakes only. Paid calls happen only in the "produce"/"morph" packages, with the spend guard on.
- Do not change the shared contracts (manifest models, schema, CLI command names, layout) without flagging it in the final report.
- Final report format: what was built (bullets), how to run it (commands), what was verified (commands + results), open issues, any contract deviation.
- Shell is zsh on macOS.

---

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Generator ignores the bridge-object size/placement | Candidates + rubric + placement hint iteration; `dsp hotspot` measures the real area and `validate` fails fast. |
| Style drifts across scenes | Anchor reference, fixed style block, deterministic grade; a side-by-side strip in `assets/review/index.html` to eyeball consistency. |
| The segmenter picks the wrong instance or over-segments | Claude verification with box-prompt retry; manual editor as the final fallback. |
| Hotspot too easy to find by cursor sweeping | Pointer cursor is a constant; can be turned off. Idle hint back-off keeps it "hidden" but fair. |
| Mobile crop hides the bridge object | Focal point + panning; validation enforces the safe area. |
| Veo morph clips look wrong or shift colours | Clips are optional per edge; crossfade in/out; CSS fallback always present. |
| Model/API names drift after this plan was written | Every provider call sits behind the adapter; the implementer verifies model ids against the live docs listed in Appendix C before coding. |
| Repo bloat from images | Candidates git-ignored; masters as PNG only for 10 scenes; revisit LFS if the loop grows past ~30 scenes. |
| Provider refuses a prompt (safety filters) | Prompts avoid people; "prison" scene is empty and non-violent; log refusals and rephrase. |

---

## 15. Decisions confirmed by Thiago (2026-10-19)

1. **Repo and hosting**: private GitHub repo under `thiagorobert`, deployed to GitHub Pages. Confirmed.
2. **Keys**: `.env` populated from Thiago's shell profile with `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`. No fal account; hotspots use Gemini segmentation by default (see §2 and §6.2). `FAL_KEY` stays optional.
3. **Hint policy and chrome**: as described in §2 and §3. Confirmed.
4. **The ten scenes and the style bible**: approved as drafted.
5. **Morph clips**: the "morph" package is in scope for this round, after "produce".
6. **Loop size**: ten scenes.

---

## Appendix A — Subagent briefs (ready to paste)

Every brief starts with this preamble:

> You are implementing one work package of the Dreamscapes project. Read `docs/plans/2026-10-19-dreamscapes-v1-plan.md` in full before doing anything; your package is defined there and the ground rules in §13.4 apply to you. Work on the branch named below inside your worktree, commit in small steps, push, and open a PR with `gh pr create` whose body explains the change in plain language. Do not run any paid API from tests. When done, reply with the report format from §13.4.

### A.1 scaffold (`scaffold-repo`)
Goal: turn the empty directory into a working monorepo with every shared contract in place.
Deliverables:
- `git init` with `main`; initial commit; create the private GitHub repo with `gh repo create dreamscapes --private` (account `thiagorobert`, SSH) and push. A `.env` holding real keys and a minimal `.gitignore` already exist in the directory: never print, read or commit `.env`, and extend `.gitignore` rather than replacing it (node_modules, .venv, .env, assets/candidates, assets/review, web/dist, playwright-report), `.env.example` listing the variables in §9.3, and a README that explains the concept, layout and the commands.
- `pipeline/`: `uv init` with Python 3.12, deps `typer`, `pydantic`, `pyyaml`, `pillow`, `numpy`, `opencv-python-headless`, `python-dotenv`, `httpx`; dev deps `pytest`, `ruff`. `dreamscapes_pipeline/manifest.py` with the pydantic models for §8.2 and §8.3 (ids as slugs, polygons as lists of `[x, y]` floats in 0–1, `next` as a slug). `cli.py` registering every command from §10 as stubs that print "not implemented" except `validate`, `schema` and `placeholders`, which must work. `validate` implements §6.4 fully. `placeholders` writes synthetic masters (soft gradients in the style palette), masks, exports and `web/public/scenes.json` for the ten slugs in §4.3 with one hotspot each (a rectangle, varied positions) so the web package has real data to run against. Export `schema/scenes.schema.json`.
- `web/`: Vite + TypeScript strict scaffold with ESLint, Prettier, Vitest, Playwright installed and a trivial passing test of each kind; `src/types.ts` mirroring the schema; `index.html` that renders "Dreamscapes" only. Scripts: `dev`, `build`, `lint`, `typecheck`, `test`, `e2e`, `format`.
- `.github/workflows/ci.yml` and `deploy.yml` per §9.2 (deploy may be a stub that builds; enabling Pages is done in the qa-and-deploy package).
- `tools/annotate/` directory with a placeholder README (the hotspots package fills it).
Acceptance: `cd pipeline && uv run pytest -q && uv run ruff check . && uv run dsp validate && uv run dsp placeholders && uv run dsp schema`; `cd web && pnpm lint && pnpm typecheck && pnpm test && pnpm build`; CI green on the PR.

### A.2 content (`author-scene-loop`)
Goal: write the authored content. No code.
Deliverables: `content/style.md` from Appendix B refined into a tight, generator-friendly block (≤ 120 words) plus a "do not" list; `content/grade.toml` with the §5.5 parameters; `content/scenes.yaml` with all ten scenes from §4.3: `id`, `title`, `alt`, a 60–120 word `scene` description that names distractors, `bridge.object` (noun phrase usable as a segmentation prompt), `bridge.placement` (position in the frame inside the safe area, relative size), `next`. Leave `polygon` empty. Check with `uv run dsp validate` (it must pass except for the missing polygons, which validate treats as warnings until the produce package fills them).
Acceptance: validate passes with only "missing polygon" warnings; each scene description mentions the previous scene's bridge object; prompts contain no text/logo/people requests.

### A.3 pipeline (`add-generation-pipeline`)
Goal: implement §5 and §10 except `hotspot` and `morph`.
Deliverables: `providers/base.py` (`ImageProvider` protocol, `GenerationResult`), `providers/gemini.py` (verify the current model id, request shape, aspect ratio, `image_size`, reference-image parts and any seed parameter against the live docs listed in Appendix C before coding), `providers/openai_images.py` (edits endpoint with reference images, custom `size`), `generate.py` (prompt assembly from `content/style.md` + scene, anchor reference logic, candidates, idempotence, logging to `logs/generations.jsonl`, spend guard, `--dry-run`), `select.py` (Claude vision scoring per §5.4 using the `anthropic` Python SDK; load the `claude-api` skill for the exact current API shapes; structured output validated with pydantic; contact sheet HTML), `grade.py` (§5.5, parameters from `content/grade.toml`, golden test), `export.py` (AVIF/WebP variants via Pillow with `pillow-avif-plugin` or `pillow-heif`, LQIP, byte-hash suffixes, `scenes.json`), `review.py`, `bakeoff` command. Fakes for every provider in tests.
Acceptance: `uv run pytest -q` green with no network; `uv run dsp generate --scene cliff-cottage --dry-run` prints the exact request; `uv run dsp grade --preview` works on placeholder masters; `uv run dsp export` regenerates `scenes.json` identical to `placeholders` output for placeholder inputs.

### A.4 hotspots (`add-hotspot-extraction`)
Goal: implement §6.
Deliverables: `hotspots.py` with `segment(image, concept) -> Polygon` behind a `SegmentationProvider` protocol and a Gemini implementation using the `google-genai` SDK (segmentation through a JSON-schema response with `box_2d` and a polygon `mask` in 0–1000 coordinates; verify the current recommended model id, prompt wording and the minimal-thinking advice on https://ai.google.dev/gemini-api/docs/image-understanding before coding); an optional fal SAM 3 implementation (`fal-ai/sam-3/image`, raster masks) selectable with `DSP_SEGMENTATION_PROVIDER=sam3` is welcome but not required; `clean_polygon(polygon) -> Polygon` and `mask_to_polygon(mask) -> Polygon` (largest component, simplification ≤ 40 vertices, dilation, normalisation), `verify_with_claude(image, polygon, concept) -> Verdict` (overlay render + structured yes/no via the `anthropic` SDK; load the `claude-api` skill), the `dsp hotspot` command wiring (write back to `content/scenes.yaml`, overlay PNG), and `tools/annotate/index.html` per §6.3 (single file, no build step, loads an image from disk, exports the YAML snippet, shows the §4.4 checks live).
Acceptance: unit tests for polygonisation on synthetic masks (circle, L-shape, two blobs) with a fake segmentation provider; `uv run dsp hotspot --scene prison-yard --dry-run` on a placeholder shows the pipeline without network; the annotator works when opened in a browser (describe the manual check you did).

### A.5 web (`build-web-app`)
Goal: implement §3, §7.1 and §11 against the placeholder assets.
Deliverables: all modules in §11.2, `styles.css` (full-bleed stage, wordmark, counter, overlay, ripple, shimmer clipped to the polygon via an SVG `clipPath`, reduced-motion rules), the transition in §7.1, hints in §3.2, routing, preloading, a11y, the video transition path from §7.2 implemented behind `transition.type === "video"` with fallback (no clips exist yet; test with a tiny generated MP4 fixture), unit tests and the Playwright suite from §11.5 at both viewports, bundle size check.
Acceptance: `pnpm lint && pnpm typecheck && pnpm test && pnpm build && pnpm e2e` green locally and in CI; walking the loop by hand in a browser on placeholders works, including panning on a phone-sized viewport.

### A.6 produce (`produce-v1-scenes`)
Goal: make the ten real scenes. Requires `.env` with keys; stop and report if any key is missing.
Steps: `dsp bakeoff` → report to the orchestrator and wait for the provider decision (default Gemini) → `dsp generate --scene cliff-cottage --candidates 6` → `dsp select --scene cliff-cottage` (this becomes the style anchor) → `dsp generate --all` (anchor reference on) → `dsp select --all` → `dsp grade` → `dsp hotspot --all` → fix any `needs_manual_hotspot` scene by regenerating with a stronger placement hint (max two tries) and otherwise leave it for the orchestrator with the annotator → `dsp validate` → `dsp export` → `dsp review`. Commit masters, masks, exports, logs and manifest. Keep total spend under $15 and list the actual spend from the log in the report.
Acceptance: validate passes with zero warnings; `pnpm e2e` passes on the real assets; the report links every review sheet and lists per-scene scores, hotspot areas and anything the orchestrator should eyeball.

### A.7 qa-and-deploy (`deploy-v1`)
Goal: ship v1.
Deliverables: enable GitHub Pages for the repo (Actions source), finish `deploy.yml`, verify the deployed URL loads and the loop closes, Lighthouse performance ≥ 90 (report the numbers), README section "How to regenerate a scene" and "How to add a scene", fix anything found while walking the loop at desktop and phone sizes.
Acceptance: deployed URL in the report; CI green; README accurate.

### A.8 morph (`add-morph-transitions`)
Goal: §7.2. Verify the current Veo model id, `last_frame` parameter name, durations and pricing on the live docs first. Implement `morph.py` and the `dsp morph` command, ffmpeg post-processing (fail clearly if ffmpeg is absent, suggesting `brew install ffmpeg`), manifest updates, and a review sheet with first/middle/last frames per clip. Generate all ten edges, keep spend under $15, commit the clips.
Acceptance: each clip's first frame visually matches master A and last frame matches master B (report a diff metric per clip), the site plays them with the fallback intact, e2e still green.

---

## Appendix B — Style bible draft

> A still, dreamlike scene rendered in soft painterly realism: not a cartoon, not a photograph. Diffused light with a faint haze, long gentle shadows, a thin mist in the distance. Muted palette of dusty teal, slate blue and lavender, with warm amber glow on every light source. Gentle bloom around lights, fine film grain, slightly lifted blacks, subtle vignette. Wide establishing shot from eye level with a 35 mm lens and deep depth of field. The place is empty and quiet, as if everyone has just left. No people, no faces, no legible text, no signs, no logos, no watermarks. Calm, melancholic, inviting.

Do-not list for implementers: never add "masterpiece / trending / 8k" boilerplate; never request text; never request a person; keep the block identical across scenes (it is hashed into the generation log so drift is detectable).

---

## Appendix C — Sources consulted (2026-10-19)

- Google Gemini image generation docs (models `gemini-nano-banana-2.1`, `gemini-3.1-flash-image`, `gemini-3-pro-image`; `aspect_ratio`, `image_size`, up to 14 reference images): https://ai.google.dev/gemini-api/docs/image-generation
- Gemini API pricing (image models per 1K/2K/4K; Veo 3.1 per second): https://ai.google.dev/gemini-api/docs/pricing
- Veo 3.1 (`image` + `last_frame`, 4/6/8 s, 720p/1080p/4K, audio always on): https://ai.google.dev/gemini-api/docs/veo
- OpenAI image generation guide (`gpt-image-2.5-flare`, `gpt-image-2.5-sunburst`, custom sizes, `image[]` references on the edits endpoint, `quality` up to `max`): https://developers.openai.com/api/docs/guides/image-generation
- OpenAI GPT Image 2.5 pricing commentary (token-billed; third-party estimates): https://www.eesel.ai/blog/chatgpt-images-2-5-pricing and https://docs.kanaries.net/articles/gpt-image-2-5
- Black Forest Labs docs (FLUX.2 / FLUX 3 image, up to ten references): https://docs.bfl.ai/ and https://docs.bfl.ai/flux_3/flux3_image_overview
- Gemini image understanding (object detection and segmentation: `box_2d` as `[ymin, xmin, ymax, xmax]` 0–1000, polygon `mask`): https://ai.google.dev/gemini-api/docs/image-understanding
- SAM 3 on fal (optional provider; `fal-ai/sam-3/image`: `prompt`, `image_url`, masks, boxes, scores): https://fal.ai/models/fal-ai/sam-3/image/api
- SAM 3 concept prompts overview: https://blog.roboflow.com/what-is-sam3/ and https://docs.ultralytics.com/models/sam-3
- First/last-frame video model landscape (Kling O1, Luma Ray3.2, Veo 3.1): https://www.segmind.com/models/all/first-and-last-frame-video-models and https://aifilmcontests.com/guide/first-frame-for-image-to-video
- Claude models and API shapes: the bundled `claude-api` skill (model `claude-opus-5-5`, adaptive thinking, structured outputs via `output_config.format`).

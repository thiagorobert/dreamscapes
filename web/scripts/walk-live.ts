// Walks a deployed copy of the site the way a visitor would and reports what happened.
//
// Usage: pnpm walk <url> [--morph] [--json <file>] [--screenshots <dir>]
//   e.g. pnpm walk https://blog.thiago.pub/dreamscapes/
//
// Unlike the e2e suite, which runs against a local build, this drives the deployed site at
// 1440x900 and in the iPhone 13 profile. By default it walks with the CSS dissolve (?morph=0) and
// checks that no morph clip is ever requested; --morph walks with the real clips (?morph=1). The
// mode is set in the address explicitly, so the walk does not depend on the site's default, and
// the report says which one it used and what the page's morph switch showed. For each profile
// it opens the site in a fresh browser profile, dismisses the first-visit
// overlay, clicks (or taps) the middle of every scene's hidden object, checks the address and the
// depth counter after every step, notes which transition each step played, then checks the back
// button and a deep link. Console errors, page errors, failed requests and HTTP errors are
// recorded throughout. Exits 1 if any check failed. Needs Playwright's Chromium
// (`pnpm exec playwright install chromium`). Node strips the types, so no build step is needed.

import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { chromium, devices, type BrowserContextOptions, type Page } from '@playwright/test';

type Point = [number, number];

interface SceneInfo {
  id: string;
  next: string;
  bridge: { object: string; centroid: Point };
  transition: { type: 'zoom' } | { type: 'video'; video: string; duration: number };
}

interface ManifestInfo {
  start: string;
  scenes: SceneInfo[];
}

interface StepReport {
  step: number;
  from: string;
  to: string;
  object: string;
  /** Click point in viewport pixels, from the centroid of the polygon on the page. */
  point: { x: number; y: number };
  /** How far (in normalised image units) the page's polygon centroid is from scenes.json's. */
  centroidDelta: number;
  /** Whether the browser's own hit test put the click point on the hotspot. */
  hitTest: boolean;
  hash: string;
  counter: string;
  /** The value of #app[data-transition] once the step settled. */
  transition: string;
  expectedTransition: string;
  /** Furthest playback position seen in a clip on stage during the step, in seconds. */
  clipPlayedSeconds: number;
  elapsedMs: number;
  ok: boolean;
  problems: string[];
}

type Mode = 'dissolve' | 'morph';

interface ProfileReport {
  profile: string;
  viewport: string;
  mode: Mode;
  /** The text of the page's morph switch, or "missing" when there is none. */
  morphSwitch: string;
  loadMs: number;
  firstImage: string;
  introShown: boolean;
  introDismissed: boolean;
  steps: StepReport[];
  loopClosed: boolean;
  back: { ok: boolean; detail: string };
  forward: { ok: boolean; detail: string };
  deepLink: { ok: boolean; detail: string };
  unknownHash: { ok: boolean; detail: string };
  consoleErrors: string[];
  pageErrors: string[];
  failedRequests: string[];
  /** Requests the page itself cancelled, such as a preloaded clip it let go of. Not failures. */
  abortedRequests: string[];
  httpErrors: string[];
  /** Requests for morph clips; there must be none when walking with the dissolve. */
  clipRequests: string[];
  requests: number;
  ok: boolean;
}

const PROFILES: { name: string; options: BrowserContextOptions }[] = [
  {
    name: 'desktop',
    options: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } },
  },
  // Chromium with the iPhone 13 viewport, pixel ratio, touch and user agent, as in the e2e suite.
  { name: 'iphone-13', options: { ...devices['iPhone 13'] } },
];

/** Generous: a morph clip runs for eight seconds and has to download first. */
const STEP_TIMEOUT_MS = 45_000;

interface Args {
  url: string;
  mode: Mode;
  json?: string;
  screenshots?: string;
}

function parseArgs(argv: string[]): Args {
  let url: string | undefined;
  let json: string | undefined;
  let screenshots: string | undefined;
  let mode: Mode = 'dissolve';
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--morph') {
      mode = 'morph';
    } else if (arg === '--json') {
      json = argv[(i += 1)];
    } else if (arg === '--screenshots') {
      screenshots = argv[(i += 1)];
    } else if (arg !== undefined && !arg.startsWith('--')) {
      url = arg;
    }
  }
  if (!url) {
    console.error('Usage: pnpm walk <url> [--morph] [--json <file>] [--screenshots <dir>]');
    process.exit(2);
  }
  // The site resolves its files relative to the page, so the URL must end in a slash.
  const base = new URL(url);
  base.hash = '';
  if (!base.pathname.endsWith('/')) {
    base.pathname += '/';
  }
  base.search = '';
  const result: Args = { url: base.href, mode };
  if (json !== undefined) {
    result.json = json;
  }
  if (screenshots !== undefined) {
    result.screenshots = screenshots;
  }
  return result;
}

/** Area-weighted centroid of a polygon, as the pipeline computes it. */
function polygonCentroid(points: Point[]): Point {
  let area = 0;
  let cx = 0;
  let cy = 0;
  points.forEach(([x0, y0], i) => {
    const [x1, y1] = points[(i + 1) % points.length] ?? [x0, y0];
    const cross = x0 * y1 - x1 * y0;
    area += cross;
    cx += (x0 + x1) * cross;
    cy += (y0 + y1) * cross;
  });
  area /= 2;
  if (Math.abs(area) < 1e-12) {
    const n = points.length;
    return [points.reduce((s, p) => s + p[0], 0) / n, points.reduce((s, p) => s + p[1], 0) / n];
  }
  return [cx / (6 * area), cy / (6 * area)];
}

/** The address to open: the site with its morph switch set to `mode` through the URL. */
function addressFor(url: string, mode: Mode, hash = ''): string {
  const address = new URL(url);
  address.searchParams.set('morph', mode === 'morph' ? '1' : '0');
  address.hash = hash;
  return address.href;
}

async function waitIdle(page: Page): Promise<void> {
  await page.waitForFunction(
    () => document.querySelector('#app')?.getAttribute('data-busy') === 'false',
    null,
    {
      timeout: STEP_TIMEOUT_MS,
    },
  );
}

async function appState(page: Page): Promise<{
  scene: string;
  hash: string;
  counter: string;
  transition: string;
  busy: string;
}> {
  return page.evaluate(() => {
    const app = document.querySelector('#app');
    return {
      scene: app?.getAttribute('data-scene') ?? '',
      hash: location.hash,
      counter: document.querySelector('.depth')?.textContent ?? '',
      transition: app?.getAttribute('data-transition') ?? '',
      busy: app?.getAttribute('data-busy') ?? '',
    };
  });
}

/** Screen point of the current hotspot's polygon centroid, plus the polygon's own centroid. */
async function hotspotTarget(
  page: Page,
): Promise<{ point: { x: number; y: number }; centroid: Point }> {
  const found = await page.evaluate(() => {
    const layer = document.querySelector('.scene-layer.is-current');
    const shape = layer?.querySelector('.hotspot-shape');
    if (!layer || !shape) {
      return null;
    }
    const box = layer.getBoundingClientRect();
    return {
      points: shape.getAttribute('points') ?? '',
      box: { x: box.x, y: box.y, width: box.width, height: box.height },
    };
  });
  if (!found) {
    throw new Error('No hotspot on the current scene');
  }
  const points = found.points
    .trim()
    .split(/\s+/)
    .map((pair) => pair.split(',').map(Number) as Point);
  const centroid = polygonCentroid(points);
  return {
    centroid,
    point: {
      x: found.box.x + centroid[0] * found.box.width,
      y: found.box.y + centroid[1] * found.box.height,
    },
  };
}

/** Starts sampling the clip overlay on stage; read the result with `stopProbe`. */
async function startProbe(page: Page): Promise<void> {
  await page.evaluate(() => {
    const probe = { maxTime: 0, timer: 0 };
    probe.timer = window.setInterval(() => {
      for (const video of document.querySelectorAll<HTMLVideoElement>(
        '.morph video, video.morph',
      )) {
        probe.maxTime = Math.max(probe.maxTime, video.currentTime);
      }
    }, 100);
    (window as unknown as { walkProbe: typeof probe }).walkProbe = probe;
  });
}

async function stopProbe(page: Page): Promise<number> {
  return page.evaluate(() => {
    const probe = (window as unknown as { walkProbe?: { maxTime: number; timer: number } })
      .walkProbe;
    if (!probe) {
      return 0;
    }
    window.clearInterval(probe.timer);
    return probe.maxTime;
  });
}

async function press(page: Page, touch: boolean, x: number, y: number): Promise<void> {
  if (touch) {
    await page.touchscreen.tap(x, y);
  } else {
    await page.mouse.click(x, y);
  }
}

async function walkProfile(
  url: string,
  mode: Mode,
  manifest: ManifestInfo,
  profile: { name: string; options: BrowserContextOptions },
  screenshots: string | undefined,
): Promise<ProfileReport> {
  const byId = new Map(manifest.scenes.map((scene) => [scene.id, scene]));
  const browser = await chromium.launch();
  const context = await browser.newContext(profile.options);
  const touch = profile.options.hasTouch === true;
  const viewport = profile.options.viewport;
  const report: ProfileReport = {
    profile: profile.name,
    viewport: viewport ? `${String(viewport.width)}x${String(viewport.height)}` : 'default',
    mode,
    morphSwitch: 'missing',
    loadMs: 0,
    firstImage: '',
    introShown: false,
    introDismissed: false,
    steps: [],
    loopClosed: false,
    back: { ok: false, detail: 'not run' },
    forward: { ok: false, detail: 'not run' },
    deepLink: { ok: false, detail: 'not run' },
    unknownHash: { ok: false, detail: 'not run' },
    consoleErrors: [],
    pageErrors: [],
    failedRequests: [],
    abortedRequests: [],
    httpErrors: [],
    clipRequests: [],
    requests: 0,
    ok: false,
  };

  const watch = (page: Page): void => {
    page.on('console', (message) => {
      if (message.type() === 'error') {
        report.consoleErrors.push(message.text());
      }
    });
    page.on('pageerror', (error) => {
      report.pageErrors.push(error.message);
    });
    page.on('request', (request) => {
      report.requests += 1;
      if (new URL(request.url()).pathname.includes('/morphs/')) {
        report.clipRequests.push(request.url());
      }
    });
    page.on('requestfailed', (request) => {
      const reason = request.failure()?.errorText ?? 'unknown';
      const line = `${reason} ${request.url()}`;
      if (reason.includes('ERR_ABORTED')) {
        report.abortedRequests.push(line);
      } else {
        report.failedRequests.push(line);
      }
    });
    page.on('response', (response) => {
      if (response.status() >= 400) {
        report.httpErrors.push(`${String(response.status())} ${response.url()}`);
      }
    });
  };

  const page = await context.newPage();
  watch(page);
  const started = Date.now();
  await page.goto(addressFor(url, mode), { waitUntil: 'load' });
  await page.waitForSelector('.scene-layer.is-current img.is-loaded', { timeout: STEP_TIMEOUT_MS });
  report.loadMs = Date.now() - started;
  report.firstImage = await page.evaluate(
    () => document.querySelector<HTMLImageElement>('.scene-layer.is-current img')?.currentSrc ?? '',
  );
  if (screenshots) {
    await page.screenshot({ path: join(screenshots, `${profile.name}-00-intro.png`) });
  }

  // The first-visit overlay: any click or tap dismisses it without counting as a miss.
  const intro = page.locator('.intro');
  report.introShown = await intro.isVisible();
  if (report.introShown) {
    const box = await intro.boundingBox();
    if (box) {
      await press(page, touch, box.x + box.width / 2, box.y + box.height / 2);
    }
    await intro.waitFor({ state: 'detached', timeout: 5000 }).catch(() => undefined);
    report.introDismissed = (await intro.count()) === 0;
  }
  await waitIdle(page);
  report.morphSwitch =
    (
      await page
        .locator('.morph-toggle')
        // A site from before the switch existed has none; do not wait long for it.
        .textContent({ timeout: 3000 })
        .catch(() => null)
    )?.trim() ?? 'missing';

  let state = await appState(page);
  const expectedStart = manifest.start;
  const problemsAtStart: string[] = [];
  if (state.scene !== expectedStart) {
    problemsAtStart.push(`started on ${state.scene}, expected ${expectedStart}`);
  }
  if (state.hash !== `#/${expectedStart}`) {
    problemsAtStart.push(`start hash ${state.hash}`);
  }
  if (state.counter !== 'dream 1') {
    problemsAtStart.push(`start counter "${state.counter}"`);
  }
  const expectedSwitch = mode === 'morph' ? 'morph on' : 'morph off';
  if (report.morphSwitch !== expectedSwitch) {
    problemsAtStart.push(
      `morph switch reads "${report.morphSwitch}", expected "${expectedSwitch}"`,
    );
  }

  for (let step = 1; step <= manifest.scenes.length; step += 1) {
    const fromId = state.scene;
    const from = byId.get(fromId);
    const problems: string[] = step === 1 ? [...problemsAtStart] : [];
    if (!from) {
      throw new Error(`Unknown scene on stage: ${fromId}`);
    }
    const target = await hotspotTarget(page);
    const delta = Math.hypot(
      target.centroid[0] - from.bridge.centroid[0],
      target.centroid[1] - from.bridge.centroid[1],
    );
    const hitTest = await page.evaluate(
      ({ x, y }) => document.elementFromPoint(x, y)?.closest('.hotspot') != null,
      target.point,
    );
    const onScreen =
      viewport !== null &&
      viewport !== undefined &&
      target.point.x >= 0 &&
      target.point.y >= 0 &&
      target.point.x <= viewport.width &&
      target.point.y <= viewport.height;
    if (!onScreen) {
      problems.push('click point is off-screen');
    }
    if (!hitTest) {
      problems.push('click point is not on the hotspot');
    }

    await startProbe(page);
    const clicked = Date.now();
    await press(page, touch, target.point.x, target.point.y);
    await page.waitForFunction((next) => location.hash === `#/${next}`, from.next, {
      timeout: STEP_TIMEOUT_MS,
    });
    await waitIdle(page);
    const elapsedMs = Date.now() - clicked;
    const clipPlayedSeconds = await stopProbe(page);
    state = await appState(page);

    const expectedCounter = `dream ${String(step + 1)}`;
    // With the dissolve every edge zooms, whatever scenes.json says about clips.
    const expectedTransition =
      mode === 'morph' && from.transition.type === 'video' ? 'video' : 'zoom';
    if (state.scene !== from.next) {
      problems.push(`scene ${state.scene}, expected ${from.next}`);
    }
    if (state.hash !== `#/${from.next}`) {
      problems.push(`hash ${state.hash}`);
    }
    if (state.counter !== expectedCounter) {
      problems.push(`counter "${state.counter}", expected "${expectedCounter}"`);
    }
    if (state.transition !== expectedTransition) {
      problems.push(`played ${state.transition}, expected ${expectedTransition}`);
    }
    if (expectedTransition === 'video' && clipPlayedSeconds < 1) {
      problems.push(`clip reached only ${clipPlayedSeconds.toFixed(2)} s`);
    }
    report.steps.push({
      step,
      from: fromId,
      to: state.scene,
      object: from.bridge.object,
      point: { x: Math.round(target.point.x), y: Math.round(target.point.y) },
      centroidDelta: delta,
      hitTest,
      hash: state.hash,
      counter: state.counter,
      transition: state.transition,
      expectedTransition,
      clipPlayedSeconds,
      elapsedMs,
      ok: problems.length === 0,
      problems,
    });
    if (screenshots) {
      await page.screenshot({
        path: join(
          screenshots,
          `${profile.name}-${String(step).padStart(2, '0')}-${state.scene}.png`,
        ),
      });
    }
  }
  report.loopClosed =
    state.scene === manifest.start &&
    state.hash === `#/${manifest.start}` &&
    state.counter === `dream ${String(manifest.scenes.length + 1)}`;

  // Back returns to the previous dream (and its depth); forward comes back.
  const last = report.steps.at(-1);
  if (last) {
    await page.goBack();
    await page.waitForFunction((id) => location.hash === `#/${id}`, last.from);
    await waitIdle(page);
    await page.waitForFunction(
      (id) => document.querySelector('#app')?.getAttribute('data-scene') === id,
      last.from,
    );
    state = await appState(page);
    const wantCounter = `dream ${String(manifest.scenes.length)}`;
    report.back = {
      ok: state.scene === last.from && state.counter === wantCounter,
      detail: `${state.hash}, ${state.counter}, transition ${state.transition}`,
    };

    await page.goForward();
    await page.waitForFunction((id) => location.hash === `#/${id}`, manifest.start);
    await waitIdle(page);
    await page.waitForFunction(
      (id) => document.querySelector('#app')?.getAttribute('data-scene') === id,
      manifest.start,
    );
    state = await appState(page);
    report.forward = {
      ok:
        state.scene === manifest.start &&
        state.counter === `dream ${String(manifest.scenes.length + 1)}`,
      detail: `${state.hash}, ${state.counter}, transition ${state.transition}`,
    };
  }

  // A deep link in a new tab opens that scene directly, at depth 1, with its image loaded.
  const deepId =
    manifest.scenes.find((scene) => scene.id === 'endless-library')?.id ??
    manifest.scenes[Math.floor(manifest.scenes.length / 2)]?.id ??
    manifest.start;
  const deep = await context.newPage();
  watch(deep);
  await deep.goto(addressFor(url, mode, `#/${deepId}`), { waitUntil: 'load' });
  await deep.waitForSelector('.scene-layer.is-current img.is-loaded', { timeout: STEP_TIMEOUT_MS });
  await waitIdle(deep);
  state = await appState(deep);
  const deepTitle = await deep.title();
  report.deepLink = {
    ok: state.scene === deepId && state.hash === `#/${deepId}` && state.counter === 'dream 1',
    detail: `${state.hash}, ${state.counter}, title "${deepTitle}"`,
  };
  await deep.close();

  // An unknown scene falls back to the start scene and rewrites the address.
  const unknown = await context.newPage();
  watch(unknown);
  await unknown.goto(addressFor(url, mode, '#/no-such-dream'), { waitUntil: 'load' });
  await unknown.waitForSelector('.scene-layer.is-current', { timeout: STEP_TIMEOUT_MS });
  await waitIdle(unknown);
  state = await appState(unknown);
  report.unknownHash = {
    ok: state.scene === manifest.start && state.hash === `#/${manifest.start}`,
    detail: `${state.hash}, ${state.counter}`,
  };
  await unknown.close();

  await context.close();
  await browser.close();

  report.ok =
    report.introShown &&
    report.introDismissed &&
    report.steps.every((step) => step.ok) &&
    report.loopClosed &&
    report.back.ok &&
    report.forward.ok &&
    report.deepLink.ok &&
    report.unknownHash.ok &&
    report.consoleErrors.length === 0 &&
    report.pageErrors.length === 0 &&
    report.failedRequests.length === 0 &&
    report.httpErrors.length === 0 &&
    (mode === 'morph' || report.clipRequests.length === 0);
  return report;
}

function printReport(report: ProfileReport): void {
  console.log(`\n== ${report.profile} (${report.viewport}), ${report.mode} ==`);
  console.log(`morph switch on the page: ${report.morphSwitch}`);
  console.log(`first image on screen after ${String(report.loadMs)} ms: ${report.firstImage}`);
  console.log(
    `intro shown: ${String(report.introShown)}, dismissed: ${String(report.introDismissed)}`,
  );
  for (const step of report.steps) {
    const clip =
      step.clipPlayedSeconds > 0 ? `, clip played to ${step.clipPlayedSeconds.toFixed(1)} s` : '';
    console.log(
      `${step.ok ? 'ok  ' : 'FAIL'} ${String(step.step).padStart(2)}. ${step.from} -> ${step.to} ` +
        `(${step.object}) at (${String(step.point.x)}, ${String(step.point.y)}), ` +
        `hit ${String(step.hitTest)}, centroid delta ${step.centroidDelta.toFixed(4)}, ` +
        `${step.counter}, ${step.transition}${clip}, ${String(step.elapsedMs)} ms` +
        (step.problems.length > 0 ? ` -- ${step.problems.join('; ')}` : ''),
    );
  }
  console.log(`loop closed: ${String(report.loopClosed)}`);
  // Each step was already checked against its own edge in the manifest; these are totals.
  for (const kind of ['video', 'zoom']) {
    const played = report.steps.filter((step) => step.transition === kind).length;
    const expected = report.steps.filter((step) => step.expectedTransition === kind).length;
    console.log(`${kind} transitions: ${String(played)} played, ${String(expected)} expected`);
  }
  console.log(`back: ${report.back.ok ? 'ok' : 'FAIL'} (${report.back.detail})`);
  console.log(`forward: ${report.forward.ok ? 'ok' : 'FAIL'} (${report.forward.detail})`);
  console.log(`deep link: ${report.deepLink.ok ? 'ok' : 'FAIL'} (${report.deepLink.detail})`);
  console.log(
    `unknown hash: ${report.unknownHash.ok ? 'ok' : 'FAIL'} (${report.unknownHash.detail})`,
  );
  console.log(`requests: ${String(report.requests)}`);
  console.log(
    `morph clip requests: ${String(report.clipRequests.length)}` +
      (report.mode === 'dissolve' ? ' (must be 0 with the dissolve)' : ''),
  );
  const lists: [string, string[]][] = [
    ['console errors', report.consoleErrors],
    ['page errors', report.pageErrors],
    ['failed requests', report.failedRequests],
    ['HTTP errors', report.httpErrors],
    ['requests cancelled by the page', report.abortedRequests],
  ];
  for (const [label, items] of lists) {
    console.log(`${label}: ${String(items.length)}`);
    for (const item of items) {
      console.log(`  ${item}`);
    }
  }
  console.log(report.ok ? 'RESULT: ok' : 'RESULT: FAIL');
}

const args = parseArgs(process.argv.slice(2));
if (args.screenshots) {
  mkdirSync(args.screenshots, { recursive: true });
}
const response = await fetch(new URL('scenes.json', args.url));
if (!response.ok) {
  console.error(`Could not fetch scenes.json: HTTP ${String(response.status)}`);
  process.exit(1);
}
const manifest = (await response.json()) as ManifestInfo;
const modeNote =
  args.mode === 'morph' ? 'with the morph clips (?morph=1)' : 'with the CSS dissolve (?morph=0)';
console.log(
  `Walking ${args.url} ${modeNote} (${String(manifest.scenes.length)} scenes, ` +
    `start ${manifest.start})`,
);

const reports: ProfileReport[] = [];
for (const profile of PROFILES) {
  const report = await walkProfile(args.url, args.mode, manifest, profile, args.screenshots);
  printReport(report);
  reports.push(report);
}
if (args.json) {
  writeFileSync(args.json, JSON.stringify(reports, null, 2));
}
process.exit(reports.every((report) => report.ok) ? 0 : 1);

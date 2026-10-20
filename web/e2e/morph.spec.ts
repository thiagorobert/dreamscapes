// The morph-clip transition. Every test opens the site with ?morph=1, since visitors get the
// dissolve until they switch the clips on. Most tests rewrite scenes.json in flight to give the
// start scene's edge a fixture clip, so they control exactly how the clip loads; one plays the
// real clip the committed manifest names for that edge.
//
// fixtures/morph.mp4 is one second of H.264 at 320x180, a crossfade between the first two
// placeholder scenes, made with:
//   ffmpeg -loop 1 -t 1.2 -i public/dreams/cliff-cottage/cliff-cottage-640.<hash>.webp \
//     -loop 1 -t 1.2 -i public/dreams/prison-yard/prison-yard-640.<hash>.webp \
//     -filter_complex "[0:v]scale=320:180,setsar=1,format=yuv420p[a];\
//       [1:v]scale=320:180,setsar=1,format=yuv420p[b];\
//       [a][b]xfade=transition=fade:duration=0.8:offset=0.2,fps=24,format=yuv420p[v]" \
//     -map "[v]" -t 1.0 -c:v libx264 -profile:v baseline -level 3.0 -crf 32 \
//     -movflags +faststart -an e2e/fixtures/morph.mp4

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { expect, test, type Page } from '@playwright/test';

import type { Manifest } from '../src/types';
import {
  advance,
  hashPattern,
  hotspotCentre,
  open,
  press,
  recordClipRequests,
  skipIntro,
  start,
  waitUntilIdle,
  WITH_MORPHS,
} from './helpers';

const FIXTURE = fileURLToPath(new URL('fixtures/morph.mp4', import.meta.url));

// fixtures/stall.mp4 is three seconds of ffmpeg's test pattern, large enough (about 68 KB) that
// Chromium starts playing before all of it has arrived:
//   ffmpeg -f lavfi -i "testsrc2=s=320x180:d=3:r=24" -c:v libx264 -profile:v baseline \
//     -level 3.0 -pix_fmt yuv420p -crf 26 -movflags +faststart -an e2e/fixtures/stall.mp4
const STALL_FIXTURE = fileURLToPath(new URL('fixtures/stall.mp4', import.meta.url));
/** Bytes served before the network goes quiet: about 1.3 s of the 3 s clip. */
const STALL_AFTER_BYTES = 48 * 1024;

/** Serves scenes.json with a video transition on the start scene's edge. */
async function useClip(page: Page, video: string, duration = 1): Promise<void> {
  await page.route('**/scenes.json', async (route) => {
    const response = await route.fetch();
    const data = (await response.json()) as Manifest;
    for (const scene of data.scenes) {
      if (scene.id === start.id) {
        scene.transition = { type: 'video', video, duration };
      }
    }
    await route.fulfill({ response, json: data });
  });
}

/** readyState of the clip buffering off-screen, if any. */
function bufferedClipState(page: Page): Promise<number | undefined> {
  return page.evaluate(
    () => document.querySelector<HTMLVideoElement>('.offscreen video')?.readyState,
  );
}

test.beforeEach(async ({ page }) => {
  await skipIntro(page);
});

test('a missing clip falls back to the zoom', async ({ page, hasTouch }) => {
  await useClip(page, 'morphs/does-not-exist.mp4');
  await open(page, WITH_MORPHS);
  await advance(page, hasTouch);
  await expect(page.locator('#app')).toHaveAttribute('data-transition', 'zoom-fallback');
  await expect(page.locator('.depth')).toHaveText('dream 2');
  await expect(page.locator('.morph')).toHaveCount(0);
});

test('a clip that is not playable within a second falls back to the zoom', async ({
  page,
  hasTouch,
}) => {
  await useClip(page, 'morphs/slow.mp4');
  await page.route('**/morphs/slow.mp4', async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 5000));
    try {
      await route.fulfill({ path: FIXTURE, contentType: 'video/mp4' });
    } catch {
      // The test already finished and closed the page.
    }
  });
  await open(page, WITH_MORPHS);
  const clicked = Date.now();
  await advance(page, hasTouch);
  await expect(page.locator('#app')).toHaveAttribute('data-transition', 'zoom-fallback');
  // About a second of waiting for the clip, then the zoom: well before the clip arrives.
  expect(Date.now() - clicked).toBeLessThan(4500);
});

test('a playable clip plays over the stage, then the next scene shows', async ({
  page,
  hasTouch,
}) => {
  await useClip(page, 'morphs/e2e-fixture.mp4');
  await page.route('**/morphs/e2e-fixture.mp4', (route) =>
    route.fulfill({ path: FIXTURE, contentType: 'video/mp4' }),
  );
  await open(page, WITH_MORPHS);
  // The clip buffers off-screen once the next images are in.
  await expect.poll(() => bufferedClipState(page)).toBeGreaterThanOrEqual(3);

  await press(page, hasTouch, await hotspotCentre(page));
  await expect(page.locator('.morph video')).toHaveCount(1);
  await expect
    .poll(() =>
      page.evaluate(() => {
        const video = document.querySelector<HTMLVideoElement>('.morph video');
        return video !== null && !video.paused && video.currentTime > 0;
      }),
    )
    .toBe(true);

  await expect(page).toHaveURL(hashPattern(start.next));
  await waitUntilIdle(page);
  await expect(page.locator('#app')).toHaveAttribute('data-transition', 'video');
  await expect(page.locator('.morph')).toHaveCount(0);
  await expect(page.locator('.scene-layer')).toHaveCount(1);
  await expect(page.locator('#app')).toHaveAttribute('data-scene', start.next);
  await expect(page.locator('.depth')).toHaveText('dream 2');
});

test("the start scene's real clip plays from the committed manifest", async ({
  page,
  hasTouch,
}) => {
  const clip = start.transition.type === 'video' ? start.transition.video : null;
  test.skip(clip === null, 'the start scene has no morph clip in scenes.json');
  // An eight-second clip, plus buffering and the fades.
  test.setTimeout(60_000);
  await open(page, WITH_MORPHS);
  await expect.poll(() => bufferedClipState(page), { timeout: 20_000 }).toBeGreaterThanOrEqual(3);

  await press(page, hasTouch, await hotspotCentre(page));
  await expect(page.locator('.morph video')).toHaveCount(1);
  await expect
    .poll(() =>
      page.evaluate(() => {
        const video = document.querySelector<HTMLVideoElement>('.morph video');
        return video !== null && !video.paused && video.currentTime > 0 && video.videoWidth > 0;
      }),
    )
    .toBe(true);
  const playing = await page.evaluate(
    () => document.querySelector<HTMLVideoElement>('.morph video')?.currentSrc ?? '',
  );
  expect(playing).toContain(clip ?? 'no clip');

  await expect(page.locator('#app')).toHaveAttribute('data-busy', 'false', { timeout: 30_000 });
  await expect(page).toHaveURL(hashPattern(start.next));
  await expect(page.locator('#app')).toHaveAttribute('data-transition', 'video');
  await expect(page.locator('.morph')).toHaveCount(0);
  await expect(page.locator('#app')).toHaveAttribute('data-scene', start.next);
});

test('a clip that stalls mid-play is cut short and the next scene shows', async ({
  page,
  hasTouch,
}) => {
  const clip = readFileSync(STALL_FIXTURE);
  await useClip(page, 'morphs/stall.mp4', 3);
  await page.route('**/morphs/stall.mp4', async (route) => {
    const range = /bytes=(\d+)-/.exec(route.request().headers().range ?? '');
    if (Number(range?.[1] ?? 0) === 0) {
      await route.fulfill({
        status: 206,
        headers: {
          'Accept-Ranges': 'bytes',
          'Content-Range': `bytes 0-${String(STALL_AFTER_BYTES - 1)}/${String(clip.length)}`,
          'Content-Type': 'video/mp4',
        },
        body: clip.subarray(0, STALL_AFTER_BYTES),
      });
    }
    // Every later range request is left hanging, like a network that stopped delivering.
  });
  await open(page, WITH_MORPHS);
  await expect.poll(() => bufferedClipState(page)).toBeGreaterThanOrEqual(3);

  const clicked = Date.now();
  await press(page, hasTouch, await hotspotCentre(page));
  await expect(page.locator('.morph video')).toHaveCount(1);
  await expect(page).toHaveURL(hashPattern(start.next));
  const elapsed = Date.now() - clicked;
  await waitUntilIdle(page);
  await expect(page.locator('#app')).toHaveAttribute('data-transition', 'video');
  await expect(page.locator('.morph')).toHaveCount(0);
  await expect(page.locator('#app')).toHaveAttribute('data-scene', start.next);
  // About 1.3 s of playing, then the 0.4 s stall grace and the fades. Waiting for the clip's
  // full 3 s plus the 1.5 s end slack instead would take over 4.5 s.
  expect(elapsed).toBeLessThan(3500);
});

test.describe('with reduced motion', () => {
  test.use({ reducedMotion: 'reduce' });

  test('the clip is skipped for the crossfade, and never fetched', async ({ page, hasTouch }) => {
    const clips = recordClipRequests(page);
    await useClip(page, 'morphs/e2e-fixture.mp4');
    await page.route('**/morphs/e2e-fixture.mp4', (route) =>
      route.fulfill({ path: FIXTURE, contentType: 'video/mp4' }),
    );
    await open(page, WITH_MORPHS);
    await expect(page.getByRole('button', { name: 'morph on' })).toHaveAttribute(
      'aria-pressed',
      'true',
    );
    await advance(page, hasTouch);
    await expect(page.locator('#app')).toHaveAttribute('data-transition', 'fade');
    expect(clips).toEqual([]);
  });
});

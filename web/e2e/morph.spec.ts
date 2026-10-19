// The morph-clip transition. No real clips exist yet, so each test rewrites scenes.json in flight
// to give the start scene's edge a video transition.
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

import { fileURLToPath } from 'node:url';

import { expect, test, type Page } from '@playwright/test';

import type { Manifest } from '../src/types';
import {
  advance,
  hashPattern,
  hotspotCentre,
  open,
  press,
  skipIntro,
  start,
  waitUntilIdle,
} from './helpers';

const FIXTURE = fileURLToPath(new URL('fixtures/morph.mp4', import.meta.url));

/** Serves scenes.json with a video transition on the start scene's edge. */
async function useClip(page: Page, video: string): Promise<void> {
  await page.route('**/scenes.json', async (route) => {
    const response = await route.fetch();
    const data = (await response.json()) as Manifest;
    for (const scene of data.scenes) {
      if (scene.id === start.id) {
        scene.transition = { type: 'video', video, duration: 1 };
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
  await open(page);
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
  await open(page);
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
  await open(page);
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

test.describe('with reduced motion', () => {
  test.use({ reducedMotion: 'reduce' });

  test('the clip is skipped for the crossfade', async ({ page, hasTouch }) => {
    await useClip(page, 'morphs/e2e-fixture.mp4');
    await page.route('**/morphs/e2e-fixture.mp4', (route) =>
      route.fulfill({ path: FIXTURE, contentType: 'video/mp4' }),
    );
    await open(page);
    await advance(page, hasTouch);
    await expect(page.locator('#app')).toHaveAttribute('data-transition', 'fade');
  });
});

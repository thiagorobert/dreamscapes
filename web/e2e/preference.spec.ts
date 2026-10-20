// The morph switch: the CSS dissolve or the clips, chosen by the visitor or by the URL.
//
// Apart from the first test, which checks DEFAULT_TRANSITION itself, every test starts from an
// explicit choice, so the default can be flipped without touching them.

import { expect, test, type Locator, type Page } from '@playwright/test';

import { DEFAULT_TRANSITION, MORPH_TOGGLE_LABEL } from '../src/preference';
import type { Manifest } from '../src/types';
import {
  CONTINUE_LABEL,
  WITH_MORPHS,
  WITHOUT_MORPHS,
  advance,
  hashPattern,
  hotspotCentre,
  open,
  press,
  recordClipRequests,
  sceneById,
  skipIntro,
  start,
  waitUntilIdle,
} from './helpers';

function morphToggle(page: Page): Locator {
  return page.getByRole('button', { name: MORPH_TOGGLE_LABEL });
}

async function flip(toggle: Locator, hasTouch: boolean): Promise<void> {
  await (hasTouch ? toggle.tap() : toggle.click());
}

async function expectMorph(page: Page, on: boolean): Promise<void> {
  const toggle = morphToggle(page);
  await expect(toggle).toHaveAttribute('aria-pressed', String(on));
  await expect(toggle).toHaveText(on ? 'morph on' : 'morph off');
}

/** The clip path scenes.json gives the edge leaving `id`. */
function clipLeaving(id: string): string {
  const { transition } = sceneById(id);
  if (transition.type !== 'video') {
    throw new Error(`The edge leaving ${id} has no clip in scenes.json`);
  }
  return transition.video;
}

/** Serves scenes.json with every edge set to the zoom, so steps stay quick with clips on. */
async function withoutClipEdges(page: Page): Promise<void> {
  await page.route('**/scenes.json', async (route) => {
    const response = await route.fetch();
    const data = (await response.json()) as Manifest;
    for (const scene of data.scenes) {
      scene.transition = { type: 'zoom' };
    }
    await route.fulfill({ response, json: data });
  });
}

test.beforeEach(async ({ page }) => {
  await skipIntro(page);
});

test('a first visit gets DEFAULT_TRANSITION', async ({ page }) => {
  await open(page, './');
  await expectMorph(page, DEFAULT_TRANSITION === 'morph');
});

test('with the dissolve, video edges zoom and no clip is ever fetched', async ({
  page,
  hasTouch,
}) => {
  const clips = recordClipRequests(page);
  const second = sceneById(start.next);
  const third = sceneById(second.next);
  // The first two edges of the committed loop have clips, which the dissolve must leave alone.
  clipLeaving(start.id);
  clipLeaving(second.id);

  // A clip is only ever buffered after the next two scenes' images, so wait for those first.
  const thirdImage = page.waitForResponse((response) =>
    response.url().includes(`/dreams/${third.id}/`),
  );
  // While the dissolve is the default, come in through the bare address so the default itself
  // is what gets tested.
  await open(page, DEFAULT_TRANSITION === 'dissolve' ? './' : WITHOUT_MORPHS);
  await expectMorph(page, false);
  await thirdImage;

  await advance(page, hasTouch);
  await expect(page.locator('#app')).toHaveAttribute('data-transition', 'zoom');
  await advance(page, hasTouch);
  await expect(page.locator('#app')).toHaveAttribute('data-transition', 'zoom');
  await page.waitForTimeout(1000);

  expect(clips).toEqual([]);
  await expect(page.locator('video')).toHaveCount(0);
});

test('the toggle switches the clips on and off and remembers the choice', async ({
  page,
  hasTouch,
}) => {
  const clips = recordClipRequests(page);
  // Save "off", then drop the parameter, so the reloads below read the saved choice.
  await open(page, WITHOUT_MORPHS);
  await open(page, './');
  await expectMorph(page, false);

  await flip(morphToggle(page), hasTouch);
  await expectMorph(page, true);
  // Switched on, the clip for the edge ahead starts buffering straight away.
  await expect.poll(() => clips.some((url) => url.includes(clipLeaving(start.id)))).toBe(true);
  await expect(page.locator('.offscreen video')).toHaveCount(1);
  // Flipping the switch is not a miss and does not move on.
  await expect(page.locator('.ripple')).toHaveCount(0);
  await expect(page).toHaveURL(hashPattern(start.id));

  await page.reload();
  await waitUntilIdle(page);
  await expectMorph(page, true);

  // Switched off, the buffered clip is dropped.
  await flip(morphToggle(page), hasTouch);
  await expectMorph(page, false);
  await expect(page.locator('video')).toHaveCount(0);

  await page.reload();
  await waitUntilIdle(page);
  await expectMorph(page, false);
});

test('a ?morph= parameter overrides the saved choice and replaces it', async ({
  page,
  hasTouch,
}) => {
  await open(page, WITH_MORPHS);
  await expectMorph(page, true);
  await open(page, './');
  await expectMorph(page, true);

  await open(page, WITHOUT_MORPHS);
  await expectMorph(page, false);
  // The parameter saved its choice: without it, the dissolve stays.
  await open(page, './');
  await expectMorph(page, false);

  await open(page, `${WITH_MORPHS}#/${start.next}`);
  await expectMorph(page, true);
  await expect(page.locator('#app')).toHaveAttribute('data-scene', start.next);
  // The toggle keeps the parameter in step, so a reload does not undo the switch.
  await flip(morphToggle(page), hasTouch);
  await expect(page).toHaveURL(/\?morph=0#\//);
  await page.reload();
  await waitUntilIdle(page);
  await expectMorph(page, false);
});

test('back and forward bring an older address in line with the switch', async ({
  page,
  hasTouch,
}) => {
  await withoutClipEdges(page);
  // Both history entries are created with ?morph=1 in the address...
  await open(page, WITH_MORPHS);
  await advance(page, hasTouch);
  // ...and the switch is turned off on the second.
  await flip(morphToggle(page), hasTouch);
  await expectMorph(page, false);
  await expect(page).toHaveURL(new RegExp(`\\?morph=0#/${start.next}$`));

  // Back lands on the first entry, whose address said ?morph=1; it now says what the switch says.
  await page.goBack();
  await expect(page.locator('#app')).toHaveAttribute('data-scene', start.id);
  await expect(page).toHaveURL(new RegExp(`\\?morph=0#/${start.id}$`));
  await waitUntilIdle(page);
  await expectMorph(page, false);

  // So a reload there keeps the clips off instead of switching them back on.
  await page.reload();
  await waitUntilIdle(page);
  await expectMorph(page, false);

  await page.goForward();
  await expect(page).toHaveURL(new RegExp(`\\?morph=0#/${start.next}$`));
  await expectMorph(page, false);
});

test('Tab reaches the toggle after the Continue button, and Enter flips it', async ({ page }) => {
  await open(page, WITHOUT_MORPHS);
  await page.keyboard.press('Tab');
  await expect(page.getByRole('button', { name: CONTINUE_LABEL })).toBeFocused();
  await page.keyboard.press('Tab');
  await expect(morphToggle(page)).toBeFocused();
  await page.keyboard.press('Enter');
  await expectMorph(page, true);
  await page.keyboard.press('Enter');
  await expectMorph(page, false);
  await expect(page).toHaveURL(hashPattern(start.id));
});

test('the toggle is a 44 px or taller target that keeps the counter look', async ({ page }) => {
  await open(page, WITHOUT_MORPHS);
  const toggle = morphToggle(page);
  const box = await toggle.boundingBox();
  expect(box?.height).toBeGreaterThanOrEqual(44);
  const looks = await page.evaluate(() => {
    const style = (selector: string): string[] => {
      const element = document.querySelector(selector);
      if (!element) {
        return [];
      }
      const { fontSize, letterSpacing, fontFamily } = getComputedStyle(element);
      return [fontSize, letterSpacing, fontFamily];
    };
    return { toggle: style('.morph-toggle'), counter: style('.depth') };
  });
  expect(looks.toggle).toEqual(looks.counter);
  // Faded like the counter once it has rested, fully visible under the pointer.
  await expect(toggle).toHaveCSS('opacity', '0.4', { timeout: 6000 });
  await toggle.hover();
  await expect(toggle).toHaveCSS('opacity', '1');
});

test('switching mid-transition takes effect from the next activation', async ({
  page,
  hasTouch,
}) => {
  const clips = recordClipRequests(page);
  // Hold the third scene's image, so the start scene's preloading is still running when the
  // object is clicked, and finishes mid-transition: the moment a clip for the edge already being
  // crossed could slip out.
  const third = sceneById(sceneById(start.next).next);
  let release = (): void => undefined;
  const released = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route(`**/dreams/${third.id}/**`, async (route) => {
    await released;
    await route.continue();
  });

  await open(page, WITHOUT_MORPHS);
  await press(page, hasTouch, await hotspotCentre(page));
  await expect(page.locator('#app')).toHaveAttribute('data-busy', 'true');
  await flip(morphToggle(page), hasTouch);
  await expectMorph(page, true);
  release();

  // The transition already under way stays the dissolve...
  await expect(page).toHaveURL(hashPattern(start.next));
  await waitUntilIdle(page);
  await expect(page.locator('#app')).toHaveAttribute('data-transition', 'zoom');
  expect(clips.filter((url) => url.includes(clipLeaving(start.id)))).toEqual([]);
  // ...and the edge ahead gets its clip ready.
  await expect.poll(() => clips.some((url) => url.includes(clipLeaving(start.next)))).toBe(true);
});

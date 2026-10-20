// The morph switch: dissolve by default, clips when the visitor (or the URL) asks for them.

import { expect, test, type Locator, type Page } from '@playwright/test';

import {
  CONTINUE_LABEL,
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
  return page.getByRole('button', { name: /^morph (on|off)$/ });
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

test.beforeEach(async ({ page }) => {
  await skipIntro(page);
});

test('by default video edges play the dissolve and no clip is ever fetched', async ({
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
  await open(page);
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
  await open(page);
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
  await open(page);
  await flip(morphToggle(page), hasTouch);
  await expectMorph(page, true);

  await open(page, './?morph=0');
  await expectMorph(page, false);
  // The parameter saved its choice: without it, the dissolve stays.
  await open(page, './');
  await expectMorph(page, false);

  await open(page, `./?morph=1#/${start.next}`);
  await expectMorph(page, true);
  await expect(page.locator('#app')).toHaveAttribute('data-scene', start.next);
  // The toggle keeps the parameter in step, so a reload does not undo the switch.
  await flip(morphToggle(page), hasTouch);
  await expect(page).toHaveURL(/\?morph=0#\//);
  await page.reload();
  await waitUntilIdle(page);
  await expectMorph(page, false);
});

test('Tab reaches the toggle after the Continue button, and Enter flips it', async ({ page }) => {
  await open(page);
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

test('switching mid-transition takes effect from the next activation', async ({
  page,
  hasTouch,
}) => {
  const clips = recordClipRequests(page);
  await open(page);
  await press(page, hasTouch, await hotspotCentre(page));
  await expect(page.locator('#app')).toHaveAttribute('data-busy', 'true');
  await flip(morphToggle(page), hasTouch);
  await expectMorph(page, true);

  // The transition already under way stays the dissolve...
  await expect(page).toHaveURL(hashPattern(start.next));
  await waitUntilIdle(page);
  await expect(page.locator('#app')).toHaveAttribute('data-transition', 'zoom');
  expect(clips.filter((url) => url.includes(clipLeaving(start.id)))).toEqual([]);
  // ...and the edge ahead gets its clip ready.
  await expect.poll(() => clips.some((url) => url.includes(clipLeaving(start.next)))).toBe(true);
});

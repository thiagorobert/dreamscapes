import { expect, test } from '@playwright/test';

import { TOUCH_MARGIN_PX } from '../src/hotspot';
import {
  CONTINUE_LABEL,
  INTRO_TEXT,
  advance,
  currentScene,
  hashPattern,
  hotspotCentre,
  manifest,
  missPoint,
  open,
  press,
  sceneById,
  screenPoint,
  skipIntro,
  start,
  waitUntilIdle,
} from './helpers';

test.describe('walking the loop', () => {
  test.beforeEach(async ({ page }) => {
    await skipIntro(page);
  });

  test('opens on the start scene with the wordmark and counter', async ({ page }) => {
    await open(page);
    await expect(page).toHaveURL(hashPattern(start.id));
    await expect(page.locator('#app')).toHaveAttribute('data-scene', start.id);
    await expect(page.getByRole('img', { name: start.alt })).toBeVisible();
    await expect(page.locator('.scene-layer.is-current img')).toHaveClass(/is-loaded/);
    await expect(page.getByRole('heading', { name: 'dreamscapes' })).toBeVisible();
    await expect(page.locator('.depth')).toHaveText('dream 1');
    await expect(page.getByTestId('announcer')).toHaveText(start.title);
    await expect(page).toHaveTitle(`${start.title} · Dreamscapes`);
  });

  test('frames the focal point as near the middle as the image edges allow', async ({
    page,
    viewport,
  }) => {
    await open(page);
    const width = viewport?.width ?? 0;
    const box = await page.locator('.scene-layer.is-current').boundingBox();
    const stageWidth = box?.width ?? 0;
    const [fx] = start.focal;
    // Centred, unless that would pull the image's edge into view.
    const offset = Math.min(0, Math.max(width - stageWidth, width / 2 - fx * stageWidth));
    expect(box?.x).toBeCloseTo(offset, 0);
    const focal = await screenPoint(page, start.focal);
    expect(focal.x).toBeGreaterThan(0);
    expect(focal.x).toBeLessThan(width);
  });

  test('a click outside the object only ripples', async ({ page, hasTouch }) => {
    await open(page);
    await press(page, hasTouch, await missPoint(page));
    await expect(page.locator('.ripple')).toHaveCount(1);
    await expect(page.locator('#app')).toHaveAttribute('data-busy', 'false');
    await expect(page).toHaveURL(hashPattern(start.id));
    await expect(page.locator('.depth')).toHaveText('dream 1');
  });

  test('the object shows a pointer cursor', async ({ page }) => {
    await open(page);
    const centre = await hotspotCentre(page);
    const cursor = await page.evaluate(
      ({ x, y }) => {
        const hit = document.elementFromPoint(x, y);
        return hit ? getComputedStyle(hit).cursor : null;
      },
      { x: centre.x, y: centre.y },
    );
    expect(cursor).toBe('pointer');
  });

  test('the touch margin reaches just outside the outline, and no farther', async ({ page }) => {
    await open(page);
    const polygon = (await currentScene(page)).bridge.polygon;
    // The topmost vertex has nothing of the polygon above it, so "up" is straight outwards.
    const top = polygon.reduce((best, point) => (point[1] < best[1] ? point : best));
    const vertex = await screenPoint(page, top);
    const hitsHotspot = (pixelsOut: number): Promise<boolean> =>
      page.evaluate(
        ({ x, y }) => document.elementFromPoint(x, y)?.classList.contains('hotspot-shape') ?? false,
        { x: vertex.x, y: vertex.y - pixelsOut },
      );
    expect(await hitsHotspot(TOUCH_MARGIN_PX - 3)).toBe(true);
    expect(await hitsHotspot(TOUCH_MARGIN_PX + 5)).toBe(false);
  });

  test('activating the object at its centroid moves to the next scene', async ({
    page,
    hasTouch,
  }) => {
    await open(page);
    const next = await advance(page, hasTouch);
    expect(next.id).toBe(start.next);
    await expect(page.locator('#app')).toHaveAttribute('data-transition', 'zoom');
    await expect(page.locator('.depth')).toHaveText('dream 2');
    await expect(page.getByTestId('announcer')).toHaveText(next.title);
    await expect(page.locator('.scene-layer')).toHaveCount(1);
  });

  test('ten activations walk the loop back to the start with the counter at 11', async ({
    page,
    hasTouch,
  }) => {
    test.setTimeout(90_000);
    // The loop as the manifest describes it: every scene once, ending back at the start.
    const route: string[] = [];
    let id = start.next;
    while (id !== start.id && route.length <= manifest.scenes.length) {
      route.push(id);
      id = sceneById(id).next;
    }
    route.push(start.id);
    expect(route).toHaveLength(manifest.scenes.length);

    await open(page);
    for (const expected of route) {
      expect((await advance(page, hasTouch)).id).toBe(expected);
    }
    await expect(page.locator('.depth')).toHaveText(`dream ${String(manifest.scenes.length + 1)}`);
  });

  test('the back button returns to the previous scene and its count', async ({
    page,
    hasTouch,
  }) => {
    await open(page);
    await advance(page, hasTouch);
    await page.goBack();
    await expect(page).toHaveURL(hashPattern(start.id));
    await expect(page.locator('#app')).toHaveAttribute('data-scene', start.id);
    await waitUntilIdle(page);
    await expect(page.locator('.depth')).toHaveText('dream 1');

    await page.goForward();
    await expect(page.locator('#app')).toHaveAttribute('data-scene', start.next);
    await waitUntilIdle(page);
    await expect(page.locator('.depth')).toHaveText('dream 2');
  });

  test('Tab reaches the continue button and Enter advances', async ({ page }) => {
    await open(page);
    await page.keyboard.press('Tab');
    const button = page.getByRole('button', { name: CONTINUE_LABEL });
    await expect(button).toBeFocused();
    await page.keyboard.press('Enter');
    await expect(page).toHaveURL(hashPattern(start.next));
    await waitUntilIdle(page);
    // Focus stays put, so the visitor can keep pressing Enter.
    await expect(button).toBeFocused();
    await page.keyboard.press('Enter');
    await expect(page).toHaveURL(hashPattern(sceneById(start.next).next));
  });

  test('a reload keeps the scene and the counter', async ({ page, hasTouch }) => {
    await open(page);
    await advance(page, hasTouch);
    await page.reload();
    await waitUntilIdle(page);
    await expect(page.locator('#app')).toHaveAttribute('data-scene', start.next);
    await expect(page.locator('.depth')).toHaveText('dream 2');
  });

  test('a deep link opens that scene', async ({ page }) => {
    const target = sceneById(start.next);
    await open(page, `./#/${target.id}`);
    await expect(page.locator('#app')).toHaveAttribute('data-scene', target.id);
  });

  test('an unknown hash shows the start scene', async ({ page }) => {
    await open(page, './#/nowhere-at-all');
    await expect(page.locator('#app')).toHaveAttribute('data-scene', start.id);
    await expect(page).toHaveURL(hashPattern(start.id));
  });

  test('three misses make the object shimmer', async ({ page, hasTouch }) => {
    await open(page);
    const shimmer = page.locator('.hotspot.is-shimmering');
    for (let miss = 0; miss < 3; miss += 1) {
      await expect(shimmer).toHaveCount(0);
      await press(page, hasTouch, await missPoint(page));
    }
    await expect(shimmer).toHaveCount(1);
    await expect(page).toHaveURL(hashPattern(start.id));
  });

  test('twenty idle seconds make the object shimmer', async ({ page }) => {
    await page.clock.install();
    await open(page);
    const shimmer = page.locator('.hotspot.is-shimmering');
    await page.clock.fastForward(19_000);
    await expect(shimmer).toHaveCount(0);
    await page.clock.fastForward(1_500);
    await expect(shimmer).toHaveCount(1);
  });

  test('dragging pans within the image edges and is not a miss', async ({ page, viewport }) => {
    await open(page);
    const layer = page.locator('.scene-layer.is-current');
    const width = viewport?.width ?? 0;
    const height = viewport?.height ?? 0;
    const y = height / 2;

    // Pointer events only arrive inside the viewport, so each drag crosses most of the screen and
    // is repeated until it is certainly past the image edge.
    const dragAcross = async (fromX: number, toX: number): Promise<void> => {
      for (let pass = 0; pass < 4; pass += 1) {
        await page.mouse.move(fromX, y);
        await page.mouse.down();
        await page.mouse.move((fromX + toX) / 2, y, { steps: 4 });
        await page.mouse.move(toX, y, { steps: 4 });
        await page.mouse.up();
      }
    };

    // Rightwards: the image's left edge stops at the screen's left edge.
    await dragAcross(width * 0.05, width * 0.95);
    expect((await layer.boundingBox())?.x).toBeCloseTo(0, 0);
    // Leftwards: the image's right edge stops at the screen's right edge.
    await dragAcross(width * 0.95, width * 0.05);
    const box = await layer.boundingBox();
    expect((box?.x ?? 0) + (box?.width ?? 0)).toBeCloseTo(width, 0);

    await expect(page.locator('.ripple')).toHaveCount(0);
    await expect(page).toHaveURL(hashPattern(start.id));
  });

  test('a press released outside the window leaves no drag behind', async ({ page, viewport }) => {
    await open(page);
    const layer = page.locator('.scene-layer.is-current');
    const before = (await layer.boundingBox())?.x;
    const width = viewport?.width ?? 0;
    const y = (viewport?.height ?? 0) / 2;
    // Press, leave the window before the press turns into a pan, and let go out there: the page
    // never sees the release.
    await page.mouse.move(width / 2, y);
    await page.mouse.down();
    await page.mouse.move(-40, y);
    await page.mouse.up();
    // Coming back with the button up must not pan.
    await page.mouse.move(width / 2 + 60, y, { steps: 3 });
    await page.mouse.move(width / 2 + 160, y, { steps: 3 });
    expect((await layer.boundingBox())?.x).toBe(before);
    await expect(page.locator('.stage')).not.toHaveClass(/is-panning/);
  });

  test('a finger swipe pans the stage', async ({ page, hasTouch, viewport }) => {
    test.skip(!hasTouch, 'touch devices only');
    await open(page);
    const layer = page.locator('.scene-layer.is-current');
    const before = (await layer.boundingBox())?.x ?? 0;
    const y = (viewport?.height ?? 0) / 2;
    // Playwright's touchscreen only taps, so the swipe goes through the DevTools protocol.
    const cdp = await page.context().newCDPSession(page);
    await cdp.send('Input.dispatchTouchEvent', {
      type: 'touchStart',
      touchPoints: [{ x: 150, y }],
    });
    for (const x of [170, 200, 230, 250]) {
      await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x, y }] });
    }
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
    const after = (await layer.boundingBox())?.x ?? 0;
    expect(after - before).toBeCloseTo(100, 0);
    await expect(page.locator('.ripple')).toHaveCount(0);
    await expect(page).toHaveURL(hashPattern(start.id));
  });

  test.describe('with reduced motion', () => {
    test.use({ reducedMotion: 'reduce' });

    test('still navigates, with a plain crossfade', async ({ page, hasTouch }) => {
      await open(page);
      await advance(page, hasTouch);
      await expect(page.locator('#app')).toHaveAttribute('data-transition', 'fade');
      await advance(page, hasTouch);
      await expect(page.locator('.depth')).toHaveText('dream 3');
    });

    test('the hint is a still highlight', async ({ page, hasTouch }) => {
      await open(page);
      for (let miss = 0; miss < 3; miss += 1) {
        await press(page, hasTouch, await missPoint(page));
      }
      await expect(page.locator('.hotspot.is-highlighted')).toHaveCount(1);
      await expect(page.locator('.hotspot.is-shimmering')).toHaveCount(0);
    });
  });
});

test.describe('first visit', () => {
  test('explains the rule once; the dismissing tap does nothing else', async ({
    page,
    hasTouch,
  }) => {
    await open(page);
    const intro = page.getByText(INTRO_TEXT);
    await expect(intro).toBeVisible();
    // Even a tap right on the object only dismisses the overlay.
    await press(page, hasTouch, await hotspotCentre(page));
    await expect(intro).toHaveCount(0);
    await expect(page.locator('.ripple')).toHaveCount(0);
    await expect(page).toHaveURL(hashPattern(start.id));
    expect((await currentScene(page)).id).toBe(start.id);

    await page.reload();
    await waitUntilIdle(page);
    await expect(page.locator('.scene-layer.is-current')).toHaveCount(1);
    await expect(intro).toHaveCount(0);
  });
});

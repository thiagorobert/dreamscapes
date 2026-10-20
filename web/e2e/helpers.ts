import { readFileSync } from 'node:fs';

import { expect, type Page } from '@playwright/test';

import type { Manifest, Point, Scene } from '../src/types';

export const manifest = JSON.parse(
  readFileSync(new URL('../public/scenes.json', import.meta.url), 'utf8'),
) as Manifest;

const byId = new Map(manifest.scenes.map((scene) => [scene.id, scene]));

export function sceneById(id: string): Scene {
  const scene = byId.get(id);
  if (!scene) {
    throw new Error(`No scene "${id}" in scenes.json`);
  }
  return scene;
}

export const start = sceneById(manifest.start);

export const INTRO_TEXT = 'Something in each dream leads to the next. Find it.';
export const CONTINUE_LABEL = 'Continue to the next dream';

export interface ScreenPoint {
  x: number;
  y: number;
}

/**
 * The site with the morph clips switched on. Without it every edge plays the dissolve (the
 * default), so the loop tests take about a second a step instead of a clip's eight;
 * morph.spec.ts opens this address to cover the clips.
 */
export const WITH_MORPHS = './?morph=1';

/** Every request for a morph clip the page makes from now on, as a live list of URLs. */
export function recordClipRequests(page: Page): string[] {
  const urls: string[] = [];
  page.on('request', (request) => {
    if (new URL(request.url()).pathname.includes('/morphs/')) {
      urls.push(request.url());
    }
  });
  return urls;
}

/** Marks the first-visit overlay as already seen, so tests start on a bare scene. */
export async function skipIntro(page: Page): Promise<void> {
  await page.addInitScript(() => {
    try {
      localStorage.setItem('dreamscapes:intro-seen', '1');
    } catch {
      // Storage blocked: the overlay will show and the test will notice.
    }
  });
}

export function hashPattern(id: string): RegExp {
  return new RegExp(`#/${id}$`);
}

/** Waits until no transition is running. */
export async function waitUntilIdle(page: Page): Promise<void> {
  await expect(page.locator('#app')).toHaveAttribute('data-busy', 'false');
}

/** Opens the site and waits for the start scene to settle. */
export async function open(page: Page, path = './'): Promise<void> {
  await page.goto(path);
  await expect(page.locator('.scene-layer.is-current')).toHaveCount(1);
  await waitUntilIdle(page);
}

export async function currentScene(page: Page): Promise<Scene> {
  const id = await page.locator('#app').getAttribute('data-scene');
  return sceneById(id ?? '');
}

/** Viewport coordinates of a point given in the current scene's normalised image coordinates. */
export async function screenPoint(page: Page, [x, y]: Point): Promise<ScreenPoint> {
  const box = await page.locator('.scene-layer.is-current').boundingBox();
  if (!box) {
    throw new Error('No scene on stage');
  }
  return { x: box.x + x * box.width, y: box.y + y * box.height };
}

export async function hotspotCentre(page: Page): Promise<ScreenPoint> {
  return screenPoint(page, (await currentScene(page)).bridge.centroid);
}

/** A tap on touch devices, a click everywhere else. */
export async function press(page: Page, hasTouch: boolean, point: ScreenPoint): Promise<void> {
  if (hasTouch) {
    await page.touchscreen.tap(point.x, point.y);
  } else {
    await page.mouse.click(point.x, point.y);
  }
}

/**
 * A point on the stage well clear of the hotspot (and of its touch margin), checked with the
 * browser's own hit testing.
 */
export async function missPoint(page: Page): Promise<ScreenPoint> {
  const point = await page.evaluate(() => {
    const clear = (x: number, y: number): boolean =>
      [-24, 0, 24].every((dx) =>
        [-24, 0, 24].every((dy) => {
          const hit = document.elementFromPoint(x + dx, y + dy);
          return hit !== null && hit.closest('.stage') !== null && hit.closest('.hotspot') === null;
        }),
      );
    const { innerWidth: width, innerHeight: height } = window;
    for (const fy of [0.5, 0.75, 0.25, 0.9, 0.1]) {
      for (const fx of [0.15, 0.85, 0.5, 0.3, 0.7]) {
        if (clear(width * fx, height * fy)) {
          return { x: width * fx, y: height * fy };
        }
      }
    }
    return null;
  });
  if (!point) {
    throw new Error('Found no point on the stage away from the hotspot');
  }
  return point;
}

/** Activates the current scene's hotspot and waits for the next scene to settle. */
export async function advance(page: Page, hasTouch: boolean): Promise<Scene> {
  const from = await currentScene(page);
  await press(page, hasTouch, await hotspotCentre(page));
  await expect(page).toHaveURL(hashPattern(from.next));
  await waitUntilIdle(page);
  await expect(page.locator('#app')).toHaveAttribute('data-scene', from.next);
  return sceneById(from.next);
}

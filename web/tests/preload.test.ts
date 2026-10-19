import { describe, expect, test } from 'vitest';

import {
  MAX_PIXEL_RATIO,
  PHONE_MAX_WIDTH,
  neededPixelWidth,
  pickSource,
  pixelRatioCap,
} from '../src/preload';
import { coverSize } from '../src/stage';
import type { ImageSource } from '../src/types';

const sources: ImageSource[] = [640, 1024, 1536, 2048].map((width) => ({
  width,
  src: `dreams/x/x-${String(width)}.webp`,
}));

const RATIO = 16 / 9;

/** The file width the site would fetch for a viewport, as main.ts wires it up. */
function chosenWidth(viewport: { width: number; height: number }, pixelRatio: number): number {
  const stageWidth = coverSize(viewport, RATIO).width;
  return pickSource(sources, neededPixelWidth(stageWidth, pixelRatio, viewport.width))?.width ?? 0;
}

describe('pickSource', () => {
  test('picks the smallest file at least as wide as needed', () => {
    expect(pickSource(sources, 1000)?.width).toBe(1024);
    expect(pickSource(sources, 1024)?.width).toBe(1024);
    expect(pickSource(sources, 1025)?.width).toBe(1536);
    expect(pickSource(sources, 1)?.width).toBe(640);
  });

  test('falls back to the largest file when none is wide enough', () => {
    expect(pickSource(sources, 4500)?.width).toBe(2048);
  });

  test('does not rely on the list being sorted', () => {
    expect(pickSource([...sources].reverse(), 1500)?.width).toBe(1536);
  });

  test('returns nothing for an empty list (scenes without AVIF)', () => {
    expect(pickSource([], 1000)).toBeUndefined();
  });
});

describe('neededPixelWidth', () => {
  const desktop = 1440;
  const phone = 390;

  test('scales by the device pixel ratio', () => {
    expect(neededPixelWidth(800, 1, desktop)).toBe(800);
    expect(neededPixelWidth(800, 1.5, desktop)).toBe(1200);
  });

  test('caps the pixel ratio at 2 on larger screens', () => {
    expect(neededPixelWidth(1000, 3, desktop)).toBe(1000 * MAX_PIXEL_RATIO);
  });

  test('caps the pixel ratio at 1 on phone-width screens', () => {
    expect(pixelRatioCap(PHONE_MAX_WIDTH - 1)).toBe(1);
    expect(pixelRatioCap(PHONE_MAX_WIDTH)).toBe(MAX_PIXEL_RATIO);
    expect(neededPixelWidth(1500, 3, phone)).toBe(1500);
  });

  test('never asks for less than one pixel per CSS pixel', () => {
    expect(neededPixelWidth(800, 0.5, desktop)).toBe(800);
  });
});

describe('the file each device fetches', () => {
  test('an iPhone 13 fetches the 1536 file, not the 2048 one', () => {
    // The phone's screen at 3x, and the shorter viewport Playwright's iPhone 13 profile uses.
    expect(chosenWidth({ width: 390, height: 844 }, 3)).toBe(1536);
    expect(chosenWidth({ width: 390, height: 664 }, 3)).toBe(1536);
  });

  test('a 1440x900 desktop fetches the 2048 file at 1x and 2x', () => {
    // 1440x900 is narrower than 16:9, so the stage is 1600 CSS px wide.
    expect(chosenWidth({ width: 1440, height: 900 }, 1)).toBe(2048);
    expect(chosenWidth({ width: 1440, height: 900 }, 2)).toBe(2048);
  });

  test('a small laptop window at 1x fetches a smaller file', () => {
    expect(chosenWidth({ width: 900, height: 506 }, 1)).toBe(1024);
  });
});

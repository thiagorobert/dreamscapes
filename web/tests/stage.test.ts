import { describe, expect, test } from 'vitest';

import { clampOffset, coverSize, focalOffset, type Offset, type Size } from '../src/stage';

const RATIO = 16 / 9;

function expectOffset(actual: Offset, expected: Offset): void {
  expect(actual.x).toBeCloseTo(expected.x, 6);
  expect(actual.y).toBeCloseTo(expected.y, 6);
}

describe('coverSize', () => {
  test('a viewport wider than the image fills the width and overflows vertically', () => {
    const size = coverSize({ width: 2560, height: 1080 }, RATIO);
    expect(size.width).toBe(2560);
    expect(size.height).toBeCloseTo(1440, 6);
  });

  test('a viewport taller than the image fills the height and overflows horizontally', () => {
    const size = coverSize({ width: 390, height: 844 }, RATIO);
    expect(size.height).toBe(844);
    expect(size.width).toBeCloseTo(1500.444, 3);
  });

  test('a viewport with exactly the image ratio fits without overflow', () => {
    expect(coverSize({ width: 1600, height: 900 }, RATIO)).toEqual({ width: 1600, height: 900 });
  });

  test('a desktop viewport slightly narrower than 16:9 overflows a little horizontally', () => {
    const size = coverSize({ width: 1440, height: 900 }, RATIO);
    expect(size.height).toBe(900);
    expect(size.width).toBeCloseTo(1600, 6);
  });

  test('never divides by a zero-sized viewport', () => {
    const size = coverSize({ width: 0, height: 0 }, RATIO);
    expect(Number.isFinite(size.width)).toBe(true);
    expect(Number.isFinite(size.height)).toBe(true);
  });
});

describe('focalOffset', () => {
  test('wide viewport: centres the focal point vertically', () => {
    const viewport = { width: 2560, height: 1080 };
    const stage = coverSize(viewport, RATIO);
    // 1080 / 2 - 0.5 * 1440 = -180; no horizontal overflow.
    expectOffset(focalOffset(viewport, stage, [0.3, 0.5]), { x: 0, y: -180 });
  });

  test('tall viewport: centres the focal point horizontally', () => {
    const viewport = { width: 390, height: 844 };
    const stage = coverSize(viewport, RATIO);
    const offset = focalOffset(viewport, stage, [0.7, 0.3]);
    expectOffset(offset, { x: 195 - 0.7 * stage.width, y: 0 });
    // The focal point lands in the middle of the screen.
    expect(offset.x + 0.7 * stage.width).toBeCloseTo(195, 6);
  });

  test('exact ratio: no offset whatever the focal point', () => {
    const viewport = { width: 1600, height: 900 };
    const stage = coverSize(viewport, RATIO);
    expectOffset(focalOffset(viewport, stage, [0.9, 0.1]), { x: 0, y: 0 });
  });

  test('a focal point near an edge stops at that edge instead of opening a gap', () => {
    const viewport = { width: 390, height: 844 };
    const stage = coverSize(viewport, RATIO);
    expectOffset(focalOffset(viewport, stage, [0.02, 0.5]), { x: 0, y: 0 });
    expectOffset(focalOffset(viewport, stage, [0.98, 0.5]), { x: 390 - stage.width, y: 0 });
  });
});

describe('clampOffset (panning bounds)', () => {
  const viewport: Size = { width: 390, height: 844 };
  const stage = coverSize(viewport, RATIO);
  const minX = viewport.width - stage.width;

  test('keeps an offset that is already within bounds', () => {
    expectOffset(clampOffset(viewport, stage, { x: -500, y: 0 }), { x: -500, y: 0 });
  });

  test('stops at the left edge of the image when dragged right', () => {
    expectOffset(clampOffset(viewport, stage, { x: 300, y: 0 }), { x: 0, y: 0 });
  });

  test('stops at the right edge of the image when dragged left', () => {
    expectOffset(clampOffset(viewport, stage, { x: -5000, y: 0 }), { x: minX, y: 0 });
  });

  test('does not pan along an axis without overflow', () => {
    expectOffset(clampOffset(viewport, stage, { x: -500, y: -120 }), { x: -500, y: 0 });
    expectOffset(clampOffset(viewport, stage, { x: -500, y: 120 }), { x: -500, y: 0 });
  });

  test('pans vertically on a wide viewport', () => {
    const wide = { width: 2560, height: 1080 };
    const wideStage = coverSize(wide, RATIO);
    expectOffset(clampOffset(wide, wideStage, { x: -40, y: -1000 }), { x: 0, y: -360 });
  });
});

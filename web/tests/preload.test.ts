import { describe, expect, test } from 'vitest';

import { MAX_PIXEL_RATIO, neededPixelWidth, pickSource } from '../src/preload';
import type { ImageSource } from '../src/types';

const sources: ImageSource[] = [640, 1024, 1536, 2048].map((width) => ({
  width,
  src: `dreams/x/x-${String(width)}.webp`,
}));

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
  test('scales by the device pixel ratio', () => {
    expect(neededPixelWidth(800, 1)).toBe(800);
    expect(neededPixelWidth(800, 1.5)).toBe(1200);
  });

  test('caps the pixel ratio', () => {
    expect(neededPixelWidth(1000, 3)).toBe(1000 * MAX_PIXEL_RATIO);
  });

  test('never asks for less than one pixel per CSS pixel', () => {
    expect(neededPixelWidth(800, 0.5)).toBe(800);
  });

  test('a desktop at 1440x900 needs the 2048 file; a phone at 1x the 1536 file', () => {
    // 1440x900 is narrower than 16:9, so the stage is 1600 CSS px wide.
    expect(pickSource(sources, neededPixelWidth(1600, 1))?.width).toBe(2048);
    expect(pickSource(sources, neededPixelWidth(1500, 1))?.width).toBe(1536);
  });
});

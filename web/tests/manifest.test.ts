import { readFileSync } from 'node:fs';

import Ajv2020 from 'ajv/dist/2020.js';
import { describe, expect, test } from 'vitest';

import { ManifestError, parseManifest } from '../src/manifest';
import type { Manifest, Point } from '../src/types';

function readJson(relative: string): unknown {
  return JSON.parse(readFileSync(new URL(relative, import.meta.url), 'utf8'));
}

const schema = readJson('../../schema/scenes.schema.json') as Record<string, unknown>;
const manifest = readJson('../public/scenes.json') as Manifest;

describe('public/scenes.json', () => {
  test('matches the schema exported by the pipeline', () => {
    const ajv = new Ajv2020({ allErrors: true, strict: true });
    const validate = ajv.compile(schema);
    validate(manifest);
    expect(validate.errors ?? []).toEqual([]);
  });

  test('names a start scene and every next pointer resolves', () => {
    const ids = new Set(manifest.scenes.map((scene) => scene.id));
    expect(ids.has(manifest.start)).toBe(true);
    for (const scene of manifest.scenes) {
      expect(ids.has(scene.next)).toBe(true);
    }
  });
});

/** Even-odd ray casting: is `point` inside `polygon`? */
function insidePolygon([x, y]: Point, polygon: readonly Point[]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const [xi, yi] = polygon[i] ?? [0, 0];
    const [xj, yj] = polygon[j] ?? [0, 0];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

describe('hotspot polygons', () => {
  test('the point-in-polygon check itself is sound', () => {
    const square: Point[] = [
      [0.2, 0.2],
      [0.4, 0.2],
      [0.4, 0.4],
      [0.2, 0.4],
    ];
    expect(insidePolygon([0.3, 0.3], square)).toBe(true);
    expect(insidePolygon([0.5, 0.3], square)).toBe(false);
  });

  // The end-to-end tests click each scene at its centroid, and so does a visitor aiming for the
  // middle of the object; a centroid outside its own polygon (possible for concave shapes) would
  // turn that click into a miss.
  test.each(manifest.scenes.map((scene) => [scene.id, scene] as const))(
    '%s has its centroid inside its polygon',
    (_id, scene) => {
      expect(insidePolygon(scene.bridge.centroid, scene.bridge.polygon)).toBe(true);
    },
  );
});

describe('parseManifest', () => {
  test('accepts the committed manifest and indexes every scene', () => {
    const dreams = parseManifest(manifest);
    expect(dreams.start.id).toBe(manifest.start);
    expect([...dreams.scenes.keys()]).toEqual(manifest.scenes.map((scene) => scene.id));
  });

  test('rejects a manifest without scenes', () => {
    expect(() => parseManifest({ version: 1, start: 'x', scenes: [] })).toThrow(ManifestError);
    expect(() => parseManifest('nonsense')).toThrow(ManifestError);
  });

  test('describes what is wrong', () => {
    const broken = structuredClone(manifest);
    const [first, second] = broken.scenes;
    if (!first || !second) {
      throw new Error('the committed manifest needs at least two scenes');
    }
    first.bridge.polygon = first.bridge.polygon.slice(0, 2);
    expect(() => parseManifest(broken)).toThrow(/fewer than 3 points/);

    first.bridge.polygon = manifest.scenes[0]?.bridge.polygon ?? [];
    second.next = 'nowhere';
    expect(() => parseManifest(broken)).toThrow(/leads to "nowhere"/);
  });

  test('rejects an unknown start scene', () => {
    expect(() => parseManifest({ ...manifest, start: 'nowhere' })).toThrow(/start scene/);
  });

  test('rejects a video transition without a clip', () => {
    const broken = structuredClone(manifest);
    const first = broken.scenes[0];
    if (!first) {
      throw new Error('the committed manifest has no scenes');
    }
    first.transition = { type: 'video', video: '', duration: 2 };
    expect(() => parseManifest(broken)).toThrow(/without a clip/);
  });
});

import { readFileSync } from 'node:fs';

import Ajv2020 from 'ajv/dist/2020.js';
import { describe, expect, test } from 'vitest';

import type { Manifest } from '../src/types';

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

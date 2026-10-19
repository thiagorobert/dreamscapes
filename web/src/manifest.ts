/**
 * Loads `scenes.json` and checks it is usable before anything renders.
 *
 * The pipeline validates the manifest thoroughly against its schema, and a unit test checks the
 * committed file against that schema, so these runtime checks are deliberately light: they catch
 * a truncated or hand-edited file with a readable message instead of a blank page later on.
 */

import type { Manifest, Point, Scene } from './types';

export class ManifestError extends Error {
  override name = 'ManifestError';
}

export interface Dreams {
  start: Scene;
  /** Every scene by id; `next` pointers are guaranteed to resolve. */
  scenes: ReadonlyMap<string, Scene>;
}

/** Turns a site-relative path from the manifest into a URL that honours Vite's base path. */
export function assetUrl(path: string): string {
  return `${import.meta.env.BASE_URL}${path.replace(/^\/+/, '')}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isUnit(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1;
}

function isPoint(value: unknown): value is Point {
  return Array.isArray(value) && value.length === 2 && value.every(isUnit);
}

function nonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0;
}

/** Problems with one scene, described for a person reading the console. */
function sceneProblems(scene: unknown, index: number): string[] {
  const where = `scenes[${String(index)}]`;
  if (!isRecord(scene)) {
    return [`${where} is not an object`];
  }
  const problems: string[] = [];
  const label = nonEmptyString(scene.id) ? `scene "${scene.id}"` : where;
  for (const field of ['id', 'title', 'alt', 'next'] as const) {
    if (!nonEmptyString(scene[field])) {
      problems.push(`${label} has no ${field}`);
    }
  }
  if (!isPoint(scene.focal)) {
    problems.push(`${label} has no focal point`);
  }

  const bridge = scene.bridge;
  if (!isRecord(bridge)) {
    problems.push(`${label} has no bridge object`);
  } else {
    if (!Array.isArray(bridge.polygon) || bridge.polygon.length < 3) {
      problems.push(`${label} has a hotspot polygon with fewer than 3 points`);
    } else if (!bridge.polygon.every(isPoint)) {
      problems.push(`${label} has a hotspot point outside the 0-1 range`);
    }
    if (!isPoint(bridge.centroid)) {
      problems.push(`${label} has no hotspot centroid`);
    }
  }

  const image = scene.image;
  if (!isRecord(image) || !isRecord(image.sources)) {
    problems.push(`${label} has no image`);
  } else {
    const { width, height, sources, lqip } = image;
    if (typeof width !== 'number' || typeof height !== 'number' || width <= 0 || height <= 0) {
      problems.push(`${label} has no image size`);
    }
    if (!Array.isArray(sources.webp) || sources.webp.length === 0) {
      problems.push(`${label} has no WebP image`);
    }
    if (!Array.isArray(sources.avif)) {
      problems.push(`${label} has no AVIF list`);
    }
    if (!nonEmptyString(lqip)) {
      problems.push(`${label} has no placeholder image`);
    }
  }

  const transition = scene.transition;
  if (!isRecord(transition)) {
    problems.push(`${label} has no transition`);
  } else if (transition.type === 'video') {
    if (!nonEmptyString(transition.video)) {
      problems.push(`${label} has a video transition without a clip`);
    }
    if (typeof transition.duration !== 'number' || !(transition.duration > 0)) {
      problems.push(`${label} has a video transition without a duration`);
    }
  } else if (transition.type !== 'zoom') {
    problems.push(`${label} has an unknown transition type`);
  }
  return problems;
}

/** Checks the parsed JSON and indexes the scenes. Throws `ManifestError` listing every problem. */
export function parseManifest(data: unknown): Dreams {
  if (!isRecord(data) || !Array.isArray(data.scenes) || data.scenes.length === 0) {
    throw new ManifestError('scenes.json has no scenes');
  }
  const problems = data.scenes.flatMap(sceneProblems);
  if (problems.length > 0) {
    throw new ManifestError(`scenes.json is not usable: ${problems.join('; ')}`);
  }

  const manifest = data as unknown as Manifest;
  const scenes = new Map<string, Scene>();
  for (const scene of manifest.scenes) {
    if (scenes.has(scene.id)) {
      problems.push(`scene "${scene.id}" appears twice`);
    }
    scenes.set(scene.id, scene);
  }
  for (const scene of manifest.scenes) {
    if (!scenes.has(scene.next)) {
      problems.push(`scene "${scene.id}" leads to "${scene.next}", which does not exist`);
    }
  }
  const start = scenes.get(manifest.start);
  if (!start) {
    problems.push(`the start scene "${manifest.start}" does not exist`);
  }
  if (problems.length > 0 || !start) {
    throw new ManifestError(`scenes.json is not usable: ${problems.join('; ')}`);
  }
  return { start, scenes };
}

export async function loadManifest(url = assetUrl('scenes.json')): Promise<Dreams> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new ManifestError(`Could not load scenes.json (HTTP ${String(response.status)})`);
  }
  return parseManifest(await response.json());
}

/** The scene `scene` leads to. `parseManifest` guarantees it exists. */
export function nextScene(dreams: Dreams, scene: Scene): Scene {
  return dreams.scenes.get(scene.next) ?? dreams.start;
}

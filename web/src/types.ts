/**
 * Types for `public/scenes.json`, the built manifest.
 *
 * Mirrors `schema/scenes.schema.json`, which the pipeline exports from its pydantic models
 * (`pipeline/dreamscapes_pipeline/manifest.py`, the source of truth). Keep the two in step;
 * `tests/manifest.test.ts` validates the committed `scenes.json` against the schema.
 *
 * Coordinates are normalised to the image: x and y run from 0 to 1, origin top-left.
 */

/** Lower-case words joined by single hyphens, e.g. `prison-yard`. */
export type Slug = string;

/** `[x, y]` in normalised image coordinates. */
export type Point = [x: number, y: number];

/** `[x0, y0, x1, y1]` axis-aligned box in normalised image coordinates. */
export type BBox = [x0: number, y0: number, x1: number, y1: number];

export interface ImageSource {
  /** Pixel width of this file. */
  width: number;
  /** Path relative to the site root (prefix with `import.meta.env.BASE_URL`). */
  src: string;
}

/** Files per format, sorted by ascending width. `avif` may be empty; `webp` never is. */
export interface ImageSources {
  avif: ImageSource[];
  webp: ImageSource[];
}

export interface SceneImage {
  /** Size of the master the exports were made from (2048 x 1152). */
  width: number;
  height: number;
  sources: ImageSources;
  /** Tiny blurred placeholder as a `data:image/...;base64,` URL. */
  lqip: string;
}

/** The scene's single clickable object. */
export interface Bridge {
  /** Noun phrase naming the object, e.g. "basketball hoop". */
  object: string;
  /** Hotspot outline, at least 3 points. */
  polygon: Point[];
  bbox: BBox;
  centroid: Point;
  /** Fraction of the frame the polygon covers (0-1). */
  area: number;
}

export interface ZoomTransition {
  type: 'zoom';
}

export interface VideoTransition {
  type: 'video';
  /** Clip path relative to the site root. */
  video: string;
  /** Clip length in seconds. */
  duration: number;
}

export type Transition = ZoomTransition | VideoTransition;

export interface Scene {
  id: Slug;
  title: string;
  /** Describes the scene, never where the hotspot is. */
  alt: string;
  /** Scene reached by activating the bridge object. */
  next: Slug;
  /** Pan focus on narrow viewports; the pipeline resolves it to the hotspot centroid by default. */
  focal: Point;
  bridge: Bridge;
  image: SceneImage;
  transition: Transition;
}

export interface Manifest {
  version: number;
  /** Scene shown when the page opens without a hash. */
  start: Slug;
  scenes: Scene[];
}

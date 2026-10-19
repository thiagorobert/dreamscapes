/**
 * Image and clip preloading.
 *
 * Each scene's image is built as a detached `<picture>` (AVIF and WebP sources of the width the
 * viewport needs), loaded and decoded ahead of time, and later moved onto the stage as is. Reusing
 * the decoded element means the stage never waits and the browser never fetches a second copy in
 * another format. A small least-recently-used cache holds the current scene, the next two, and the
 * previous one for the back button.
 *
 * Morph clips are preloaded with a muted, off-screen `<video preload="auto">`, because
 * `<link rel="preload" as="video">` is not reliably honoured by browsers.
 */

import type { ImageSource, Scene } from './types';

/**
 * Highest device-pixel ratio worth paying for. The images are soft and painterly; beyond 2x the
 * extra pixels are invisible but the bytes are not.
 */
export const MAX_PIXEL_RATIO = 2;

/**
 * Viewports narrower than this (CSS px) are treated as phones and get one image pixel per CSS
 * pixel. In portrait the cover box is three to four times the screen width, so even at 1x the
 * picked file is the 1536 one; at 2x every phone would download the 2048 file, mostly for the
 * parts of the image that are off-screen.
 */
export const PHONE_MAX_WIDTH = 768;

/** The pixel-ratio ceiling for a viewport `viewportWidth` CSS pixels wide. */
export function pixelRatioCap(viewportWidth: number): number {
  return viewportWidth < PHONE_MAX_WIDTH ? 1 : MAX_PIXEL_RATIO;
}

/** Decoded pictures kept: current, next, next-next and previous. */
export const PICTURE_CACHE_SIZE = 4;

/**
 * The smallest candidate at least `neededWidth` pixels wide, or the largest one when none is wide
 * enough. Returns undefined only for an empty list.
 */
export function pickSource(
  sources: readonly ImageSource[],
  neededWidth: number,
): ImageSource | undefined {
  const sorted = [...sources].sort((a, b) => a.width - b.width);
  return sorted.find((source) => source.width >= neededWidth) ?? sorted.at(-1);
}

/**
 * Image pixels worth fetching to draw an image `cssWidth` CSS pixels wide at `pixelRatio`, on a
 * viewport `viewportWidth` CSS pixels wide.
 */
export function neededPixelWidth(
  cssWidth: number,
  pixelRatio: number,
  viewportWidth: number,
): number {
  return Math.ceil(cssWidth * Math.min(Math.max(pixelRatio, 1), pixelRatioCap(viewportWidth)));
}

export interface PreparedPicture {
  readonly scene: Scene;
  readonly picture: HTMLPictureElement;
  readonly img: HTMLImageElement;
  /** Resolves true once the image is decoded, false if it failed (the placeholder stays). */
  readonly ready: Promise<boolean>;
  loaded: boolean;
  /** Width of the chosen WebP candidate, to know when a larger viewport needs a bigger file. */
  width: number;
  readonly avifSource: HTMLSourceElement | undefined;
  readonly webpSource: HTMLSourceElement;
}

export interface PreloaderOptions {
  /** CSS width the scene's image will be drawn at. */
  cssWidthFor: (scene: Scene) => number;
  /** Current viewport width in CSS pixels. */
  viewportWidth: () => number;
  /** Turns a manifest path into a URL. */
  resolve: (path: string) => string;
  /** Where off-screen clip elements live while they buffer. */
  clipHost: HTMLElement;
  pixelRatio?: () => number;
  capacity?: number;
}

function waitForLoad(img: HTMLImageElement): Promise<void> {
  return new Promise((resolve, reject) => {
    img.addEventListener(
      'load',
      () => {
        resolve();
      },
      { once: true },
    );
    img.addEventListener(
      'error',
      () => {
        reject(new Error(`Could not load ${img.currentSrc || img.src}`));
      },
      { once: true },
    );
  });
}

/** A muted, inline, buffering video element for `url`. */
export function createClip(url: string): HTMLVideoElement {
  const video = document.createElement('video');
  video.muted = true;
  video.defaultMuted = true;
  video.playsInline = true;
  // iOS Safari reads the attributes, not only the properties, when deciding about inline playback.
  video.setAttribute('muted', '');
  video.setAttribute('playsinline', '');
  video.disablePictureInPicture = true;
  video.preload = 'auto';
  video.src = url;
  return video;
}

/** Stops a clip and lets the browser drop its buffer. */
export function releaseClip(video: HTMLVideoElement): void {
  video.pause();
  video.removeAttribute('src');
  video.load();
  video.remove();
}

export class Preloader {
  private readonly pictures = new Map<string, PreparedPicture>();
  private readonly clips = new Map<string, HTMLVideoElement>();
  private readonly capacity: number;
  private readonly pixelRatio: () => number;

  constructor(private readonly options: PreloaderOptions) {
    this.capacity = options.capacity ?? PICTURE_CACHE_SIZE;
    this.pixelRatio = options.pixelRatio ?? (() => window.devicePixelRatio);
  }

  /** The prepared picture for `scene`, from the cache or freshly started. */
  picture(scene: Scene): PreparedPicture {
    const needed = neededPixelWidth(
      this.options.cssWidthFor(scene),
      this.pixelRatio(),
      this.options.viewportWidth(),
    );
    const cached = this.pictures.get(scene.id);
    if (cached) {
      // Re-insert to mark it most recently used.
      this.pictures.delete(scene.id);
      this.pictures.set(scene.id, cached);
      this.upgrade(cached, needed);
      return cached;
    }
    const prepared = this.build(scene, needed);
    this.pictures.set(scene.id, prepared);
    for (const id of this.pictures.keys()) {
      if (this.pictures.size <= this.capacity) {
        break;
      }
      // Dropping the reference is enough: a picture on stage stays there, and the browser's
      // own cache usually still has the file.
      this.pictures.delete(id);
    }
    return prepared;
  }

  /** Loads and decodes `scenes` one after another, in priority order. */
  async warm(scenes: readonly Scene[]): Promise<void> {
    for (const scene of scenes) {
      await this.picture(scene).ready;
    }
  }

  /** Starts buffering the morph clip that leaves `scene`, dropping any other buffered clip. */
  preloadClip(scene: Scene): void {
    for (const [id, video] of this.clips) {
      if (id !== scene.id) {
        releaseClip(video);
        this.clips.delete(id);
      }
    }
    if (scene.transition.type !== 'video' || this.clips.has(scene.id)) {
      return;
    }
    const video = createClip(this.options.resolve(scene.transition.video));
    this.options.clipHost.append(video);
    this.clips.set(scene.id, video);
  }

  /**
   * Hands over the clip that leaves `scene` (buffered if possible, otherwise just started), or
   * null when that edge has no clip. The caller owns the element afterwards.
   */
  takeClip(scene: Scene): HTMLVideoElement | null {
    if (scene.transition.type !== 'video') {
      return null;
    }
    const buffered = this.clips.get(scene.id);
    if (buffered) {
      this.clips.delete(scene.id);
      return buffered;
    }
    return createClip(this.options.resolve(scene.transition.video));
  }

  private build(scene: Scene, needed: number): PreparedPicture {
    const { sources, width, height } = scene.image;
    const picture = document.createElement('picture');
    const avif = pickSource(sources.avif, needed);
    let avifSource: HTMLSourceElement | undefined;
    if (avif) {
      avifSource = document.createElement('source');
      avifSource.type = 'image/avif';
      avifSource.srcset = this.options.resolve(avif.src);
      picture.append(avifSource);
    }
    const webp = pickSource(sources.webp, needed);
    const webpSource = document.createElement('source');
    webpSource.type = 'image/webp';
    picture.append(webpSource);

    const img = document.createElement('img');
    img.alt = scene.alt;
    img.width = width;
    img.height = height;
    img.decoding = 'async';
    img.draggable = false;
    picture.append(img);
    if (webp) {
      webpSource.srcset = this.options.resolve(webp.src);
      // Set last, with the sources in place, so the browser picks a format once.
      img.src = this.options.resolve(webp.src);
    }

    const ready = waitForLoad(img)
      // decode() keeps the first paint from stalling on a large image; a decode failure after a
      // successful load still leaves a drawable image, so it is not fatal.
      .then(() => img.decode().catch(() => undefined))
      .then(
        () => {
          prepared.loaded = true;
          img.classList.add('is-loaded');
          return true;
        },
        () => false,
      );
    const prepared: PreparedPicture = {
      scene,
      picture,
      img,
      ready,
      loaded: false,
      width: webp?.width ?? 0,
      avifSource,
      webpSource,
    };
    return prepared;
  }

  /** Swaps in bigger files when the viewport grew; the old image stays until the new one loads. */
  private upgrade(prepared: PreparedPicture, needed: number): void {
    const { sources } = prepared.scene.image;
    const webp = pickSource(sources.webp, needed);
    if (!webp || webp.width <= prepared.width) {
      return;
    }
    const avif = pickSource(sources.avif, needed);
    if (avif && prepared.avifSource) {
      prepared.avifSource.srcset = this.options.resolve(avif.src);
    }
    prepared.webpSource.srcset = this.options.resolve(webp.src);
    prepared.img.src = this.options.resolve(webp.src);
    prepared.width = webp.width;
  }
}

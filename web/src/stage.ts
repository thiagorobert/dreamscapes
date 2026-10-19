/**
 * The stage: one full-bleed layer per scene, scaled to cover the viewport.
 *
 * Each layer keeps the image's aspect ratio and is sized so the image covers the viewport, which
 * means it overflows in one direction unless the viewport has exactly the image's ratio. The
 * overflow is positioned so the scene's focal point sits as close to the middle of the screen as
 * the image edges allow, and the visitor can drag or swipe to pan within those edges. The hotspot
 * overlay lives inside the layer, so it shares the image's coordinate system and never needs any
 * coordinate mapping.
 *
 * Layers rather than a single image: during a transition the outgoing and incoming scenes are on
 * screen together, each framed around its own focal point.
 */

import type { PreparedPicture } from './preload';
import type { Point, Scene } from './types';

export interface Size {
  width: number;
  height: number;
}

/** Top-left corner of a layer relative to the viewport, in CSS pixels (zero or negative). */
export interface Offset {
  x: number;
  y: number;
}

/** Pointer travel in CSS pixels before a press counts as a pan rather than a click. */
export const DRAG_THRESHOLD_PX = 8;

/**
 * Size of a box with aspect ratio `ratio` (width / height) that covers `viewport`:
 * as wide as the viewport when the viewport is at least as wide as the image, otherwise as tall.
 */
export function coverSize(viewport: Size, ratio: number): Size {
  const width = Math.max(1, viewport.width);
  const height = Math.max(1, viewport.height);
  if (width / height >= ratio) {
    return { width, height: width / ratio };
  }
  return { width: height * ratio, height };
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/** Keeps a layer of size `stage` covering `viewport`: no gap may open at any edge. */
export function clampOffset(viewport: Size, stage: Size, offset: Offset): Offset {
  return {
    x: clamp(offset.x, Math.min(0, viewport.width - stage.width), 0),
    y: clamp(offset.y, Math.min(0, viewport.height - stage.height), 0),
  };
}

/** Offset that centres `focal` (normalised image coordinates) in the viewport, within bounds. */
export function focalOffset(viewport: Size, stage: Size, focal: Point): Offset {
  return clampOffset(viewport, stage, {
    x: viewport.width / 2 - focal[0] * stage.width,
    y: viewport.height / 2 - focal[1] * stage.height,
  });
}

export function imageRatio(scene: Scene): number {
  return scene.image.width / scene.image.height;
}

export function translate(offset: Offset): string {
  return `translate3d(${String(offset.x)}px, ${String(offset.y)}px, 0)`;
}

export interface SceneLayer {
  readonly scene: Scene;
  /** Positioned and sized like the cover box; holds the art, the hotspot and the bloom. */
  readonly root: HTMLDivElement;
  /** The picture over its placeholder; transitions animate this element. */
  readonly art: HTMLDivElement;
  size: Size;
  offset: Offset;
}

interface Drag {
  pointerId: number;
  startX: number;
  startY: number;
  origin: Offset;
  panning: boolean;
}

export class Stage {
  /** The viewport-sized element that clips the layers and receives pointer input. */
  readonly element: HTMLDivElement;
  private readonly layers = new Set<SceneLayer>();
  private currentLayer: SceneLayer | undefined;
  private drag: Drag | undefined;
  /** Set when a press turned into a pan, so the click that ends it is not treated as a miss. */
  private swallowClick = false;
  private interactive = true;
  private readonly layoutListeners: (() => void)[] = [];

  constructor(parent: HTMLElement) {
    this.element = document.createElement('div');
    this.element.className = 'stage';
    parent.append(this.element);

    window.addEventListener('resize', () => {
      this.layout();
    });
    this.element.addEventListener('pointerdown', (event) => {
      this.onPointerDown(event);
    });
    this.element.addEventListener('pointermove', (event) => {
      this.onPointerMove(event);
    });
    this.element.addEventListener('pointerup', (event) => {
      this.onPointerEnd(event);
    });
    this.element.addEventListener('pointercancel', (event) => {
      this.onPointerEnd(event);
    });
    // Capture phase, so the click that ends a pan never reaches the hotspot or the miss handler.
    this.element.addEventListener(
      'click',
      (event) => {
        if (this.swallowClick) {
          this.swallowClick = false;
          event.stopPropagation();
          event.preventDefault();
        }
      },
      { capture: true },
    );
  }

  get current(): SceneLayer | undefined {
    return this.currentLayer;
  }

  viewport(): Size {
    return {
      width: this.element.clientWidth || window.innerWidth,
      height: this.element.clientHeight || window.innerHeight,
    };
  }

  /** Cover-box size for `scene` at the current viewport. */
  coverFor(scene: Scene): Size {
    return coverSize(this.viewport(), imageRatio(scene));
  }

  /** Initial framing for `scene` at the current viewport. */
  offsetFor(scene: Scene, size = this.coverFor(scene)): Offset {
    return focalOffset(this.viewport(), size, scene.focal);
  }

  /** Replaces whatever is on stage with `scene`, without any transition. */
  showScene(scene: Scene, picture: PreparedPicture): SceneLayer {
    const layer = this.mountBelow(scene, picture);
    this.promote(layer);
    return layer;
  }

  /** Adds `scene` underneath the current layer, for a transition to reveal. */
  mountBelow(scene: Scene, picture: PreparedPicture): SceneLayer {
    const root = document.createElement('div');
    root.className = 'scene-layer';
    root.dataset.scene = scene.id;
    const art = document.createElement('div');
    art.className = 'scene-art';
    art.style.backgroundImage = `url("${scene.image.lqip}")`;
    // A picture element can only be in one place; a scene that leads to itself would otherwise
    // pull its own image out of the outgoing layer mid-transition.
    art.append(picture.picture.isConnected ? picture.picture.cloneNode(true) : picture.picture);
    root.append(art);

    const size = this.coverFor(scene);
    const layer: SceneLayer = { scene, root, art, size, offset: this.offsetFor(scene, size) };
    this.place(layer);
    this.layers.add(layer);
    if (this.currentLayer) {
      this.element.insertBefore(root, this.currentLayer.root);
    } else {
      this.element.prepend(root);
    }
    return layer;
  }

  /** Makes `layer` the current scene and removes every other layer. */
  promote(layer: SceneLayer): void {
    for (const other of this.layers) {
      if (other !== layer) {
        other.root.remove();
        this.layers.delete(other);
      }
    }
    this.currentLayer?.root.classList.remove('is-current');
    this.currentLayer = layer;
    layer.root.classList.add('is-current');
  }

  /**
   * Puts `content` above every layer, framed like a layer of size `size` at `offset`.
   * Used for morph clips, which are drawn over the stage while it swaps scenes underneath.
   */
  mountOverlay(content: HTMLElement, size: Size, offset: Offset): HTMLDivElement {
    const overlay = document.createElement('div');
    overlay.className = 'morph';
    overlay.style.width = `${String(size.width)}px`;
    overlay.style.height = `${String(size.height)}px`;
    overlay.style.transform = translate(offset);
    overlay.append(content);
    this.element.append(overlay);
    return overlay;
  }

  /** Pointer panning and clicks are ignored while false (during transitions). */
  setInteractive(interactive: boolean): void {
    this.interactive = interactive;
    if (!interactive) {
      this.endDrag();
    }
  }

  onLayout(listener: () => void): void {
    this.layoutListeners.push(listener);
  }

  /** Refits every layer to the viewport and re-centres it on its focal point. */
  layout(): void {
    for (const layer of this.layers) {
      layer.size = this.coverFor(layer.scene);
      layer.offset = this.offsetFor(layer.scene, layer.size);
      this.place(layer);
    }
    for (const listener of this.layoutListeners) {
      listener();
    }
  }

  private place(layer: SceneLayer): void {
    layer.root.style.width = `${String(layer.size.width)}px`;
    layer.root.style.height = `${String(layer.size.height)}px`;
    layer.root.style.transform = translate(layer.offset);
  }

  private onPointerDown(event: PointerEvent): void {
    this.swallowClick = false;
    if (!this.interactive || !event.isPrimary || event.button !== 0 || !this.currentLayer) {
      return;
    }
    this.drag = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      origin: this.currentLayer.offset,
      panning: false,
    };
  }

  private onPointerMove(event: PointerEvent): void {
    const drag = this.drag;
    const layer = this.currentLayer;
    if (drag?.pointerId !== event.pointerId || !layer) {
      return;
    }
    const dx = event.clientX - drag.startX;
    const dy = event.clientY - drag.startY;
    if (!drag.panning) {
      if (Math.hypot(dx, dy) < DRAG_THRESHOLD_PX) {
        return;
      }
      drag.panning = true;
      // Capture only once the press is a pan: capturing on every press would retarget the
      // click away from the hotspot polygon.
      try {
        this.element.setPointerCapture(event.pointerId);
      } catch {
        // The pointer is already gone; the pan simply ends early.
      }
      this.element.classList.add('is-panning');
    }
    layer.offset = clampOffset(this.viewport(), layer.size, {
      x: drag.origin.x + dx,
      y: drag.origin.y + dy,
    });
    layer.root.style.transform = translate(layer.offset);
  }

  private onPointerEnd(event: PointerEvent): void {
    if (this.drag?.pointerId !== event.pointerId) {
      return;
    }
    if (this.drag.panning) {
      this.swallowClick = true;
    }
    this.endDrag();
  }

  private endDrag(): void {
    if (this.drag?.panning) {
      try {
        this.element.releasePointerCapture(this.drag.pointerId);
      } catch {
        // Already released.
      }
    }
    this.drag = undefined;
    this.element.classList.remove('is-panning');
  }
}

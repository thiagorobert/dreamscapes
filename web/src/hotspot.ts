/**
 * The clickable bridge object.
 *
 * An SVG whose viewBox is the unit square, stretched over the scene layer without preserving
 * aspect ratio, so the polygon's normalised image coordinates can be used as they are. The
 * polygon is invisible (transparent fill) but still hit-tested (`pointer-events="all"`), with a
 * transparent stroke of a fixed screen width around it so fingers do not have to be precise. The
 * browser does the point-in-polygon test, the cursor changes to a pointer over it, and nothing
 * else gives it away until a hint asks for a shimmer.
 *
 * Keyboard users do not reach the polygon itself: focusing it would draw a focus ring around the
 * secret. They get the visually hidden "Continue" button from `a11y.ts`, which triggers the same
 * activation, and Enter on the polygon still works for anything that focuses it programmatically.
 */

import type { SceneLayer } from './stage';
import type { BBox, Point } from './types';

const SVG_NS = 'http://www.w3.org/2000/svg';

/** How far outside the polygon, in screen pixels, a click or tap still counts as a hit. */
export const TOUCH_MARGIN_PX = 10;

/** Length of the shimmer sweep; matches the `hotspot-sweep` keyframes in styles.css. */
export const SHIMMER_MS = 1200;

/** How long the reduced-motion hint (a still highlight) stays up. */
export const STATIC_HIGHLIGHT_MS = 1000;

let renderCount = 0;

function svgElement<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attributes: Record<string, string>,
): SVGElementTagNameMap[K] {
  const element = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attributes)) {
    element.setAttribute(name, value);
  }
  return element;
}

export function pointsAttribute(polygon: readonly Point[]): string {
  return polygon.map(([x, y]) => `${String(x)},${String(y)}`).join(' ');
}

export class Hotspot {
  private svg: SVGSVGElement | undefined;
  private hintTimer: ReturnType<typeof setTimeout> | undefined;

  constructor(private readonly onActivate: () => void) {}

  /** Draws the hotspot for `layer`'s scene inside that layer, replacing any previous one. */
  render(layer: SceneLayer): void {
    this.clear();
    renderCount += 1;
    const id = `hotspot-${String(renderCount)}`;
    const { polygon, bbox } = layer.scene.bridge;
    const points = pointsAttribute(polygon);
    const [x0, y0, x1, y1]: BBox = bbox;

    const svg = svgElement('svg', {
      class: 'hotspot',
      viewBox: '0 0 1 1',
      preserveAspectRatio: 'none',
      'aria-hidden': 'true',
      focusable: 'false',
    });

    // The shimmer: a soft band swept across the bounding box, visible only inside the polygon.
    const defs = svgElement('defs', {});
    const clip = svgElement('clipPath', { id: `${id}-clip`, clipPathUnits: 'userSpaceOnUse' });
    clip.append(svgElement('polygon', { points }));
    const gradient = svgElement('linearGradient', {
      id: `${id}-sweep`,
      x1: '0',
      y1: '0',
      x2: '1',
      y2: '0.35',
    });
    for (const [offset, alpha] of [
      ['0', '0'],
      ['0.38', '0.12'],
      ['0.5', '0.55'],
      ['0.62', '0.12'],
      ['1', '0'],
    ] as const) {
      gradient.append(
        svgElement('stop', { offset, 'stop-color': '#fff8ec', 'stop-opacity': alpha }),
      );
    }
    defs.append(clip, gradient);

    const glow = svgElement('g', { class: 'hotspot-glow', 'clip-path': `url(#${id}-clip)` });
    glow.append(
      svgElement('rect', {
        class: 'hotspot-sweep',
        x: String(x0),
        y: String(y0),
        width: String(x1 - x0),
        height: String(y1 - y0),
        fill: `url(#${id}-sweep)`,
      }),
    );

    const shape = svgElement('polygon', {
      class: 'hotspot-shape',
      points,
      fill: 'transparent',
      stroke: 'transparent',
      // A stroke is centred on the outline, so it must be twice the margin wide.
      'stroke-width': String(2 * TOUCH_MARGIN_PX),
      'stroke-linejoin': 'round',
      // Measures the stroke in screen pixels despite the stretched unit viewBox.
      'vector-effect': 'non-scaling-stroke',
      'pointer-events': 'all',
    });
    shape.addEventListener('click', (event) => {
      // Handled here, so the stage's miss handler never sees a hit.
      event.stopPropagation();
      this.onActivate();
    });
    shape.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        this.onActivate();
      }
    });

    svg.append(defs, glow, shape);
    layer.root.append(svg);
    this.svg = svg;
  }

  clear(): void {
    clearTimeout(this.hintTimer);
    this.svg?.remove();
    this.svg = undefined;
  }

  /**
   * Plays the hint once: a luminance sweep across the object, or, with reduced motion, a still
   * highlight for a second.
   */
  shimmer(reducedMotion: boolean): void {
    const svg = this.svg;
    if (!svg) {
      return;
    }
    clearTimeout(this.hintTimer);
    svg.classList.remove('is-shimmering', 'is-highlighted');
    // Force a style flush so re-adding the class restarts the animation.
    svg.getBoundingClientRect();
    const className = reducedMotion ? 'is-highlighted' : 'is-shimmering';
    svg.classList.add(className);
    this.hintTimer = setTimeout(
      () => {
        svg.classList.remove(className);
      },
      reducedMotion ? STATIC_HIGHLIGHT_MS : SHIMMER_MS,
    );
  }

  /** A gentle pulse while the next scene is still loading after the object was clicked. */
  setPending(pending: boolean): void {
    this.svg?.classList.toggle('is-pending', pending);
  }
}

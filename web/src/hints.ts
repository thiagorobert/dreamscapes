/**
 * Hints that keep the hidden object "not obvious, but not frustrating".
 *
 * - A click that misses the object shows a soft ripple where it landed, and nothing else.
 * - After a stretch without finding the object, or after a few misses, the object shimmers once.
 *   The wait backs off after every shimmer so the hint never nags, and starts over on each scene.
 * - The first visit shows one line explaining the rule, remembered in localStorage.
 *
 * Every threshold is in `HINTS` below.
 */

import { readStorage, writeStorage } from './storage';

export const HINTS = {
  /** Idle waits before each shimmer on a scene; the last one repeats. */
  idleDelaysMs: [20_000, 40_000, 60_000],
  /** Misses on one scene that trigger a shimmer straight away. */
  missesBeforeHint: 3,
  /** Length of the miss ripple; matches the `ripple` keyframes in styles.css. */
  rippleMs: 700,
  /** Remembers that the first-visit overlay was dismissed. */
  introStorageKey: 'dreamscapes:intro-seen',
  introText: 'Something in each dream leads to the next. Find it.',
} as const;

export type HintReason = 'idle' | 'misses';

/**
 * When to shimmer, as a small state machine over timers.
 *
 * `start()` on every new scene, `miss()` on every miss, `stop()` while a transition runs.
 */
export class HintScheduler {
  private step = 0;
  private misses = 0;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private running = false;

  constructor(
    private readonly onHint: (reason: HintReason) => void,
    private readonly idleDelaysMs: readonly number[] = HINTS.idleDelaysMs,
    private readonly missesBeforeHint: number = HINTS.missesBeforeHint,
  ) {}

  /** A scene became visible: reset the back-off and the miss count, start the idle wait. */
  start(): void {
    this.stop();
    this.running = true;
    this.step = 0;
    this.misses = 0;
    this.arm();
  }

  stop(): void {
    clearTimeout(this.timer);
    this.timer = undefined;
    this.running = false;
  }

  /** A click missed the object. Misses do not reset the idle wait: only finding it does. */
  miss(): void {
    if (!this.running) {
      return;
    }
    this.misses += 1;
    if (this.misses >= this.missesBeforeHint) {
      this.fire('misses');
    }
  }

  /** The wait before the next idle shimmer, in milliseconds. */
  get currentDelayMs(): number {
    const delays = this.idleDelaysMs;
    return delays[Math.min(this.step, delays.length - 1)] ?? 0;
  }

  get isRunning(): boolean {
    return this.running;
  }

  private arm(): void {
    clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.fire('idle');
    }, this.currentDelayMs);
  }

  private fire(reason: HintReason): void {
    this.misses = 0;
    this.step += 1;
    this.onHint(reason);
    // Every shimmer restarts the idle wait, one step further along the back-off.
    if (this.running) {
      this.arm();
    }
  }
}

/** A soft ring at the viewport point (x, y), removed once it has played. */
export function showRipple(parent: HTMLElement, x: number, y: number): void {
  const ripple = document.createElement('div');
  ripple.className = 'ripple';
  ripple.style.left = `${String(x)}px`;
  ripple.style.top = `${String(y)}px`;
  ripple.setAttribute('aria-hidden', 'true');
  parent.append(ripple);
  setTimeout(() => {
    ripple.remove();
  }, HINTS.rippleMs);
}

/**
 * Shows the one-line rule on a first visit. `onDone` runs once the overlay is gone, or straight
 * away when it was already dismissed on an earlier visit.
 */
export function showIntro(parent: HTMLElement, onDone: () => void): void {
  if (readStorage('local', HINTS.introStorageKey) !== null) {
    onDone();
    return;
  }
  const overlay = document.createElement('div');
  overlay.className = 'intro';
  overlay.setAttribute('role', 'status');
  const text = document.createElement('p');
  text.textContent = HINTS.introText;
  overlay.append(text);
  parent.append(overlay);

  let dismissed = false;
  const dismiss = (): void => {
    if (dismissed) {
      return;
    }
    dismissed = true;
    writeStorage('local', HINTS.introStorageKey, '1');
    window.removeEventListener('keydown', dismiss);
    overlay.classList.add('is-leaving');
    setTimeout(() => {
      overlay.remove();
    }, 400);
    onDone();
  };
  overlay.addEventListener('click', (event) => {
    // The click only dismisses; it must not count as a miss or reach the object underneath.
    event.stopPropagation();
    dismiss();
  });
  // Any key dismisses too, without swallowing it: Tab still moves on to the continue button.
  window.addEventListener('keydown', dismiss);
}

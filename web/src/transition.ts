/**
 * Moving from one scene to the next.
 *
 * Three ways, chosen per activation:
 * - zoom: the outgoing scene falls into its bridge object (scale and blur towards the hotspot
 *   centroid while it dissolves), the incoming scene settles in underneath, and a brief bloom of
 *   light at the hotspot sells the "dissolving into light" feel;
 * - video: when the edge has a pre-rendered morph clip, the clip plays over the stage while the
 *   stage swaps scenes underneath, with short crossfades in and out. If the clip fails, or is not
 *   playable and started within a second, the zoom runs instead; a clip that stalls mid-play is
 *   cut short, revealing the new scene already waiting underneath;
 * - fade: with reduced motion, a plain crossfade with no scaling or blur.
 *
 * Every animation uses the Web Animations API, so its end is a promise rather than a guess.
 */

import { releaseClip, type PreparedPicture } from './preload';
import { translate, type Stage, type SceneLayer } from './stage';
import type { Scene } from './types';

export const TIMING = {
  zoomOutMs: 900,
  zoomOutScale: 1.6,
  zoomOutBlurPx: 10,
  zoomOutEasing: 'cubic-bezier(.7, 0, .84, 0)',
  zoomInMs: 900,
  zoomInDelayMs: 150,
  zoomInScale: 1.12,
  zoomInBlurPx: 6,
  zoomInEasing: 'cubic-bezier(.16, 1, .3, 1)',
  bloomMs: 500,
  bloomPeak: 0.25,
  fadeMs: 300,
  clipFadeMs: 120,
  /** A clip that is not playable this soon after the click is skipped for the zoom. */
  clipReadyTimeoutMs: 1000,
  /** A clip whose play() has not started it this soon is skipped for the zoom. */
  clipStartTimeoutMs: 1000,
  /** A clip that stops for data (waiting or stalled) this long is cut short. */
  clipStallGraceMs: 400,
  /** Grace period past a clip's length before giving up on its `ended` event. */
  clipEndSlackMs: 1500,
} as const;

export type TransitionKind = 'zoom' | 'fade' | 'video' | 'zoom-fallback';

export interface TransitionRequest {
  stage: Stage;
  from: SceneLayer;
  to: Scene;
  picture: PreparedPicture;
  /** The morph clip for this edge, or null when it has none. */
  clip: HTMLVideoElement | null;
  reducedMotion: boolean;
}

export interface TransitionResult {
  /** The incoming scene's layer, now current. */
  layer: SceneLayer;
  kind: TransitionKind;
}

function percent(value: number): string {
  return `${String(value * 100)}%`;
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Waits for `animations` to finish. Capped at `limitMs` so a paused document timeline (a tab in
 * the background) can never leave the site stuck mid-transition.
 */
async function settle(animations: readonly Animation[], limitMs: number): Promise<void> {
  await Promise.race([
    Promise.all(animations.map((animation) => animation.finished.catch(() => undefined))),
    delay(limitMs),
  ]);
}

function fade(element: HTMLElement, from: number, to: number, ms: number): Promise<void> {
  const animation = element.animate([{ opacity: from }, { opacity: to }], {
    duration: ms,
    easing: 'ease-in-out',
    fill: 'forwards',
  });
  return settle([animation], ms + 400);
}

/** The zoom-dissolve: "fall into the object". */
export async function zoom(
  stage: Stage,
  from: SceneLayer,
  to: Scene,
  picture: PreparedPicture,
): Promise<SceneLayer> {
  const incoming = stage.mountBelow(to, picture);
  const [cx, cy] = from.scene.bridge.centroid;
  from.art.style.transformOrigin = `${percent(cx)} ${percent(cy)}`;
  incoming.art.style.transformOrigin = `${percent(to.focal[0])} ${percent(to.focal[1])}`;

  const bloom = document.createElement('div');
  bloom.className = 'bloom';
  bloom.style.left = percent(cx);
  bloom.style.top = percent(cy);
  from.root.append(bloom);

  const t = TIMING;
  const animations = [
    from.art.animate(
      [
        { transform: 'scale(1)', filter: 'blur(0px)', opacity: 1 },
        {
          transform: `scale(${String(t.zoomOutScale)})`,
          filter: `blur(${String(t.zoomOutBlurPx)}px)`,
          opacity: 0,
        },
      ],
      { duration: t.zoomOutMs, easing: t.zoomOutEasing, fill: 'forwards' },
    ),
    incoming.art.animate(
      [
        {
          transform: `scale(${String(t.zoomInScale)})`,
          filter: `blur(${String(t.zoomInBlurPx)}px)`,
          opacity: 0,
        },
        { transform: 'scale(1)', filter: 'blur(0px)', opacity: 1 },
      ],
      { duration: t.zoomInMs, delay: t.zoomInDelayMs, easing: t.zoomInEasing, fill: 'backwards' },
    ),
    bloom.animate([{ opacity: 0 }, { opacity: t.bloomPeak }, { opacity: 0 }], {
      duration: t.bloomMs,
      easing: 'ease-in-out',
    }),
  ];
  await settle(animations, t.zoomInDelayMs + t.zoomInMs + 600);
  stage.promote(incoming);
  incoming.art.style.transformOrigin = '';
  return incoming;
}

/** A plain crossfade, used for reduced motion and for the back button. */
export async function crossfade(
  stage: Stage,
  from: SceneLayer,
  to: Scene,
  picture: PreparedPicture,
  ms: number = TIMING.fadeMs,
): Promise<SceneLayer> {
  const incoming = stage.mountBelow(to, picture);
  await fade(from.art, 1, 0, ms);
  stage.promote(incoming);
  return incoming;
}

/**
 * Resolves true once `video` can play without stalling, false if it errors or `timeoutMs`
 * passes first.
 */
export function whenPlayable(video: HTMLVideoElement, timeoutMs: number): Promise<boolean> {
  if (video.error) {
    return Promise.resolve(false);
  }
  if (video.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA) {
    return Promise.resolve(true);
  }
  return new Promise((resolve) => {
    const finish = (playable: boolean): void => {
      clearTimeout(timer);
      video.removeEventListener('canplay', onReady);
      video.removeEventListener('error', onError);
      resolve(playable);
    };
    const onReady = (): void => {
      finish(true);
    };
    const onError = (): void => {
      finish(false);
    };
    const timer = setTimeout(onError, timeoutMs);
    video.addEventListener('canplay', onReady);
    video.addEventListener('error', onError);
  });
}

/**
 * Resolves when `video` ends or errors; when it has been stuck waiting for data for
 * `stallGraceMs` (a `waiting` or `stalled` event with no `playing` or progress after it); or
 * after `limitMs` whatever happens. The incoming scene is already on stage under the clip, so
 * cutting a stuck clip short just reveals it.
 */
function whenFinished(
  video: HTMLVideoElement,
  limitMs: number,
  stallGraceMs: number,
): Promise<void> {
  return new Promise((resolve) => {
    let stallTimer: ReturnType<typeof setTimeout> | undefined;
    let stalledAt = 0;
    const recover = (): void => {
      clearTimeout(stallTimer);
      stallTimer = undefined;
    };
    const onStall = (): void => {
      if (stallTimer === undefined) {
        stalledAt = video.currentTime;
        stallTimer = setTimeout(finish, stallGraceMs);
      }
    };
    // `stalled` can also fire while playback carries on from the buffer, so moving time counts
    // as recovery as well as an explicit `playing`.
    const onTimeUpdate = (): void => {
      if (video.currentTime > stalledAt) {
        recover();
      }
    };
    const listeners: [string, () => void][] = [
      ['ended', finish],
      ['error', finish],
      ['waiting', onStall],
      ['stalled', onStall],
      ['playing', recover],
      ['timeupdate', onTimeUpdate],
    ];
    function finish(): void {
      clearTimeout(limitTimer);
      recover();
      for (const [type, listener] of listeners) {
        video.removeEventListener(type, listener);
      }
      resolve();
    }
    const limitTimer = setTimeout(finish, limitMs);
    for (const [type, listener] of listeners) {
      video.addEventListener(type, listener);
    }
  });
}

/** Starts `video`; false if it refuses or has not started within `timeoutMs`. */
async function startClip(video: HTMLVideoElement, timeoutMs: number): Promise<boolean> {
  try {
    video.currentTime = 0;
  } catch {
    // Not seekable yet; it starts from wherever it is, which is the beginning.
  }
  const started = await Promise.race([
    video.play().then(
      () => true,
      () => false,
    ),
    delay(timeoutMs).then(() => false),
  ]);
  if (!started) {
    // Stops a play() that is still pending, so it cannot start behind the zoom.
    video.pause();
  }
  return started;
}

function clipSeconds(video: HTMLVideoElement, scene: Scene): number {
  if (Number.isFinite(video.duration) && video.duration > 0) {
    return video.duration;
  }
  return scene.transition.type === 'video' ? scene.transition.duration : 0;
}

/**
 * Plays the morph clip over the stage. Returns the incoming layer, or null when the clip could
 * not start, in which case the stage is untouched and the caller falls back to the zoom.
 */
async function playClip(
  request: TransitionRequest,
  clip: HTMLVideoElement,
): Promise<SceneLayer | null> {
  const { stage, from, to, picture } = request;
  if (!(await whenPlayable(clip, TIMING.clipReadyTimeoutMs))) {
    return null;
  }
  const overlay = stage.mountOverlay(clip, from.size, from.offset);
  overlay.style.opacity = '0';
  if (!(await startClip(clip, TIMING.clipStartTimeoutMs))) {
    overlay.remove();
    return null;
  }
  // The clip's first frame may differ slightly from the still, so it fades in over it.
  await fade(overlay, 0, 1, TIMING.clipFadeMs);

  // Hidden by the clip, the stage now shows the incoming scene underneath.
  const incoming = stage.mountBelow(to, picture);
  stage.promote(incoming);

  // On narrow screens each scene is framed around its own focal point; drift the clip from the
  // outgoing framing to the incoming one so its last frame lands exactly over the new scene.
  const remainingMs = Math.max(0, (clipSeconds(clip, from.scene) - clip.currentTime) * 1000);
  overlay.animate(
    [{ transform: translate(from.offset) }, { transform: translate(incoming.offset) }],
    {
      duration: Math.max(1, remainingMs),
      easing: 'ease-in-out',
      fill: 'forwards',
    },
  );
  await whenFinished(clip, remainingMs + TIMING.clipEndSlackMs, TIMING.clipStallGraceMs);
  await fade(overlay, 1, 0, TIMING.clipFadeMs);
  overlay.remove();
  return incoming;
}

/** Runs the transition this edge calls for and returns the new current layer. */
export async function playTransition(request: TransitionRequest): Promise<TransitionResult> {
  const { stage, from, to, picture, clip, reducedMotion } = request;
  try {
    if (reducedMotion) {
      return { layer: await crossfade(stage, from, to, picture), kind: 'fade' };
    }
    if (clip) {
      const layer = await playClip(request, clip);
      if (layer) {
        return { layer, kind: 'video' };
      }
      return { layer: await zoom(stage, from, to, picture), kind: 'zoom-fallback' };
    }
    return { layer: await zoom(stage, from, to, picture), kind: 'zoom' };
  } finally {
    if (clip) {
      releaseClip(clip);
    }
  }
}

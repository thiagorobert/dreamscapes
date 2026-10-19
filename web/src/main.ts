import './styles.css';

import { Accessibility } from './a11y';
import { Hotspot } from './hotspot';
import { HintScheduler, showIntro, showRipple } from './hints';
import { assetUrl, loadManifest, nextScene, type Dreams } from './manifest';
import { Preloader, type PreparedPicture } from './preload';
import { Router, type Route } from './router';
import { Stage, type SceneLayer } from './stage';
import { readStorage, writeStorage } from './storage';
import { crossfade, playTransition, type TransitionKind } from './transition';
import type { Scene } from './types';

/** Keeps the depth counter across reloads of the same tab. */
const DEPTH_KEY = 'dreamscapes:depth';

/** How long a click on the object waits for the next image before going ahead on its placeholder. */
const IMAGE_WAIT_MS = 8000;

/** Shorter wait for the back button: going back should feel immediate. */
const BACK_IMAGE_WAIT_MS = 1500;

function storedDepth(): number | null {
  const depth = Number(readStorage('session', DEPTH_KEY));
  return Number.isInteger(depth) && depth >= 1 ? depth : null;
}

/** True once `ready` resolves, false if it is still pending after `ms`. */
function within(ready: Promise<boolean>, ms: number): Promise<boolean> {
  return Promise.race([
    ready,
    new Promise<boolean>((resolve) => {
      setTimeout(() => {
        resolve(false);
      }, ms);
    }),
  ]);
}

function element<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) {
    node.textContent = text;
  }
  return node;
}

class Dreamscapes {
  private readonly stage: Stage;
  private readonly hotspot: Hotspot;
  private readonly hints: HintScheduler;
  private readonly a11y: Accessibility;
  private readonly router: Router;
  private readonly preloader: Preloader;
  private readonly counter: HTMLParagraphElement;
  private readonly motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  private scene: Scene;
  private depth = 1;
  private busy = false;
  /** Set when the visitor used back/forward mid-transition; honoured once it ends. */
  private routeChangedWhileBusy = false;
  private introDone = false;

  constructor(
    private readonly root: HTMLElement,
    private readonly dreams: Dreams,
  ) {
    this.scene = dreams.start;
    root.dataset.busy = 'false';
    root.replaceChildren();
    this.stage = new Stage(root);
    root.append(element('h1', 'chrome wordmark', 'dreamscapes'));
    this.counter = element('p', 'chrome depth');
    root.append(this.counter);
    const clipHost = element('div', 'offscreen');
    clipHost.setAttribute('aria-hidden', 'true');
    root.append(clipHost);

    this.a11y = new Accessibility(root, () => {
      void this.advance();
    });
    this.hotspot = new Hotspot(() => {
      void this.advance();
    });
    this.hints = new HintScheduler(() => {
      this.hotspot.shimmer(this.reducedMotion);
    });
    this.router = new Router((id) => dreams.scenes.has(id), dreams.start.id);
    this.preloader = new Preloader({
      cssWidthFor: (scene) => this.stage.coverFor(scene).width,
      resolve: assetUrl,
      clipHost,
    });

    this.stage.element.addEventListener('click', (event) => {
      // Clicks on the object stop at the hotspot, and the click that ends a pan is swallowed by
      // the stage, so whatever arrives here missed.
      if (this.busy) {
        return;
      }
      showRipple(this.root, event.clientX, event.clientY);
      this.hints.miss();
    });
    this.stage.onLayout(() => {
      // A larger viewport may need a larger file for the scene on screen.
      this.preloader.picture(this.scene);
    });
    this.router.listen((route) => {
      this.onRoute(route);
    });
  }

  private get reducedMotion(): boolean {
    return this.motion.matches;
  }

  start(): void {
    const route = this.router.current();
    this.depth = route.state?.depth ?? storedDepth() ?? 1;
    this.scene = this.dreams.scenes.get(route.id) ?? this.dreams.start;
    // Name the scene in the URL even when the visitor landed on a bare or unknown address.
    this.router.replace(this.scene.id, { depth: this.depth });
    const picture = this.preloader.picture(this.scene);
    this.settle(this.stage.showScene(this.scene, picture), picture);
    showIntro(this.root, () => {
      this.introDone = true;
      if (!this.busy) {
        this.hints.start();
      }
    });
  }

  /** The visitor found the object (or pressed Continue): go one dream deeper. */
  private async advance(): Promise<void> {
    const from = this.stage.current;
    if (this.busy || !from) {
      return;
    }
    this.setBusy(true);
    const to = nextScene(this.dreams, from.scene);
    const picture = this.preloader.picture(to);
    if (!picture.loaded) {
      // Rare: the click beat the preload. Pulse the object until the next image is ready.
      this.hotspot.setPending(true);
      await within(picture.ready, IMAGE_WAIT_MS);
      this.hotspot.setPending(false);
    }
    this.hotspot.clear();
    const reducedMotion = this.reducedMotion;
    const { layer, kind } = await playTransition({
      stage: this.stage,
      from,
      to,
      picture,
      clip: reducedMotion ? null : this.preloader.takeClip(from.scene),
      reducedMotion,
    });
    this.depth += 1;
    this.finish(layer, picture, kind);
  }

  /** Back, forward, or an edited hash. */
  private onRoute(route: Route): void {
    if (this.busy) {
      this.routeChangedWhileBusy = true;
      return;
    }
    if (route.state) {
      this.depth = route.state.depth;
    }
    const scene = this.dreams.scenes.get(route.id) ?? this.dreams.start;
    if (!route.matched || !route.state) {
      // An address typed or pasted by hand: give the entry a proper hash and depth.
      this.router.replace(scene.id, { depth: this.depth });
    }
    if (scene.id === this.scene.id) {
      this.showDepth();
      return;
    }
    void this.jumpTo(scene);
  }

  private async jumpTo(scene: Scene): Promise<void> {
    const from = this.stage.current;
    if (!from) {
      return;
    }
    this.setBusy(true);
    this.hotspot.clear();
    const picture = this.preloader.picture(scene);
    await within(picture.ready, BACK_IMAGE_WAIT_MS);
    const layer = await crossfade(this.stage, from, scene, picture);
    this.finish(layer, picture, 'fade', false);
  }

  private finish(
    layer: SceneLayer,
    picture: PreparedPicture,
    kind: TransitionKind,
    pushHistory = true,
  ): void {
    this.root.dataset.transition = kind;
    this.settle(layer, picture);
    this.setBusy(false);
    if (this.routeChangedWhileBusy) {
      // The visitor went back or forward mid-transition: follow the address bar instead of
      // stacking a new entry on top of where they asked to be.
      this.routeChangedWhileBusy = false;
      this.onRoute(this.router.current());
      return;
    }
    if (pushHistory) {
      this.router.push(layer.scene.id, { depth: this.depth });
    }
  }

  /** Everything that follows a scene becoming current, however it got there. */
  private settle(layer: SceneLayer, picture: PreparedPicture): void {
    this.scene = layer.scene;
    this.root.dataset.scene = layer.scene.id;
    document.title = `${layer.scene.title} · Dreamscapes`;
    this.hotspot.render(layer);
    this.a11y.announce(layer.scene);
    this.showDepth();
    if (this.introDone) {
      this.hints.start();
    }
    // Preload in priority order once the scene on screen has its image: the next scene, the one
    // after, then the morph clip for the edge the visitor is about to take.
    const next = nextScene(this.dreams, layer.scene);
    const afterNext = nextScene(this.dreams, next);
    void picture.ready
      .then(() => this.preloader.warm([next, afterNext]))
      .then(() => {
        if (this.scene === layer.scene && !this.reducedMotion) {
          this.preloader.preloadClip(layer.scene);
        }
      });
  }

  private showDepth(): void {
    writeStorage('session', DEPTH_KEY, String(this.depth));
    const text = `dream ${String(this.depth)}`;
    if (this.counter.textContent !== text) {
      this.counter.textContent = text;
      // Brighten again briefly, then fade back (the animation lives in styles.css).
      this.counter.classList.remove('is-fresh');
      this.counter.getBoundingClientRect();
      this.counter.classList.add('is-fresh');
    }
  }

  private setBusy(busy: boolean): void {
    this.busy = busy;
    this.root.dataset.busy = String(busy);
    this.root.classList.toggle('is-transitioning', busy);
    this.stage.setInteractive(!busy);
    this.a11y.setBusy(busy);
    if (busy) {
      this.hints.stop();
    }
  }
}

function showError(root: HTMLElement): void {
  root.replaceChildren(
    element('h1', 'chrome wordmark', 'dreamscapes'),
    element('p', 'notice', 'The dreams could not be loaded. Try reloading the page.'),
  );
}

const root = document.querySelector<HTMLElement>('#app');
if (root) {
  loadManifest()
    .then((dreams) => {
      new Dreamscapes(root, dreams).start();
    })
    .catch((error: unknown) => {
      console.error(error);
      showError(root);
    });
}

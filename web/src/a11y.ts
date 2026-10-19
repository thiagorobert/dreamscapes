/**
 * Keyboard and screen-reader access.
 *
 * The hotspot is a secret, so it is not in the tab order. Instead a visually hidden
 * "Continue to the next dream" button is: Tab reaches it, Enter activates it, and it shows
 * itself while focused so sighted keyboard users can see where they are. It is never disabled
 * (a disabled button drops focus mid-walk); during a transition it is marked aria-disabled and
 * its clicks are ignored. A polite live region announces each new scene's title, and every image
 * carries the scene's alt text (set in preload.ts).
 */

import type { Scene } from './types';

export const CONTINUE_LABEL = 'Continue to the next dream';

export class Accessibility {
  readonly button: HTMLButtonElement;
  private readonly live: HTMLParagraphElement;
  private busy = false;

  constructor(parent: HTMLElement, onContinue: () => void) {
    this.button = document.createElement('button');
    this.button.type = 'button';
    this.button.className = 'continue';
    this.button.textContent = CONTINUE_LABEL;
    this.button.addEventListener('click', () => {
      if (!this.busy) {
        onContinue();
      }
    });

    this.live = document.createElement('p');
    this.live.className = 'visually-hidden';
    this.live.setAttribute('aria-live', 'polite');
    this.live.setAttribute('aria-atomic', 'true');
    this.live.dataset.testid = 'announcer';

    parent.append(this.button, this.live);
  }

  /** Announces the scene now on screen. */
  announce(scene: Scene): void {
    this.live.textContent = scene.title;
  }

  setBusy(busy: boolean): void {
    this.busy = busy;
    this.button.setAttribute('aria-disabled', String(busy));
  }
}

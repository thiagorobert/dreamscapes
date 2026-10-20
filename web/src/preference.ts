/**
 * The visitor's choice of transition: the CSS dissolve or the pre-rendered morph clips.
 *
 * Resolved once on load, in this order:
 * 1. the URL query parameter `?morph=1` or `?morph=0` (before the hash, e.g.
 *    `/dreamscapes/?morph=1#/prison-yard`), which is also saved as the visitor's choice;
 * 2. the choice saved in localStorage by an earlier visit or the toggle;
 * 3. `DEFAULT_TRANSITION`.
 *
 * The toggle in the bottom-left corner changes and saves it. When the URL carries the parameter,
 * the toggle rewrites it too, so a reload never undoes what the visitor just chose.
 */

import { readStorage, writeStorage } from './storage';

export type TransitionPreference = 'dissolve' | 'morph';

/** What a visitor gets until they choose. Change to 'morph' to make the clips the default. */
export const DEFAULT_TRANSITION: TransitionPreference = 'dissolve';

export const PREFERENCE_STORAGE_KEY = 'dreamscapes:transition';
export const PREFERENCE_QUERY_PARAMETER = 'morph';

/** Somewhere to keep the choice; storage.ts's never-throwing helpers in the site. */
export interface PreferenceStore {
  read: () => string | null;
  write: (value: string) => void;
}

export const localPreferenceStore: PreferenceStore = {
  read: () => readStorage('local', PREFERENCE_STORAGE_KEY),
  write: (value) => {
    writeStorage('local', PREFERENCE_STORAGE_KEY, value);
  },
};

const ON = new Set(['1', 'true', 'on']);
const OFF = new Set(['0', 'false', 'off']);

/** The choice a query string makes, or null when it makes none. */
export function preferenceFromQuery(search: string): TransitionPreference | null {
  const value = new URLSearchParams(search).get(PREFERENCE_QUERY_PARAMETER)?.trim().toLowerCase();
  if (value === undefined) {
    return null;
  }
  if (ON.has(value)) {
    return 'morph';
  }
  return OFF.has(value) ? 'dissolve' : null;
}

/** A saved choice, or null when nothing (or something unrecognisable) was saved. */
export function storedPreference(value: string | null): TransitionPreference | null {
  return value === 'dissolve' || value === 'morph' ? value : null;
}

/** Applies the resolution order above; a choice made in the URL is saved to `store`. */
export function resolvePreference(search: string, store: PreferenceStore): TransitionPreference {
  const fromQuery = preferenceFromQuery(search);
  if (fromQuery !== null) {
    store.write(fromQuery);
    return fromQuery;
  }
  return storedPreference(store.read()) ?? DEFAULT_TRANSITION;
}

/**
 * `href` with its `morph` parameter set to `preference`, or null when it has no such parameter
 * (then the address is left alone).
 */
export function hrefWithPreference(href: string, preference: TransitionPreference): string | null {
  const url = new URL(href);
  if (!url.searchParams.has(PREFERENCE_QUERY_PARAMETER)) {
    return null;
  }
  url.searchParams.set(PREFERENCE_QUERY_PARAMETER, preference === 'morph' ? '1' : '0');
  return url.href;
}

/**
 * The faint "morph off" / "morph on" switch in the bottom-left corner. A real toggle button, so
 * it is in the tab order (after the hidden Continue button) and Enter or Space flips it.
 */
export class MorphToggle {
  readonly button: HTMLButtonElement;

  constructor(
    parent: HTMLElement,
    private preference: TransitionPreference,
    private readonly onChange: (preference: TransitionPreference) => void,
  ) {
    this.button = document.createElement('button');
    this.button.type = 'button';
    this.button.className = 'chrome morph-toggle';
    this.button.title = 'Play the video morphs between dreams';
    this.button.addEventListener('click', () => {
      this.preference = this.preference === 'morph' ? 'dissolve' : 'morph';
      this.render();
      this.onChange(this.preference);
    });
    this.render();
    parent.append(this.button);
  }

  private render(): void {
    const on = this.preference === 'morph';
    this.button.textContent = on ? 'morph on' : 'morph off';
    this.button.setAttribute('aria-pressed', String(on));
  }
}

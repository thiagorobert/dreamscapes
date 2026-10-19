/**
 * Hash routes by scene slug: `#/prison-yard`.
 *
 * Hash routes work on any static host without rewrite rules, and every dream gets a shareable
 * URL. Moving forward pushes a history entry, so the browser's back button returns to the previous
 * dream. Each entry also remembers the depth counter at that point, so going back and forward
 * shows the count that belonged to that dream. An empty or unknown hash shows the start scene.
 */

export const HASH_PREFIX = '#/';

export interface RouteState {
  /** The depth counter when this history entry was created. */
  depth: number;
}

export interface Route {
  id: string;
  /** False when the hash was empty or named no known scene, so `id` is the start scene. */
  matched: boolean;
  state: RouteState | null;
}

/** The slug a hash names, or null for an empty or malformed hash. */
export function parseHash(hash: string): string | null {
  let slug = hash.startsWith('#') ? hash.slice(1) : hash;
  slug = slug.replace(/^\/+/, '').replace(/\/+$/, '');
  try {
    slug = decodeURIComponent(slug);
  } catch {
    return null;
  }
  slug = slug.trim().toLowerCase();
  return slug === '' ? null : slug;
}

export function hashFor(id: string): string {
  return `${HASH_PREFIX}${id}`;
}

/** Resolves a hash to a known scene id, falling back to `start`. */
export function resolveHash(
  hash: string,
  isKnown: (id: string) => boolean,
  start: string,
): Pick<Route, 'id' | 'matched'> {
  const slug = parseHash(hash);
  if (slug !== null && isKnown(slug)) {
    return { id: slug, matched: true };
  }
  return { id: start, matched: false };
}

/** The depth stored in a history entry, or null when the entry is not one of ours. */
export function readRouteState(state: unknown): RouteState | null {
  if (typeof state !== 'object' || state === null || !('depth' in state)) {
    return null;
  }
  const depth = state.depth;
  return typeof depth === 'number' && Number.isInteger(depth) && depth >= 1 ? { depth } : null;
}

export class Router {
  constructor(
    private readonly isKnown: (id: string) => boolean,
    private readonly start: string,
  ) {}

  current(): Route {
    return {
      ...resolveHash(window.location.hash, this.isKnown, this.start),
      state: readRouteState(window.history.state),
    };
  }

  /** A new dream: adds a history entry without firing popstate. */
  push(id: string, state: RouteState): void {
    window.history.pushState(state, '', hashFor(id));
  }

  /** Rewrites the current entry, e.g. to make the landing URL name its scene. */
  replace(id: string, state: RouteState): void {
    window.history.replaceState(state, '', hashFor(id));
  }

  /**
   * Calls `listener` when the visitor moves through history (back, forward) or edits the hash.
   * Browsers fire popstate for both.
   */
  listen(listener: (route: Route) => void): void {
    window.addEventListener('popstate', () => {
      listener(this.current());
    });
  }
}

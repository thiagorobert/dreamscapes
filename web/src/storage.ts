/**
 * Web Storage that never throws.
 *
 * `localStorage` and `sessionStorage` can be missing, full, or throw on access (private windows,
 * blocked site data, sandboxed previews). Nothing the site stores is essential, so every read
 * and write goes through these helpers and a failure just means the value is not remembered.
 */

export type StorageKind = 'local' | 'session';

function storageArea(kind: StorageKind): Storage {
  // Merely reading the property can throw when site data is blocked, so it stays inside the
  // callers' try blocks.
  return kind === 'local' ? window.localStorage : window.sessionStorage;
}

export function readStorage(kind: StorageKind, key: string): string | null {
  try {
    return storageArea(kind).getItem(key);
  } catch {
    return null;
  }
}

export function writeStorage(kind: StorageKind, key: string, value: string): void {
  try {
    storageArea(kind).setItem(key, value);
  } catch {
    // Not remembered; the experience works the same without it.
  }
}

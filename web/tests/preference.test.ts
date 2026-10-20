import { describe, expect, test } from 'vitest';

import {
  DEFAULT_TRANSITION,
  hrefWithPreference,
  localPreferenceStore,
  preferenceFromQuery,
  resolvePreference,
  storedPreference,
  type PreferenceStore,
} from '../src/preference';

/** An in-memory store that records what was written. */
function memoryStore(initial: string | null = null): PreferenceStore & { writes: string[] } {
  let value = initial;
  const writes: string[] = [];
  return {
    writes,
    read: () => value,
    write: (next) => {
      value = next;
      writes.push(next);
    },
  };
}

describe('preferenceFromQuery', () => {
  test.each([
    ['?morph=1', 'morph'],
    ['?morph=0', 'dissolve'],
    ['?morph=on', 'morph'],
    ['?morph=off', 'dissolve'],
    ['?morph=TRUE', 'morph'],
    ['?other=x&morph=0', 'dissolve'],
  ])('%s chooses %s', (search, preference) => {
    expect(preferenceFromQuery(search)).toBe(preference);
  });

  test.each(['', '?', '?morph', '?morph=maybe', '?other=1'])('%j chooses nothing', (search) => {
    expect(preferenceFromQuery(search)).toBeNull();
  });
});

describe('storedPreference', () => {
  test('accepts the two saved values and nothing else', () => {
    expect(storedPreference('morph')).toBe('morph');
    expect(storedPreference('dissolve')).toBe('dissolve');
    expect(storedPreference('zoom')).toBeNull();
    expect(storedPreference(null)).toBeNull();
  });
});

describe('resolvePreference', () => {
  test('defaults to the dissolve', () => {
    expect(DEFAULT_TRANSITION).toBe('dissolve');
    const store = memoryStore();
    expect(resolvePreference('', store)).toBe('dissolve');
    expect(store.writes).toEqual([]);
  });

  test('a saved choice beats the default', () => {
    expect(resolvePreference('', memoryStore('morph'))).toBe('morph');
  });

  test('an unrecognisable saved value falls back to the default', () => {
    expect(resolvePreference('', memoryStore('sparkles'))).toBe(DEFAULT_TRANSITION);
  });

  test('the URL beats a saved choice, and is saved in its place', () => {
    const on = memoryStore('dissolve');
    expect(resolvePreference('?morph=1', on)).toBe('morph');
    expect(on.writes).toEqual(['morph']);
    expect(resolvePreference('', on)).toBe('morph');

    const off = memoryStore('morph');
    expect(resolvePreference('?morph=0', off)).toBe('dissolve');
    expect(off.writes).toEqual(['dissolve']);
    expect(resolvePreference('', off)).toBe('dissolve');
  });

  test('an unrecognisable URL value is ignored, and nothing is saved', () => {
    const store = memoryStore('morph');
    expect(resolvePreference('?morph=maybe', store)).toBe('morph');
    expect(store.writes).toEqual([]);
  });

  test('works where storage is unavailable', () => {
    // Vitest runs in Node, where there is no localStorage at all: the storage helpers swallow
    // the error, reads come back empty, and the default applies.
    expect(resolvePreference('', localPreferenceStore)).toBe(DEFAULT_TRANSITION);
    expect(resolvePreference('?morph=1', localPreferenceStore)).toBe('morph');
  });
});

describe('hrefWithPreference', () => {
  test('rewrites a morph parameter already in the address, keeping the rest', () => {
    expect(
      hrefWithPreference('https://example.test/dreamscapes/?morph=1#/prison-yard', 'dissolve'),
    ).toBe('https://example.test/dreamscapes/?morph=0#/prison-yard');
    expect(hrefWithPreference('https://example.test/?a=b&morph=0#/x', 'morph')).toBe(
      'https://example.test/?a=b&morph=1#/x',
    );
  });

  test('leaves an address without the parameter alone', () => {
    expect(hrefWithPreference('https://example.test/dreamscapes/#/prison-yard', 'morph')).toBe(
      null,
    );
  });
});

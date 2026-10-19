import { describe, expect, test } from 'vitest';

import { hashFor, parseHash, readRouteState, resolveHash } from '../src/router';

const known = new Set(['cliff-cottage', 'prison-yard']);
const isKnown = (id: string): boolean => known.has(id);

describe('parseHash', () => {
  test.each([
    ['#/prison-yard', 'prison-yard'],
    ['#prison-yard', 'prison-yard'],
    ['#/prison-yard/', 'prison-yard'],
    ['#//prison-yard', 'prison-yard'],
    ['/prison-yard', 'prison-yard'],
    ['#/Prison-Yard', 'prison-yard'],
    ['#/%70rison-yard', 'prison-yard'],
    ['#/ prison-yard ', 'prison-yard'],
  ])('%s names %s', (hash, slug) => {
    expect(parseHash(hash)).toBe(slug);
  });

  test.each(['', '#', '#/', '#//', '#/%E0%A4%A'])('%j names nothing', (hash) => {
    expect(parseHash(hash)).toBeNull();
  });
});

describe('hashFor', () => {
  test('round-trips through parseHash', () => {
    expect(hashFor('prison-yard')).toBe('#/prison-yard');
    expect(parseHash(hashFor('prison-yard'))).toBe('prison-yard');
  });
});

describe('resolveHash', () => {
  test('a known slug resolves to itself', () => {
    expect(resolveHash('#/prison-yard', isKnown, 'cliff-cottage')).toEqual({
      id: 'prison-yard',
      matched: true,
    });
  });

  test('no hash resolves to the start scene', () => {
    expect(resolveHash('', isKnown, 'cliff-cottage')).toEqual({
      id: 'cliff-cottage',
      matched: false,
    });
  });

  test('an unknown slug resolves to the start scene', () => {
    expect(resolveHash('#/nowhere', isKnown, 'cliff-cottage')).toEqual({
      id: 'cliff-cottage',
      matched: false,
    });
  });

  test('a malformed hash resolves to the start scene', () => {
    expect(resolveHash('#/%E0%A4%A', isKnown, 'cliff-cottage').id).toBe('cliff-cottage');
  });
});

describe('readRouteState', () => {
  test('reads the depth this site stored', () => {
    expect(readRouteState({ depth: 7 })).toEqual({ depth: 7 });
  });

  test.each([null, undefined, 'x', {}, { depth: 0 }, { depth: 2.5 }, { depth: '3' }])(
    'ignores %j',
    (state) => {
      expect(readRouteState(state)).toBeNull();
    },
  );
});

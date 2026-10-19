import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

import { HINTS, HintScheduler, type HintReason } from '../src/hints';

describe('HintScheduler', () => {
  let hints: HintReason[];
  let scheduler: HintScheduler;

  beforeEach(() => {
    vi.useFakeTimers();
    hints = [];
    scheduler = new HintScheduler((reason) => hints.push(reason));
  });

  afterEach(() => {
    scheduler.stop();
    vi.useRealTimers();
  });

  test('uses the documented thresholds', () => {
    expect(HINTS.idleDelaysMs).toEqual([20_000, 40_000, 60_000]);
    expect(HINTS.missesBeforeHint).toBe(3);
  });

  test('does nothing until started', () => {
    scheduler.miss();
    scheduler.miss();
    scheduler.miss();
    vi.advanceTimersByTime(120_000);
    expect(hints).toEqual([]);
  });

  test('shimmers after 20 s idle, then backs off to 40 s and 60 s, then stays at 60 s', () => {
    scheduler.start();
    vi.advanceTimersByTime(19_999);
    expect(hints).toEqual([]);
    vi.advanceTimersByTime(1);
    expect(hints).toEqual(['idle']);

    vi.advanceTimersByTime(39_999);
    expect(hints).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(hints).toHaveLength(2);

    vi.advanceTimersByTime(59_999);
    expect(hints).toHaveLength(2);
    vi.advanceTimersByTime(1);
    expect(hints).toHaveLength(3);

    vi.advanceTimersByTime(60_000);
    expect(hints).toEqual(['idle', 'idle', 'idle', 'idle']);
  });

  test('three misses shimmer straight away', () => {
    scheduler.start();
    scheduler.miss();
    scheduler.miss();
    expect(hints).toEqual([]);
    scheduler.miss();
    expect(hints).toEqual(['misses']);
  });

  test('the miss count starts over after a shimmer', () => {
    scheduler.start();
    for (let i = 0; i < 5; i += 1) {
      scheduler.miss();
    }
    expect(hints).toEqual(['misses']);
    scheduler.miss();
    expect(hints).toEqual(['misses', 'misses']);
  });

  test('a miss-triggered shimmer restarts the idle wait one step further along', () => {
    scheduler.start();
    vi.advanceTimersByTime(15_000);
    scheduler.miss();
    scheduler.miss();
    scheduler.miss();
    expect(hints).toEqual(['misses']);
    // The 20 s wait that was running is gone; the next one is 40 s from the shimmer.
    vi.advanceTimersByTime(39_999);
    expect(hints).toEqual(['misses']);
    vi.advanceTimersByTime(1);
    expect(hints).toEqual(['misses', 'idle']);
  });

  test('misses alone do not delay the idle shimmer', () => {
    scheduler.start();
    vi.advanceTimersByTime(10_000);
    scheduler.miss();
    vi.advanceTimersByTime(10_000);
    expect(hints).toEqual(['idle']);
  });

  test('a new scene resets the back-off and the miss count', () => {
    scheduler.start();
    vi.advanceTimersByTime(20_000);
    scheduler.miss();
    scheduler.miss();
    expect(scheduler.currentDelayMs).toBe(40_000);

    scheduler.start();
    expect(scheduler.currentDelayMs).toBe(20_000);
    scheduler.miss();
    expect(hints).toEqual(['idle']);
    vi.advanceTimersByTime(20_000);
    expect(hints).toEqual(['idle', 'idle']);
  });

  test('stop cancels the pending shimmer and ignores misses', () => {
    scheduler.start();
    vi.advanceTimersByTime(19_000);
    scheduler.stop();
    expect(scheduler.isRunning).toBe(false);
    scheduler.miss();
    scheduler.miss();
    scheduler.miss();
    vi.advanceTimersByTime(120_000);
    expect(hints).toEqual([]);
  });
});

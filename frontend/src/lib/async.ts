// ─── Async helpers ─────────────────────────────────────────
//
// The recurring async shapes in the coordinators: collapsing a burst, a
// leading+trailing throttle, a cancel-previous timer, a monotonic request
// sequence, one shared in-flight request per key, an interval poll, and a single
// EventSource. Pure and dependency-free, so each is unit-testable on its own.

/**
 * Delay *fn* until *ms* of quiet; a call cancels the pending one.
 * The returned callable carries ``cancel`` for callers that must fire at once
 * (e.g. Enter committing a search instead of waiting out the debounce).
 */
export interface Debounced<A extends unknown[]> {
  (...args: A): void;
  cancel(): void;
}

export function debounce<A extends unknown[]>(
  fn: (...args: A) => void,
  ms: number,
): Debounced<A> {
  let timer: ReturnType<typeof setTimeout> | null = null;
  const wrapped = (...args: A): void => {
    if (timer !== null) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      fn(...args);
    }, ms);
  };
  wrapped.cancel = (): void => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
  };
  return wrapped;
}

/**
 * Leading + trailing throttle: the first call runs at once, calls during the window
 * coalesce, and the last runs when it closes. ``call`` resolves once the applied
 * run has finished, so a caller can await the step it scheduled.
 */
export interface Throttled<A extends unknown[]> {
  call(...args: A): Promise<void>;
  cancel(): void;
}

export function throttle<A extends unknown[]>(
  fn: (...args: A) => void | Promise<void>,
  ms: number,
): Throttled<A> {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let pending: A | null = null;
  let waiters: Array<() => void> = [];
  let lastRun = 0;

  const run = (): void => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
    const args = pending;
    pending = null;
    if (args === null) return;
    lastRun = Date.now();
    const release = waiters;
    waiters = [];
    void Promise.resolve(fn(...args)).finally(() => {
      for (const resolve of release) resolve();
    });
  };

  return {
    call(...args: A): Promise<void> {
      pending = args;
      const wait = new Promise<void>((resolve) => waiters.push(resolve));
      if (timer !== null) return wait;
      const elapsed = Date.now() - lastRun;
      if (elapsed >= ms) run();
      else timer = setTimeout(run, ms - elapsed);
      return wait;
    },
    cancel(): void {
      if (timer !== null) {
        clearTimeout(timer);
        timer = null;
      }
      pending = null;
      const release = waiters;
      waiters = [];
      for (const resolve of release) resolve();
    },
  };
}

/** A timer that is replaced on every ``set`` — the cancel-previous idiom. */
export interface NewTimer {
  set(fn: () => void, ms: number): void;
  cancel(): void;
}

export function clearOnNewTimer(): NewTimer {
  let timer: ReturnType<typeof setTimeout> | null = null;
  return {
    set(fn: () => void, ms: number): void {
      if (timer !== null) clearTimeout(timer);
      timer = setTimeout(() => {
        timer = null;
        fn();
      }, ms);
    },
    cancel(): void {
      if (timer !== null) {
        clearTimeout(timer);
        timer = null;
      }
    },
  };
}

/** A monotonic request sequence: a stale response checks ``isCurrent`` first. */
export interface Seq {
  next(): number;
  isCurrent(token: number): boolean;
}

export function latestSeq(): Seq {
  let current = 0;
  return {
    next: () => ++current,
    isCurrent: (token: number) => token === current,
  };
}

/** One in-flight request per key, so a repeat request joins the first. */
export interface KeyedInFlight<K, V> {
  has(key: K): boolean;
  run(key: K, load: () => Promise<V>): Promise<V>;
}

export function keyedInFlight<K, V>(): KeyedInFlight<K, V> {
  const slots = new Map<K, Promise<V>>();
  return {
    has: (key: K) => slots.has(key),
    run(key: K, load: () => Promise<V>): Promise<V> {
      const existing = slots.get(key);
      if (existing) return existing;
      const promise = load();
      slots.set(key, promise);
      // Clear the slot on a separate chain, so a caller awaiting the returned
      // promise resumes after the slot is already free. Swallow the rejection on
      // this chain: the caller owns the returned promise's error.
      void promise
        .finally(() => {
          if (slots.get(key) === promise) slots.delete(key);
        })
        .catch(() => {});
      return promise;
    },
  };
}

/**
 * A single library-wide task that coalesces concurrent runs: while one is in
 * flight a re-request marks exactly one trailing run, which starts when it settles.
 */
export class CoalescedTask {
  private _running = false;
  private _again = false;

  constructor(private readonly _task: () => Promise<void>) {}

  get running(): boolean {
    return this._running;
  }

  /** Run now, or mark a trailing run when one is already in flight. */
  run(): Promise<void> {
    if (this._running) {
      this._again = true;
      return Promise.resolve();
    }
    this._running = true;
    return this._task().finally(() => {
      this._running = false;
      if (this._again) {
        this._again = false;
        void this.run();
      }
    });
  }
}

/** A fixed-interval poll that replaces any running one on ``start``. */
export class Poller {
  private _timer: ReturnType<typeof setInterval> | null = null;

  constructor(
    private readonly _intervalMs: number,
    private readonly _tick: () => void,
  ) {}

  start(): void {
    this.stop();
    this._timer = setInterval(() => this._tick(), this._intervalMs);
  }

  stop(): void {
    if (this._timer !== null) {
      clearInterval(this._timer);
      this._timer = null;
    }
  }
}

/** One EventSource at a time: opening closes the previous stream. */
export class SseSubscription {
  private _source: EventSource | null = null;

  /** Open *url*, closing any previous stream, and return the new source. */
  open(url: string): EventSource {
    this.close();
    this._source = new EventSource(url);
    return this._source;
  }

  close(): void {
    this._source?.close();
    this._source = null;
  }

  get source(): EventSource | null {
    return this._source;
  }
}

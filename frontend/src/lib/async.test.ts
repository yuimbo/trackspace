import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  CoalescedTask,
  Poller,
  SseSubscription,
  clearOnNewTimer,
  debounce,
  keyedInFlight,
  latestSeq,
  throttle,
} from "./async";

describe("debounce", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("collapses a burst into one call with the last args", () => {
    const fn = vi.fn();
    const d = debounce(fn, 100);
    d("a");
    d("b");
    d("c");
    vi.advanceTimersByTime(99);
    expect(fn).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(fn).toHaveBeenCalledOnce();
    expect(fn).toHaveBeenCalledWith("c");
  });

  it("cancel drops the pending call", () => {
    const fn = vi.fn();
    const d = debounce(fn, 100);
    d("a");
    d.cancel();
    vi.advanceTimersByTime(200);
    expect(fn).not.toHaveBeenCalled();
  });
});

describe("throttle", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("runs the first call at once and the last on the trailing edge", async () => {
    const fn = vi.fn();
    const t = throttle(fn, 100);
    void t.call(1);
    expect(fn).toHaveBeenCalledWith(1);
    void t.call(2);
    void t.call(3);
    expect(fn).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(100);
    expect(fn).toHaveBeenCalledTimes(2);
    expect(fn).toHaveBeenLastCalledWith(3);
  });

  it("cancel drops the pending call and releases its waiters", async () => {
    const fn = vi.fn();
    const t = throttle(fn, 100);
    void t.call(1);
    const wait = t.call(2);
    t.cancel();
    await expect(wait).resolves.toBeUndefined();
    await vi.advanceTimersByTimeAsync(200);
    expect(fn).toHaveBeenCalledTimes(1);
  });
});

describe("clearOnNewTimer", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("replaces the pending callback", () => {
    const first = vi.fn();
    const second = vi.fn();
    const timer = clearOnNewTimer();
    timer.set(first, 100);
    timer.set(second, 100);
    vi.advanceTimersByTime(100);
    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledOnce();
  });

  it("cancel prevents the callback", () => {
    const fn = vi.fn();
    const timer = clearOnNewTimer();
    timer.set(fn, 100);
    timer.cancel();
    vi.advanceTimersByTime(200);
    expect(fn).not.toHaveBeenCalled();
  });
});

describe("latestSeq", () => {
  it("accepts only the newest token", () => {
    const seq = latestSeq();
    const a = seq.next();
    expect(seq.isCurrent(a)).toBe(true);
    const b = seq.next();
    expect(seq.isCurrent(a)).toBe(false);
    expect(seq.isCurrent(b)).toBe(true);
  });
});

describe("keyedInFlight", () => {
  it("shares one request per key and runs again after it settles", async () => {
    const flight = keyedInFlight<string, number>();
    let resolve!: (v: number) => void;
    const load = vi.fn(() => new Promise<number>((r) => (resolve = r)));
    const a = flight.run("x", load);
    const b = flight.run("x", load);
    expect(flight.has("x")).toBe(true);
    expect(load).toHaveBeenCalledOnce();
    expect(a).toBe(b);
    resolve(1);
    await expect(a).resolves.toBe(1);
    expect(flight.has("x")).toBe(false);
    await expect(flight.run("x", () => Promise.resolve(2))).resolves.toBe(2);
    expect(load).toHaveBeenCalledOnce();
  });
});

describe("CoalescedTask", () => {
  it("runs once and coalesces a concurrent re-request into a trailing run", async () => {
    let resolve!: () => void;
    const task = vi.fn(
      () => new Promise<void>((r) => (resolve = r)),
    );
    const coalesced = new CoalescedTask(task);
    const first = coalesced.run();
    expect(coalesced.running).toBe(true);
    void coalesced.run();
    expect(task).toHaveBeenCalledOnce();
    resolve();
    await first;
    await vi.waitFor(() => expect(task).toHaveBeenCalledTimes(2));
    expect(coalesced.running).toBe(true);
  });

  it("runs again after it settles", async () => {
    const task = vi.fn(() => Promise.resolve());
    const coalesced = new CoalescedTask(task);
    await coalesced.run();
    await coalesced.run();
    expect(task).toHaveBeenCalledTimes(2);
    expect(coalesced.running).toBe(false);
  });
});

describe("Poller", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("ticks on the interval and stops", () => {
    const tick = vi.fn();
    const poller = new Poller(50, tick);
    poller.start();
    vi.advanceTimersByTime(150);
    expect(tick).toHaveBeenCalledTimes(3);
    poller.stop();
    vi.advanceTimersByTime(150);
    expect(tick).toHaveBeenCalledTimes(3);
  });

  it("start replaces a running poll", () => {
    const tick = vi.fn();
    const poller = new Poller(50, tick);
    poller.start();
    vi.advanceTimersByTime(50);
    poller.start();
    vi.advanceTimersByTime(50);
    expect(tick).toHaveBeenCalledTimes(2);
  });
});

describe("SseSubscription", () => {
  class FakeEventSource {
    static instances: FakeEventSource[] = [];
    closed = false;
    constructor(readonly url: string) {
      FakeEventSource.instances.push(this);
    }
    close(): void {
      this.closed = true;
    }
  }

  beforeEach(() => {
    FakeEventSource.instances = [];
    vi.stubGlobal("EventSource", FakeEventSource);
  });
  afterEach(() => vi.unstubAllGlobals());

  it("opens one stream at a time and closes it", () => {
    const sub = new SseSubscription();
    sub.open("/a");
    const [first] = FakeEventSource.instances;
    sub.open("/b");
    const [, second] = FakeEventSource.instances;
    expect(first.closed).toBe(true);
    expect(second.closed).toBe(false);
    expect(sub.source).toBe(second);
    sub.close();
    expect(second.closed).toBe(true);
    expect(sub.source).toBeNull();
  });
});

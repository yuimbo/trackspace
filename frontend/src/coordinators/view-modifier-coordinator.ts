// ─── ViewModifierCoordinator ─────────────────────────────────
//
// Owns the stack of **active view modifiers** — reversible changes to what the
// canvas shows (a search, a proximity filter, a scope, exclusions, tag filters,
// shown externals). Each entry is one chip in the corner stack; **Esc** dismisses
// the newest and a chip's × dismisses one.
//
// A modifier owns only its *descriptor*; the coordinator owns the clear callback
// that undoes it. `dismiss` removes the entry and runs that callback, so the
// underlying state (search results, hidden paths, the scope, …) is reverted in one
// place. Callbacks are written to be idempotent: a direct close (a lens switch
// dropping a scope, `clearExclusions`) also removes its entry, so the stack never
// goes stale. See `design-patterns.md` → *Active view modifiers*.

import type { Model, ViewModifier, ViewModifierKind } from "../model";

/** The parts of the corner chip view this coordinator drives. */
export interface ViewModifierViewLike {
  render(modifiers: ViewModifier[]): void;
}

/**
 * The stack operations another coordinator needs. Structural so a test can pass a
 * stub; `ViewModifierCoordinator` satisfies it.
 */
export interface ViewModifierSink {
  push(mod: Omit<ViewModifier, "id">, clear: () => void): string;
  update(id: string, patch: Partial<Omit<ViewModifier, "id">>): void;
  dismiss(id: string): void;
  remove(id: string | null): void;
  find(kind: ViewModifierKind): ViewModifier | undefined;
}

export interface ViewModifierCoordinatorDeps {
  model: Model;
  view: ViewModifierViewLike;
}

export class ViewModifierCoordinator {
  /** Clear callback per modifier id (runs when the entry is dismissed). */
  private _clearers = new Map<string, () => void>();
  private _seq = 0;

  constructor(private deps: ViewModifierCoordinatorDeps) {}

  /** Every active modifier, oldest → newest (the rendered stack). */
  get stack(): ViewModifier[] {
    return this.deps.model.viewModifiers;
  }

  /**
   * Push a modifier on top of the stack. *clear* undoes the underlying state;
   * it must be idempotent, because a direct close may also call it.
   */
  push(mod: Omit<ViewModifier, "id">, clear: () => void): string {
    const id = `vm-${++this._seq}`;
    this._clearers.set(id, clear);
    this.deps.model.viewModifiers = [
      ...this.deps.model.viewModifiers,
      { ...mod, id },
    ];
    this._render();
    return id;
  }

  /** Update a modifier's readout in place (a live slider drag, a refit toggle). */
  update(id: string, patch: Partial<Omit<ViewModifier, "id">>): void {
    let changed = false;
    this.deps.model.viewModifiers = this.deps.model.viewModifiers.map((m) => {
      if (m.id !== id) return m;
      changed = true;
      return { ...m, ...patch };
    });
    if (changed) this._render();
  }

  /** Dismiss one modifier (a chip's ×): remove it, then clear its state. */
  dismiss(id: string): void {
    const clear = this._clearers.get(id);
    if (!clear) return;
    this._clearers.delete(id);
    this.deps.model.viewModifiers = this.deps.model.viewModifiers.filter(
      (m) => m.id !== id,
    );
    this._render();
    clear();
  }

  /**
   * Remove an entry whose state was already cleared directly (a scope dropped by a
   * lens switch, an exclusion cleared by **Alt+s**). No callback runs.
   */
  remove(id: string | null): void {
    if (!id) return;
    if (!this._clearers.has(id)) return;
    this._clearers.delete(id);
    this.deps.model.viewModifiers = this.deps.model.viewModifiers.filter(
      (m) => m.id !== id,
    );
    this._render();
  }

  /** **Esc**: dismiss the newest modifier. False when the stack is empty. */
  popNewest(): boolean {
    const stack = this.deps.model.viewModifiers;
    if (!stack.length) return false;
    this.dismiss(stack[stack.length - 1].id);
    return true;
  }

  /** The active modifier of a kind, if any (a chip's state to read). */
  find(kind: ViewModifierKind): ViewModifier | undefined {
    return this.deps.model.viewModifiers.find((m) => m.kind === kind);
  }

  /** Re-render the chip stack from the model (used after an external clear). */
  sync(): void {
    this._render();
  }

  private _render(): void {
    this.deps.view.render(this.deps.model.viewModifiers);
  }
}

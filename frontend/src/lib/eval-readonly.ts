// ─── Read-only eval guard ────────────────────────────────────
//
// While the layout-evaluation gallery is active (`?layout_eval=1`) the app is a
// read-only browser: no tag/folder/root/imprint writes and no ML re-computation
// may reach the server, and no in-memory command may run. The single source of
// truth for that is this module, so both the owned HTTP layer (`lib/api.ts`) and
// any command path can consult one flag. There is deliberately no way to mutate
// data here — only the coordinator that owns the eval session toggles it on, and
// it stays on for the whole page (exiting the gallery is a reload without the
// query param).
//
// This is *not* a `fetch` monkey-patch and it changes no backend permissions: it
// only stops the client from issuing writes it is not allowed to make in this
// mode, and surfaces a clear error instead of a silent no-op.

let _evalReadOnly = false;

/** Enter/leave read-only eval mode. Only the eval session owner may call this. */
export function setEvalReadOnly(v: boolean): void {
  _evalReadOnly = v;
}

/** True while the read-only layout-evaluation gallery is active. */
export function isEvalReadOnly(): boolean {
  return _evalReadOnly;
}

/**
 * Throw unless the page is not in read-only eval mode. Call from any owned
 * write path (HTTP helpers, command execution) so a mutation cannot slip through.
 */
export function assertWritable(what: string): void {
  if (_evalReadOnly) {
    throw new Error(
      `Read-only layout evaluation is active — ${what} is disabled. ` +
        `Exit the gallery (remove ?layout_eval=1) to edit.`,
    );
  }
}

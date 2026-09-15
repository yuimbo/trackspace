// ─── API helpers ─────────────────────────────────────────────
import { toast } from "./toast";

/**
 * Fetch JSON and surface backend errors as a toast.
 *
 * The backend returns ``{"ok": false, "error": "..."}`` for every failure, so
 * that message is preferable to a bare status line. Pass ``silent`` to suppress
 * the toast when the caller surfaces the error itself (e.g. commands, whose
 * manager already reports "action failed — reverted").
 */
async function request<T>(url: string, init: RequestInit | undefined, silent: boolean): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = (await res.json()) as { error?: unknown };
      if (body && typeof body.error === "string" && body.error) detail = body.error;
    } catch {
      /* non-JSON error body; keep the status line */
    }
    if (!silent) toast(detail, "error");
    throw new Error(detail);
  }
  return res.json();
}

export function api<T = unknown>(url: string, opts?: RequestInit): Promise<T> {
  return request<T>(url, opts, false);
}

export function postJSON<T = unknown>(url: string, body: unknown, silent = false): Promise<T> {
  return request<T>(
    url,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
    silent,
  );
}

function rootLabelForId(rootId: string): string {
  if (!rootId) return "";
  const row = document.querySelector<HTMLElement>(
    `.folder-row[data-root-id="${CSS.escape(rootId)}"]`,
  );
  return (
    row?.querySelector<HTMLElement>(".folder-label")?.textContent?.trim() ?? ""
  );
}

/**
 * Convert an internal virtual folder path (`rootId/sub/path`) to a
 * user-facing label (`Root Name/sub/path`) when possible.
 */
export function displayFolderPath(folder: string): string {
  const raw = (folder || "").trim();
  if (!raw) return "Library root";

  const slash = raw.indexOf("/");
  const rootId = slash >= 0 ? raw.slice(0, slash) : raw;
  const rest = slash >= 0 ? raw.slice(slash + 1) : "";
  const rootLabel = rootLabelForId(rootId);
  if (!rootLabel) return raw;
  return rest ? `${rootLabel}/${rest}` : rootLabel;
}

/** Leaf label for compact UI rows, preserving root-name mapping. */
export function displayFolderLeaf(folder: string): string {
  const full = displayFolderPath(folder);
  if (full === "Library root") return "(root)";
  const parts = full.split("/");
  return parts[parts.length - 1] || full;
}

// ─── Hover tooltip ("hockey card") ────────────────────────────
//
// The tooltip is a small card rather than a block of text: a header (folder +
// title), a BPM/key meta row, structure chips (the audition sections), the genre
// group (cluster + genre-spanning badge), and two stat-bar areas — user tags and
// audio features. `buildTooltip` is pure (unit-testable) and `renderTooltip` is the
// only DOM code, so the canvas view stays free of layout.

import {
  type HoverTarget,
  type Model,
  type PreviewSection,
  type Track,
  type TrackAudioFeatures,
  LIBROSA_FEATURE_AXIS_IDS,
  isAudioFeatureAxisId,
} from "../model";
import { clusterColor } from "./colors";
import { displayFolderPath } from "./path-presenter";
import { structureChips, type StructureChip } from "./structure";

/** One stat row: a 0–1 bar plus its human readout. */
export interface TooltipBar {
  label: string;
  value: number;
  readout: string;
  /** Navigation target when the row is a key/BPM stat (hover glows + S/H act on it). */
  target?: HoverTarget;
}

export interface TooltipGenre {
  /** Cluster label, or ``null`` when the track has no cluster assignment. */
  name: string | null;
  /** Swatch colour, or ``null`` when there is no cluster. */
  color: string | null;
  spanning: boolean;
  /** Hover target for the cluster name (``null`` when unassigned). */
  target: HoverTarget | null;
}

/** One ranked neighbour: the raw path plus the similarity score. */
export interface TooltipResult {
  path: string;
  score: number;
}

/** Interactive inspect state, present only while the card is pinned. */
export interface InspectState {
  /** The inspected track. */
  path: string;
  busy: boolean;
  message: string | null;
  /** Which ranking produced ``results`` (``far`` = anti-track). */
  mode: "near" | "far" | null;
  results: TooltipResult[];
  /**
   * Neighbourhood slider, 0–100 %. 100 = the whole library is visible; lowering
   * it hides everything but the closest *proximity*% of the ranked library and
   * re-homes the view as it narrows. Excluded from the re-render key so a drag
   * does not rebuild the slider under the pointer.
   */
  proximity: number;
}

export interface TooltipCard {
  /** Identity of every rendered input; the canvas re-renders only when it changes. */
  key: string;
  path: string;
  folder: string;
  /** Hover target for the folder kicker (the track's folder). */
  folderTarget: HoverTarget;
  title: string;
  meta: string[];
  chips: StructureChip[];
  genre: TooltipGenre | null;
  tags: TooltipBar[];
  features: TooltipBar[];
  /** Section currently auditioned by the hover preview (highlighted chip). */
  activeSection: PreviewSection | null;
  /** Pinned inspect card; ``null`` for the plain hover card. */
  inspect: InspectState | null;
  /** ``inspect.results`` resolved against the loaded library for display. */
  results: ResultRow[];
  /** Footer hint on the plain hover card (how to pin it). */
  hint: string | null;
}

/** A ranked neighbour resolved against the loaded library for display. */
export interface ResultRow {
  path: string;
  title: string;
  folder: string;
  score: number;
}

const SPANNING_HINT =
  "Intro and drop excerpts disagree (embedding dispersion ≥ 0.25) — the track straddles genres";

/** Footer hint on the plain hover card: how to open the interactive card. */
const HOLD_ALT_HINT = "Hold ⌥ (Alt) to interact";

const clamp01 = (v: number): number => Math.max(0, Math.min(1, v));

/** Feature readout: the real unit where one exists, the normalised value otherwise. */
function featureReadout(k: string, v: number, track: Track): string {
  if (k === "tempo" && track.bpm != null && Number.isFinite(track.bpm))
    return `${Math.round(track.bpm)} BPM`;
  if (k === "key" && track.musical_key) return track.musical_key;
  return v.toFixed(2);
}

function audioFeatureRows(track: Track, m: Model): TooltipBar[] {
  const row = track.audio_features;
  const rows: TooltipBar[] = [];
  for (const k of LIBROSA_FEATURE_AXIS_IDS) {
    const fromAudio = row ? row[k as keyof TrackAudioFeatures] : undefined;
    const v = typeof fromAudio === "number" ? fromAudio : track.tags[k];
    if (typeof v !== "number" || !Number.isFinite(v)) continue;
    const bar: TooltipBar = {
      label: m.axisDisplayName(k),
      value: clamp01(v),
      readout: featureReadout(k, v, track),
    };
    // Key and tempo are navigation targets: hovering the row glows every track
    // with the same key / a near tempo, and S/H act on them.
    if (k === "tempo" && track.bpm != null && Number.isFinite(track.bpm))
      bar.target = { kind: "bpm", value: track.bpm };
    else if (k === "key" && track.musical_key)
      bar.target = { kind: "key", value: track.musical_key };
    rows.push(bar);
  }
  return rows;
}

/** User tags (ID3), one bar each — feature axis ids are rendered separately. */
function tagRows(track: Track): TooltipBar[] {
  const rows: TooltipBar[] = [];
  for (const [k, v] of Object.entries(track.tags)) {
    if (isAudioFeatureAxisId(k)) continue;
    if (typeof v !== "number" || !Number.isFinite(v)) continue;
    rows.push({ label: k, value: clamp01(v), readout: v.toFixed(2) });
  }
  return rows;
}

/**
 * Build the tooltip card for a track. Pure over the model so it can be asserted
 * in tests; the canvas only renders the result. *inspect* is passed while the card
 * is pinned and turns the navigate/results sections on.
 */
export function buildTooltip(
  track: Track,
  m: Model,
  inspect: InspectState | null = null,
): TooltipCard {
  const folder = displayFolderPath(track.folder || "");
  const ar = (track.artist || "").trim();
  const ti = (track.title || "").trim();
  const title = ar && ti ? `${ar} — ${ti}` : ti || ar || track.filename;

  const meta: string[] = [];
  if (track.bpm != null && Number.isFinite(track.bpm))
    meta.push(`${Math.round(track.bpm)} BPM`);
  if (track.musical_key) meta.push(track.musical_key);

  const structure = m.trackStructure.get(track.path);
  const chips = structureChips(structure);
  const activeSection =
    m.previewTrackPath === track.path ? m.previewSection : null;

  const cid = m.clusterAssignments.get(track.path);
  const spanning = structure?.genre_spanning === true;
  const genre: TooltipGenre | null =
    cid === undefined && !spanning
      ? null
      : {
          name:
            cid === undefined
              ? null
              : m.clusterNames.get(cid) || `Cluster ${cid}`,
          color: cid === undefined ? null : clusterColor(cid),
          spanning,
          target: cid === undefined ? null : { kind: "cluster", id: cid },
        };

  const tags = tagRows(track);
  const features = audioFeatureRows(track, m);
  const results = inspect
    ? inspect.results.map((r) => resolveResult(r, m))
    : [];

  const key = [
    track.path,
    track.filename,
    track.artist,
    track.title,
    track.bpm ?? "",
    track.musical_key ?? "",
    track.folder ?? "",
    structure ? JSON.stringify(structure) : "",
    chips.map((c) => c.text).join("|"),
    tags.map((t) => `${t.label}:${t.readout}`).join("|"),
    features.map((f) => `${f.label}:${f.readout}:${f.value}`).join("|"),
    activeSection ?? "",
    cid ?? "",
    m.clusterRevision ?? "",
    inspect ? inspectKey(inspect) : "",
  ].join("\u0001");

  return {
    key,
    path: track.path,
    folder,
    folderTarget: { kind: "folder", prefix: track.folder ?? "" },
    title,
    meta,
    chips,
    genre,
    tags,
    features,
    activeSection,
    inspect,
    results,
    // The hint only belongs on the plain card: once pinned it is interactive.
    hint: inspect ? null : HOLD_ALT_HINT,
  };
}

/** Resolve one ranked neighbour against the loaded library for display. */
function resolveResult(result: TooltipResult, m: Model): ResultRow {
  const t = m.trackByPath(result.path);
  const ar = (t?.artist || "").trim();
  const ti = (t?.title || "").trim();
  return {
    path: result.path,
    title: ar && ti ? `${ar} — ${ti}` : ti || ar || t?.filename || result.path,
    folder: displayFolderPath(t?.folder ?? ""),
    score: result.score,
  };
}

/** Identity of the inspect state, so the card re-renders when a ranking lands.
 *
 * ``proximity`` is deliberately absent: the slider updates it live, and rebuilding
 * the card mid-drag would replace the input under the pointer.
 */
function inspectKey(inspect: InspectState): string {
  return [
    inspect.path,
    inspect.busy ? "busy" : "",
    inspect.message ?? "",
    inspect.mode ?? "",
    inspect.results.map((r) => `${r.path}:${r.score}`).join("|"),
  ].join("\u0001");
}

/* ── DOM rendering ─────────────────────────────────────────── */

function el(tag: string, className?: string, text?: string): HTMLElement {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/** Hint shown on every navigation-target element in the card. */
const TARGET_HINT = "Highlight matches · S solo · H fit";

/**
 * Mark an element as a navigation target: the canvas reads `data-hover-target` to
 * glow matching dots, and S/H act on it exactly as they do for a sidebar row.
 * *title* overrides the generic hint (a result row has its own affordance text).
 */
function markTarget(
  el: HTMLElement,
  target: HoverTarget,
  title: string = TARGET_HINT,
): HTMLElement {
  el.classList.add("tt-target");
  el.dataset.hoverTarget = JSON.stringify(target);
  el.title = title;
  return el;
}

function statSection(
  title: string,
  rows: TooltipBar[],
  extra?: string,
): HTMLElement {
  const section = el("div", `tt-section${extra ? " " + extra : ""}`);
  section.appendChild(el("div", "tt-section-title", title));
  for (const r of rows) {
    const row = el("div", "tt-stat");
    row.appendChild(el("span", "tt-stat-label", r.label));
    const bar = el("span", "tt-stat-bar");
    const fill = el("i");
    fill.style.width = `${Math.round(r.value * 100)}%`;
    bar.appendChild(fill);
    row.appendChild(bar);
    row.appendChild(el("span", "tt-stat-value", r.readout));
    if (r.target) markTarget(row, r.target);
    section.appendChild(row);
  }
  return section;
}

/** Render a built card into *host*, replacing any previous content. */
export function renderTooltip(host: HTMLElement, card: TooltipCard): void {
  const root = el("div", "tt-card");
  root.appendChild(markTarget(el("div", "tt-folder", card.folder), card.folderTarget));
  root.appendChild(el("div", "tt-title", card.title));

  if (card.meta.length) {
    const meta = el("div", "tt-meta");
    for (const part of card.meta)
      meta.appendChild(el("span", "tt-meta-chip", part));
    root.appendChild(meta);
  }

  if (card.chips.length) {
    const chips = el("div", "tt-chips");
    for (const c of card.chips) {
      const chip = el("span", "tt-chip", c.text);
      chip.dataset.section = c.section;
      if (c.section === card.activeSection) chip.classList.add("active");
      chips.appendChild(chip);
    }
    root.appendChild(chips);
  }

  if (card.genre) {
    const genre = el("div", "tt-genre");
    if (card.genre.color) {
      const swatch = el("span", "tt-genre-swatch");
      swatch.style.background = card.genre.color;
      genre.appendChild(swatch);
    }
    if (card.genre.name)
      genre.appendChild(el("span", "tt-genre-name", card.genre.name));
    if (card.genre.spanning) {
      const badge = el("span", "tt-badge", "genre-spanning");
      badge.title = SPANNING_HINT;
      genre.appendChild(badge);
    }
    if (card.genre.target) markTarget(genre, card.genre.target);
    root.appendChild(genre);
  }

  if (card.tags.length) root.appendChild(statSection("Personal tags", card.tags));
  if (card.features.length)
    root.appendChild(statSection("Features", card.features, "tt-features"));

  if (card.inspect)
    root.appendChild(inspectSection(card.inspect, card.results));
  if (card.hint) root.appendChild(el("div", "tt-hint", card.hint));

  host.replaceChildren(root);
}

/** Navigate buttons + the last ranking's rows (pinned card only). */
function inspectSection(
  inspect: InspectState,
  results: ResultRow[],
): HTMLElement {
  const section = el("div", "tt-section tt-inspect");
  const head = el("div", "tt-inspect-head");
  head.appendChild(el("div", "tt-section-title", "Navigate"));
  const close = el("button", "tt-close", "×");
  close.dataset.inspectClose = "";
  close.title = "Close inspect (Esc)";
  head.appendChild(close);
  section.appendChild(head);

  const buttons = el("div", "tt-nav-buttons");
  const actions: [string, string, string][] = [
    ["anti", "Anti-track", "The track furthest away in the embedding space"],
    ["near", "Nearest", "The closest tracks in the embedding space"],
  ];
  for (const [action, label, hint] of actions) {
    const btn = el("button", "tt-nav-btn", label) as HTMLButtonElement;
    btn.dataset.inspectAction = action;
    btn.title = hint;
    btn.disabled = inspect.busy;
    buttons.appendChild(btn);
  }
  section.appendChild(buttons);

  // Neighbourhood slider: 100 % = the whole library, narrower = the closest
  // fraction of it. Replaces the old discrete "Solo" (top-N) jump; dragging the
  // view re-homes gradually instead of snapping to ten neighbours.
  const proximity = el("div", "tt-proximity");
  const proxHead = el("div", "tt-proximity-head");
  proxHead.appendChild(el("span", "tt-proximity-label", "Neighbourhood"));
  const proxValue = el(
    "span",
    "tt-proximity-value",
    `${Math.round(inspect.proximity)}%`,
  );
  proxValue.dataset.inspectProximityValue = "";
  proxHead.appendChild(proxValue);
  proximity.appendChild(proxHead);
  const range = document.createElement("input");
  range.type = "range";
  range.className = "tt-proximity-range";
  range.min = "0";
  range.max = "100";
  range.step = "1";
  range.value = String(Math.round(inspect.proximity));
  range.dataset.inspectProximity = "";
  range.title = "Drag to grow or shrink the neighbourhood around this track";
  range.disabled = inspect.busy;
  proximity.appendChild(range);
  section.appendChild(proximity);

  if (inspect.busy) {
    section.appendChild(el("div", "tt-note", "Searching the embedding space…"));
  } else if (inspect.message) {
    section.appendChild(el("div", "tt-note tt-note-error", inspect.message));
  } else if (results.length) {
    section.appendChild(
      el(
        "div",
        "tt-section-title",
        inspect.mode === "far" ? "Anti-track" : "Nearest",
      ),
    );
    for (const r of results) section.appendChild(resultRow(r));
  }
  return section;
}

function resultRow(result: ResultRow): HTMLElement {
  const row = el("div", "tt-result");
  row.dataset.inspectFocus = result.path;
  // Hovering a row glows it in the map and auditions it immediately, like
  // hovering a canvas dot.
  row.dataset.preview = result.path;
  markTarget(
    row,
    { kind: "track", path: result.path },
    "Glow in the map · click to select, fit the view, and audition",
  );

  const text = el("div", "tt-result-text");
  text.appendChild(el("div", "tt-result-title", result.title));
  text.appendChild(el("div", "tt-result-folder", result.folder));
  row.appendChild(text);

  row.appendChild(el("span", "tt-result-score", result.score.toFixed(2)));
  return row;
}

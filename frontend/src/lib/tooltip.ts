// ─── Hover tooltip ("hockey card") ────────────────────────────
//
// The tooltip is a small card rather than a block of text: a title/artist header
// (collapsing to the filename when metadata is missing), a location stack of the
// folder and cluster as icon-distinguished coloured tags, structure chips (the
// audition sections), and two stat areas — user tags and audio features. BPM and key
// appear only in Features: tempo as a bar and key as a chip, so they are not
// duplicated near the title. `buildTooltip` is pure (unit-testable) and
// `renderTooltip` is the only DOM code, so the canvas view stays free of layout.

import {
  type HoverTarget,
  type Model,
  type PreviewSection,
  type Track,
  type TrackAudioFeatures,
  LIBROSA_FEATURE_AXIS_IDS,
  isAudioFeatureAxisId,
} from "../model";
import { clusterColor, dirTagColors } from "./colors";
import { displayFolderPath } from "./path-presenter";
import { structureChips, type StructureChip } from "./structure";

/** One stat row: a 0–1 bar (or a chip for categorical stats) plus its readout. */
export interface TooltipBar {
  label: string;
  value: number;
  readout: string;
  /** Categorical stat (key): render a chip, not a magnitude bar. */
  chip?: boolean;
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
  /** Icon colour stops: one solid colour, or a parent→own gradient when nested. */
  folderColors: string[];
  /** Hover target for the folder tag (the track's folder). */
  folderTarget: HoverTarget;
  /** Bold primary line: the title, falling back to the filename. */
  title: string;
  /** Secondary artist line, or ``null`` when the track has no artist metadata. */
  artist: string | null;
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

function audioFeatureRows(track: Track, m: Model): TooltipBar[] {
  const row = track.audio_features;
  const rows: TooltipBar[] = [];
  for (const k of LIBROSA_FEATURE_AXIS_IDS) {
    // Mode is folded into the Key chip: ``musical_key`` already carries the
    // major/minor suffix ("Am" = minor, "A" = major), so a 0/1 bar would only
    // repeat it. It stays a canvas axis / filter, just not a tooltip row.
    if (k === "mode") continue;
    // Key and BPM are labels, not 0–1 magnitudes, so they render as chips.
    // The chip's ``value`` is unused.
    if (k === "key") {
      if (!track.musical_key) continue;
      rows.push({
        label: m.axisDisplayName(k),
        value: 0,
        readout: track.musical_key,
        chip: true,
        target: { kind: "key", value: track.musical_key },
      });
      continue;
    }
    // Tempo is a BPM chip or nothing: the normalised 0–1 value is not a useful
    // standalone bar, so there is no bar fallback when BPM is unknown.
    if (k === "tempo") {
      if (track.bpm == null || !Number.isFinite(track.bpm)) continue;
      rows.push({
        label: m.axisDisplayName(k),
        value: 0,
        readout: `${Math.round(track.bpm)} BPM`,
        chip: true,
        target: { kind: "bpm", value: track.bpm },
      });
      continue;
    }
    const fromAudio = row ? row[k as keyof TrackAudioFeatures] : undefined;
    const v = typeof fromAudio === "number" ? fromAudio : track.tags[k];
    if (typeof v !== "number" || !Number.isFinite(v)) continue;
    rows.push({
      label: m.axisDisplayName(k),
      value: clamp01(v),
      readout: v.toFixed(2),
    });
  }
  return rows;
}

/** User tags (ID3), one bar each — feature axis ids are rendered separately.
 *
 * Sorted by label: the object's key order is not stable across tracks, and an
 * arbitrary order makes the card jump around as you move between dots.
 */
function tagRows(track: Track): TooltipBar[] {
  const rows: TooltipBar[] = [];
  for (const [k, v] of Object.entries(track.tags)) {
    if (isAudioFeatureAxisId(k)) continue;
    if (typeof v !== "number" || !Number.isFinite(v)) continue;
    rows.push({ label: k, value: clamp01(v), readout: v.toFixed(2) });
  }
  return rows.sort((a, b) => a.label.localeCompare(b.label));
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
  const folderPath = track.folder || "";
  const folder = displayFolderPath(folderPath);
  // Metadata is optional: with no artist and no title the card collapses to a
  // single filename line instead of an empty header.
  const artist = (track.artist || "").trim() || null;
  const title = (track.title || "").trim() || track.filename;

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
    folderColors: dirTagColors(folderPath),
    folderTarget: { kind: "folder", prefix: track.folder ?? "" },
    title,
    artist,
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

const SVG_NS = "http://www.w3.org/2000/svg";
/** Feather-style folder / tag glyphs, stroked in the tag's contrast colour. */
const FOLDER_GLYPH = "M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z";
const TAG_GLYPH = "M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z";

/** Folder or tag glyph, stroked in `currentColor` so the tag sets the contrast. */
function locationIcon(kind: "folder" | "tag"): SVGSVGElement {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("class", "tt-loc-icon");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute("d", kind === "folder" ? FOLDER_GLYPH : TAG_GLYPH);
  svg.appendChild(path);
  if (kind === "tag") {
    const dot = document.createElementNS(SVG_NS, "circle");
    dot.setAttribute("cx", "7");
    dot.setAttribute("cy", "7");
    dot.setAttribute("r", "1");
    dot.setAttribute("fill", "currentColor");
    svg.appendChild(dot);
  }
  return svg;
}

/** Tag fill: a solid colour, or a parent→own gradient for a nested folder. */
function tagFill(colors: string[]): string | null {
  if (colors.length === 0) return null;
  if (colors.length === 1) return colors[0];
  return `linear-gradient(135deg, ${colors.join(", ")})`;
}

/** Relative luminance of an `hsl(h,s%,l%)` colour, for contrast decisions. */
function hslLuminance(h: number, s: number, l: number): number {
  const a = (s / 100) * Math.min(l / 100, 1 - l / 100);
  const channel = (n: number): number => {
    const k = (n + h / 30) % 12;
    const v = (l / 100) - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
    return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
  };
  return 0.2126 * channel(0) + 0.7152 * channel(8) + 0.0722 * channel(4);
}

/** Icon/name colour with enough contrast for the tag's fill(s). */
function tagTextColor(colors: string[]): string {
  const luminances = colors.map((c) => {
    const m = /hsl\(\s*([\d.]+)\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%\s*\)/.exec(c);
    return m ? hslLuminance(+m[1], +m[2], +m[3]) : 0.5;
  });
  // 0.179 is where dark and light text have equal contrast against the fill.
  const avg = luminances.reduce((a, b) => a + b, 0) / luminances.length;
  return avg > 0.179 ? "#0d1219" : "#f4f7fb";
}

/** One filled location chip: a coloured type icon plus its name. */
function locationTag(
  kind: "folder" | "tag",
  label: string,
  colors: string[],
  target: HoverTarget,
): HTMLElement {
  const tag = el("span", `tt-loc-tag tt-${kind}`);
  const fill = tagFill(colors);
  if (fill) {
    tag.classList.add("tt-loc-filled");
    tag.style.background = fill;
    tag.style.color = tagTextColor(colors);
  }
  tag.appendChild(locationIcon(kind));
  tag.appendChild(el("span", "tt-loc-name", label));
  return markTarget(tag, target);
}

/** Cluster chip: filled tag colour + tag icon + name + spanning badge. */
function clusterTag(genre: TooltipGenre): HTMLElement {
  const tag = el("span", "tt-loc-tag tt-cluster");
  const colors = genre.color ? [genre.color] : [];
  const fill = tagFill(colors);
  if (fill) {
    tag.classList.add("tt-loc-filled");
    tag.style.background = fill;
    tag.style.color = tagTextColor(colors);
  }
  tag.appendChild(locationIcon("tag"));
  if (genre.name) tag.appendChild(el("span", "tt-loc-name", genre.name));
  if (genre.spanning) {
    const badge = el("span", "tt-badge", "genre-spanning");
    badge.title = SPANNING_HINT;
    tag.appendChild(badge);
  }
  if (genre.target) markTarget(tag, genre.target);
  return tag;
}

function statSection(
  title: string,
  rows: TooltipBar[],
  extra?: string,
): HTMLElement {
  const section = el("div", `tt-section${extra ? " " + extra : ""}`);
  section.appendChild(el("div", "tt-section-title", title));
  for (const r of rows) {
    if (r.chip) {
      // Categorical stat (key): the value has no magnitude to bar, so show a
      // chip in the same row style as the other stats.
      const chip = el("span", "tt-stat-chip");
      chip.appendChild(el("span", "tt-stat-chip-label", r.label));
      chip.appendChild(el("span", "tt-stat-chip-value", r.readout));
      if (r.target) markTarget(chip, r.target);
      section.appendChild(chip);
      continue;
    }
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
  const head = el("div", "tt-head");
  head.appendChild(el("div", "tt-title", card.title));
  if (card.artist) head.appendChild(el("div", "tt-artist", card.artist));
  root.appendChild(head);

  const location = el("div", "tt-loc");
  location.appendChild(
    locationTag("folder", card.folder, card.folderColors, card.folderTarget),
  );
  if (card.genre) location.appendChild(clusterTag(card.genre));
  root.appendChild(location);

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

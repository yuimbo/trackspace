// ─── Hover tooltip ("hockey card") ────────────────────────────
//
// The tooltip is a small card rather than a block of text: a title/artist header
// (collapsing to the filename when metadata is missing), a location stack of the
// folder and cluster as icon-distinguished coloured tags, a preview seekbar (the
// audition timeline with its structure sections as markers), and two stat areas — user
// tags and audio features. BPM and key
// appear only in Features, as chips filled with the same colours as the tempo/key
// colour modes, so they are not duplicated near the title. `buildTooltip` is pure
// (unit-testable) and `renderTooltip` is the only DOM code, so the canvas view
// stays free of layout.

import {
  type HoverTarget,
  type Model,
  type PreviewSection,
  type Track,
  type TrackAudioFeatures,
  LIBROSA_FEATURE_AXIS_IDS,
  isAudioFeatureAxisId,
} from "../model";
import {
  clusterColor,
  dirTagColors,
  hslToRgb,
  keyColor,
  tempoColor,
} from "./colors";
import { trackTempoNorm } from "./lenses";
import { displayFolderPath } from "./path-presenter";
import { hotkeyBinding, hotkeyLabel } from "../hotkeys";
import {
  formatTimestamp,
  structureMarkers,
  type StructureMarker,
} from "./structure";

/** One stat row: a 0–1 bar (or a chip for categorical stats) plus its readout. */
export interface TooltipBar {
  label: string;
  value: number;
  readout: string;
  /** Categorical stat (key): render a chip, not a magnitude bar. */
  chip?: boolean;
  /** Fill for a categorical chip (the feature-lens colour); omitted = neutral. */
  color?: string;
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

/**
 * The preview seekbar: the auditioned track's timeline. ``time``/``duration`` are
 * seconds, so the fill and playhead show how far playback has run and the readout
 * pairs the current position with the total; ``markers`` place the cached structure
 * sections along the same timeline. ``playing`` is false when the track is not the one
 * currently auditioning, which renders the inactive state.
 */
export interface TooltipSeek {
  time: number;
  duration: number;
  playing: boolean;
  markers: StructureMarker[];
}

/** One zero-shot phrase CLAP ranked for the track (a phrase bank candidate). */
export interface DescribeResult {
  text: string;
  /** Bank facet the phrase belongs to (Genre / Mood / …). */
  facet: string;
  score: number;
}

/** One semantic coordinate: where the track sits on a contrastive axis. */
export interface CoordinateResult {
  /** Axis label, e.g. ``dark <-> bright``. */
  axis: string;
  pos: string;
  neg: string;
  /** Pole the track leans to (percentile ≥ 50 % → ``pos``). */
  pole: string;
  z: number;
  /** Reference-distribution percentile, 0–1. */
  percentile: number;
  /** Axis reliability (|Cohen's d| of the best genre contrast), used to weight it. */
  reliability: number;
}

/** Describe-tab state, loaded lazily when the card is pinned. */
export interface DescribeState {
  /** Track the results belong to, so a stale fetch never shows on another card. */
  path: string;
  busy: boolean;
  message: string | null;
  results: DescribeResult[];
  coordinates: CoordinateResult[];
}

/** An empty describe state for *path* (busy while the fetch is in flight). */
export function emptyDescribe(path: string, busy = false): DescribeState {
  return { path, busy, message: null, results: [], coordinates: [] };
}

/** Identity of the describe payload, so the panel rebuilds only when it changes. */
export function describeKey(describe: DescribeState): string {
  return [
    describe.path,
    describe.busy ? "busy" : "",
    describe.message ?? "",
    describe.results.map((r) => `${r.facet}:${r.text}:${r.score}`).join("|"),
    describe.coordinates
      .map((c) => `${c.axis}:${c.pole}:${c.percentile}`)
      .join("|"),
  ].join("\u0001");
}

/** The pinned card's two views: neighbour ranking, or zero-shot descriptions. */
export type InspectTab = "navigate" | "describe";

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
  /** Active tab: the neighbour ranking or the CLAP descriptions. */
  tab: InspectTab;
  /** Zero-shot descriptions, fetched when the Describe tab is first opened. */
  describe: DescribeState;
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
  /** Folder head's predicted destination, or ``null`` when it has no prediction. */
  projectedFolder: string | null;
  /** Icon colour stops for the projected folder (``[]`` when there is none). */
  projectedColors: string[];
  /**
   * One short word for the projected row when the head has no destination to show:
   * ``No head``, ``Not analysed``, or ``Loading…``. ``null`` when a projected
   * folder is shown. The sentence that explains it lives in ``projectedHint``,
   * shown only on hover.
   */
  projectedNote: string | null;
  /** Hover elaboration for the projected row (full path, confidence, or why not). */
  projectedHint: string | null;
  /** Bold primary line: the title, falling back to the filename. */
  title: string;
  /** Secondary artist line, or ``null`` when the track has no artist metadata. */
  artist: string | null;
  /** The preview seekbar: always rendered, active only while auditioning. */
  seek: TooltipSeek;
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
        color: row ? keyColor(row.key, row.mode) : undefined,
        target: { kind: "key", value: track.musical_key },
      });
      continue;
    }
    // Tempo is a BPM chip or nothing: the normalised 0–1 value is not a useful
    // standalone bar, so there is no bar fallback when BPM is unknown.
    if (k === "tempo") {
      if (track.bpm == null || !Number.isFinite(track.bpm)) continue;
      const tnorm = trackTempoNorm(track);
      rows.push({
        label: m.axisDisplayName(k),
        value: 0,
        readout: `${Math.round(track.bpm)} BPM`,
        chip: true,
        color: tnorm != null ? tempoColor(tnorm) : undefined,
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
  const auditioning = m.previewTrackPath === track.path;
  const activeSection = auditioning ? m.previewSection : null;
  // The timeline is always rendered: its length is the probed structure duration,
  // falling back to the audio element's own duration while the fetch is in flight — but
  // only for the track being auditioned, since a non-auditioned track has no length of
  // its own. Without a duration the seekbar renders as an empty, inactive track.
  const structureDuration = structure?.duration ?? 0;
  const timeline =
    structureDuration > 0 ? structureDuration : auditioning ? m.previewDuration : 0;
  const playing = auditioning && timeline > 0;
  const seek: TooltipSeek = {
    time: playing ? m.previewTime : 0,
    duration: timeline,
    playing,
    markers: structureMarkers(structure, timeline),
  };

  const cid = m.clusterAssignments.get(track.path);
  const spanning = structure?.genre_spanning === true;
  // The head's nearest learned folder, shown whenever the head has a prediction —
  // including one equal to the current folder, so the chip always names where the
  // head places the track rather than falling back to a bare "Same".
  const predicted = m.folderPredictions.get(track.path);
  const projected = predicted ?? null;
  const projectedFolder = projected !== null ? displayFolderPath(projected) : null;
  const projectedColors = projected !== null ? dirTagColors(projected) : [];
  const projectedConfident = m.folderPredictionMatches.has(track.path);
  // The row reads one short word only when the head has nothing to say; the
  // sentence that explains it — or the full destination path and confidence —
  // lives in the hover title, so the chip keeps its width for the value itself.
  let projectedNote: string | null = null;
  let projectedHint: string | null = null;
  if (projected !== null) {
    projectedHint =
      `${projectedFolder} — the head's nearest match` +
      (projectedConfident ? "" : " · low confidence");
  } else if (!m.folderPredictionsLoaded) {
    projectedNote = "Loading…";
    projectedHint = "Loading the head's predicted folders…";
  } else if (!m.folderPredictionsAvailable) {
    projectedNote = "No head";
    projectedHint = "No trained folder head yet.";
  } else {
    projectedNote = "Not analysed";
    projectedHint = "Not analysed yet — the head has no embedding for it.";
  }
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
  // Descriptions are fetched per track: if a response for another track lands
  // after the card was re-pinned, drop it rather than label this track with it.
  const effectiveInspect: InspectState | null =
    inspect && inspect.describe.path !== track.path
      ? { ...inspect, describe: emptyDescribe(track.path) }
      : inspect;
  const results = effectiveInspect
    ? effectiveInspect.results.map((r) => resolveResult(r, m))
    : [];

  const key = [
    track.path,
    track.filename,
    track.artist,
    track.title,
    track.folder ?? "",
    structure ? JSON.stringify(structure) : "",
    tags.map((t) => `${t.label}:${t.readout}`).join("|"),
    features.map((f) => `${f.label}:${f.readout}:${f.value}`).join("|"),
    activeSection ?? "",
    // The seekbar's length and play state gate the markers and active label, so
    // they rebuild the card; the live position deliberately does not (see
    // `updateSeekProgress`).
    `${seek.duration}:${seek.playing ? "play" : ""}`,
    cid ?? "",
    projectedFolder ?? "",
    projectedNote ?? "",
    projectedHint ?? "",
    m.clusterRevision ?? "",
    effectiveInspect ? inspectKey(effectiveInspect) : "",
  ].join("\u0001");

  return {
    key,
    path: track.path,
    folder,
    folderColors: dirTagColors(folderPath),
    folderTarget: { kind: "folder", prefix: track.folder ?? "" },
    projectedFolder,
    projectedColors,
    projectedNote,
    projectedHint,
    title,
    artist,
    seek,
    genre,
    tags,
    features,
    activeSection,
    inspect: effectiveInspect,
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
    inspect.tab,
    describeKey(inspect.describe),
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

/** Folder or tag glyph, stroked in `currentColor` so the chip sets the contrast. */
function chipIcon(kind: "folder" | "tag"): SVGSVGElement {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("class", `tt-chip-icon tt-chip-icon-${kind}`);
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
  const [r, g, b] = hslToRgb(h, s / 100, l / 100);
  const linear = (v: number): number =>
    v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
  return 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b);
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

/**
 * The one chip in the card. Every pill — the folder/cluster location tags, the
 * genre rows, the tempo/key stats, the projected-head state, and the structure
 * sections — is this, so they share one geometry, fill, and contrast rule. Only
 * *value* is required; *icon* and *label* are the optional visible parts, and
 * *title* carries any elaboration on hover so the chip itself stays short.
 */
interface ChipParts {
  /** The chip's main text. */
  value: string;
  /** Leading glyph, stroked in the chip's contrast colour. */
  icon?: "folder" | "tag";
  /** Small uppercase caption before the value (e.g. "Tempo"). */
  label?: string;
  /** Colour stops: a solid fill or a parent→own gradient; `[]` = neutral. */
  colors?: string[];
  /** Navigation target: glow/solo/fit, like a sidebar row. */
  target?: HoverTarget | null;
  /** Hover elaboration; the navigation hint is appended when the chip is a target. */
  title?: string | null;
}

function chip(parts: ChipParts): HTMLElement {
  const node = el("span", "tt-chip");
  const colors = parts.colors ?? [];
  const fill = tagFill(colors);
  if (fill) {
    node.classList.add("tt-chip-filled");
    node.style.background = fill;
    node.style.color = tagTextColor(colors);
  }
  if (parts.icon) node.appendChild(chipIcon(parts.icon));
  if (parts.label) node.appendChild(el("span", "tt-chip-label", parts.label));
  node.appendChild(el("span", "tt-chip-value", parts.value));
  if (parts.target) {
    // Keep the S / H hint discoverable; prepend the chip's own elaboration.
    const hint = parts.title ? `${parts.title} · ${TARGET_HINT}` : TARGET_HINT;
    markTarget(node, parts.target, hint);
  } else if (parts.title) {
    node.title = parts.title;
  }
  return node;
}

/** Folder location chip on the plain card: a folder glyph plus its name. */
function locationTag(
  kind: "folder" | "tag",
  label: string,
  colors: string[],
  target: HoverTarget,
  title?: string,
): HTMLElement {
  return chip({ icon: kind, value: label, colors, target, title });
}

/** Cluster chip on the plain card: a tag glyph, its colour, and its name. */
function clusterTag(genre: TooltipGenre): HTMLElement {
  return chip({
    icon: "tag",
    value: genre.name ?? "",
    colors: genre.color ? [genre.color] : [],
    target: genre.target,
    title: genre.spanning ? SPANNING_HINT : null,
  });
}

/** One genre row: a caption above a value chip (a nav target). */
function genreRow(
  label: string,
  value: string,
  colors: string[],
  target: HoverTarget | null,
  title: string,
  icon?: "folder" | "tag",
): HTMLElement {
  const row = el("div", "tt-genre-row");
  row.appendChild(el("div", "tt-genre-label", label));
  row.appendChild(chip({ icon, value, colors, target, title }));
  return row;
}

/**
 * The genres list, expanded inline in the pinned card: the track's Folder, the
 * folder head's Projected folder, and its Cluster — all styled the same, each a
 * caption above a chip. The folder and cluster chips are navigation targets, so the
 * S / H / F hotkeys act on them like a sidebar row. Anything that needs a
 * sentence (a weak projection, the head's reasoning, or the genre-spanning flag)
 * is only shown on hover, so the chips stay short.
 */
function genreList(card: TooltipCard): HTMLElement {
  const genre = card.genre;
  const folderPath =
    card.folderTarget.kind === "folder" ? card.folderTarget.prefix : "";
  const list = el("div", "tt-genres");
  list.appendChild(
    genreRow(
      "Folder",
      card.folder,
      card.folderColors,
      card.folderTarget,
      folderPath || "(library root)",
      "folder",
    ),
  );
  // A missing projected folder reads one short word ("No head", …); the
  // sentence that explains it — and any low-confidence note — is the hover title.
  list.appendChild(
    genreRow(
      "Projected folder",
      card.projectedFolder ?? card.projectedNote ?? "—",
      card.projectedColors,
      null,
      card.projectedHint ?? "",
      card.projectedFolder ? "folder" : undefined,
    ),
  );
  list.appendChild(
    genreRow(
      "Cluster",
      genre?.name ?? "—",
      genre?.color ? [genre.color] : [],
      genre?.target ?? null,
      genre?.spanning ? SPANNING_HINT : "",
      "tag",
    ),
  );
  return list;
}

function statSection(
  title: string,
  rows: TooltipBar[],
  extra?: string,
): HTMLElement {
  const section = el("div", `tt-section${extra ? " " + extra : ""}`);
  section.appendChild(el("div", "tt-section-title", title));
  let chipBox: HTMLElement | null = null;
  for (const r of rows) {
    if (r.chip) {
      // Categorical stat (key/BPM): the value has no magnitude to bar, so show a
      // chip in the same row style as the other stats. A feature-lens colour fills
      // it, so the card reads with the same palette as the map.
      chipBox ??= section.appendChild(el("div", "tt-stat-chips"));
      chipBox.appendChild(
        chip({
          label: r.label,
          value: r.readout,
          colors: r.color ? [r.color] : [],
          target: r.target,
        }),
      );
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

  if (card.inspect) {
    // Pinned card: the genres expand inline (Folder / Projected folder /
    // Cluster), so Alt-focusing the card shows the placement without a hover.
    root.appendChild(genreList(card));
  } else {
    const location = el("div", "tt-loc");
    const folderPath =
      card.folderTarget.kind === "folder" ? card.folderTarget.prefix : "";
    location.appendChild(
      locationTag(
        "folder",
        card.folder,
        card.folderColors,
        card.folderTarget,
        folderPath || "(library root)",
      ),
    );
    if (card.genre) location.appendChild(clusterTag(card.genre));
    root.appendChild(location);
  }

  root.appendChild(seekbar(card.seek, card.activeSection));

  if (card.tags.length) root.appendChild(statSection("Personal tags", card.tags));
  if (card.features.length)
    root.appendChild(statSection("Features", card.features, "tt-features"));

  if (card.inspect)
    root.appendChild(inspectSection(card.inspect, card.results));
  if (card.hint) root.appendChild(el("div", "tt-hint", card.hint));

  host.replaceChildren(root);
}

/** 0–100 % of *time* through *duration*, clamped and finite-safe. */
function progressPct(time: number, duration: number): number {
  if (!(duration > 0) || !Number.isFinite(time)) return 0;
  return Math.max(0, Math.min(100, (time / duration) * 100));
}

/**
 * The preview seekbar: a track filled to the current playback position, with a
 * playhead at the leading edge, a ``pos / total`` readout, the hotkey hint, and the
 * cached structure sections as markers along it. Only the auditioned section's marker
 * carries a label, so the timeline stays legible. When the track is not auditioning
 * the seekbar is rendered inactive — dimmed, with an empty fill and the playhead at
 * the start. Playback advances far faster than the card is rebuilt, so
 * `updateSeekProgress` moves the fill, playhead, and readout.
 */
function seekbar(seek: TooltipSeek, active: PreviewSection | null): HTMLElement {
  const box = el("div", `tt-seek${seek.playing ? "" : " tt-seek-inactive"}`);
  const track = el("div", "tt-seek-track");
  // The track is the seek surface: the card's delegated mousedown/drag reads
  // `data-seek` to seek the audition to the clicked/dragged fraction.
  track.dataset.seek = "";
  const pct = progressPct(seek.time, seek.duration);
  const fill = el("i", "tt-seek-fill");
  fill.style.width = `${pct}%`;
  track.appendChild(fill);
  for (const marker of seek.markers) {
    const node = el("span", "tt-seek-marker");
    node.dataset.section = marker.section;
    node.style.left = `${marker.fraction * 100}%`;
    if (marker.section === active) node.classList.add("active");
    else node.title = marker.text;
    track.appendChild(node);
    // The label is a track child, not a marker child, so its `%` is the track
    // width. Anchoring it on the marker's side keeps it inside the card at either
    // end without measuring it.
    if (marker.section === active) {
      const label = el("span", "tt-seek-label", marker.text);
      if (marker.fraction <= 0.5)
        label.style.left = `${marker.fraction * 100}%`;
      else label.style.right = `${(1 - marker.fraction) * 100}%`;
      track.appendChild(label);
    }
  }
  // The playhead is added last so it sits above the fill and the section markers.
  const head = el("i", "tt-seek-head");
  head.style.left = `${pct}%`;
  track.appendChild(head);
  box.appendChild(track);
  // The hotkey hint floats in the readout row's empty left space, so it never
  // reflows the card; the full descriptions live in its hover title.
  const hint = el("div", "tt-seek-hint", SEEK_HINT);
  hint.title = SEEK_HINT_TITLE;
  box.appendChild(hint);
  box.appendChild(
    el("div", "tt-seek-time", seekReadout(seek.time, seek.duration)),
  );
  return box;
}

/** Compact seekbar hotkey hint; the elaborations live in `SEEK_HINT_TITLE`. */
const SEEK_W = hotkeyLabel("cycle-preview-section");
const SEEK_W_PREV = hotkeyLabel("cycle-preview-section-prev");
const SEEK_M = hotkeyLabel("toggle-preview");
const SEEK_HINT = `${SEEK_W} seek · ${SEEK_M} mute · ${SEEK_M}+scroll volume`;
const SEEK_HINT_TITLE =
  `${SEEK_W} / ${SEEK_W_PREV} — ${hotkeyBinding("cycle-preview-section")!.description}\n` +
  `${SEEK_M} — ${hotkeyBinding("toggle-preview")!.description}\n` +
  `${SEEK_M} + scroll — adjust the preview volume`;

/** ``m:ss / m:ss`` for the seekbar; an unknown total renders as an em dash. */
function seekReadout(time: number, duration: number): string {
  return `${formatTimestamp(time)} / ${duration > 0 ? formatTimestamp(duration) : "—"}`;
}

/** Advance the rendered seekbar fill, playhead, and readout without rebuilding. */
export function updateSeekProgress(
  host: HTMLElement,
  time: number,
  duration: number,
): void {
  const pct = progressPct(time, duration);
  const fill = host.querySelector<HTMLElement>(".tt-seek-fill");
  if (fill) fill.style.width = `${pct}%`;
  const head = host.querySelector<HTMLElement>(".tt-seek-head");
  if (head) head.style.left = `${pct}%`;
  const readout = host.querySelector<HTMLElement>(".tt-seek-time");
  if (readout) readout.textContent = seekReadout(time, duration);
}

/** Pinned card: a Navigate / Describe tab strip plus the active tab's body. */
function inspectSection(
  inspect: InspectState,
  results: ResultRow[],
): HTMLElement {
  const section = el("div", "tt-section tt-inspect");
  const head = el("div", "tt-inspect-head");
  const tabs = el("div", "tt-tabs");
  const defs: [InspectTab, string][] = [
    ["navigate", "Navigate"],
    ["describe", "Describe"],
  ];
  for (const [tab, label] of defs) {
    const active = inspect.tab === tab;
    const btn = el(
      "button",
      `tt-tab${active ? " active" : ""}`,
      label,
    ) as HTMLButtonElement;
    btn.dataset.inspectAction = `show-${tab}`;
    btn.setAttribute("role", "tab");
    btn.setAttribute("aria-selected", String(active));
    tabs.appendChild(btn);
  }
  head.appendChild(tabs);
  const close = el("button", "tt-close", "×");
  close.dataset.inspectClose = "";
  close.title = "Close inspect (Esc)";
  head.appendChild(close);
  section.appendChild(head);

  // On Describe the content folds out into the wide panel beside the card, so the
  // card body stays empty there; Navigate keeps its controls in the card.
  if (inspect.tab !== "describe")
    section.appendChild(navigateBody(inspect, results));
  return section;
}

/** Navigate tab: the anti-track / nearest buttons, slider, and ranked rows. */
function navigateBody(inspect: InspectState, results: ResultRow[]): HTMLElement {
  const body = el("div", "tt-tab-body");
  // One toggle flips the distance filter's direction: nearest keeps the closest
  // tracks, anti-track keeps the furthest. The slider is the main affordance.
  const mode = inspect.mode === "far" ? "far" : "near";
  const toggle = el(
    "button",
    "tt-nav-btn",
    mode === "far" ? "Anti-track" : "Nearest",
  ) as HTMLButtonElement;
  toggle.dataset.inspectAction = "toggle-rank-mode";
  toggle.title =
    mode === "far"
      ? "Anti-track: the filter keeps the furthest tracks. Click for nearest."
      : "Nearest: the filter keeps the closest tracks. Click for anti-track.";
  toggle.disabled = inspect.busy;
  body.appendChild(toggle);

  // Distance-filter slider: 100 % = the whole library visible, narrower = keep
  // the nearest (or, in anti-track mode, the furthest) fraction of it. It is the
  // main affordance and does *not* re-home the view as it moves.
  const proximity = el("div", "tt-proximity");
  const proxHead = el("div", "tt-proximity-head");
  proxHead.appendChild(el("span", "tt-proximity-label", "Distance filter"));
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
  range.title =
    mode === "far"
      ? "Drag to keep only the furthest tracks (anti-track)"
      : "Drag to keep only the closest tracks";
  range.disabled = inspect.busy;
  proximity.appendChild(range);
  body.appendChild(proximity);

  if (inspect.busy) {
    body.appendChild(el("div", "tt-note", "Searching the embedding space…"));
  } else if (inspect.message) {
    body.appendChild(el("div", "tt-note tt-note-error", inspect.message));
  } else if (results.length) {
    body.appendChild(
      el(
        "div",
        "tt-section-title",
        inspect.mode === "far" ? "Anti-track" : "Nearest",
      ),
    );
    for (const r of results) body.appendChild(resultRow(r));
  }
  return body;
}

/**
 * The Describe panel: a wide two-column fold-out beside the pinned card.
 *
 * Left is the flat list of phrases CLAP ranks closest to the track; right is the
 * semantic coordinates. Both live here rather than in the 300 px card, so the term
 * list and the sliders get the width they need. There is no title: the tab already
 * says Describe. *host* is hidden unless the Describe tab is active and something
 * is loaded.
 */
export function renderDescribePanel(
  host: HTMLElement,
  describe: DescribeState,
  active: boolean,
): void {
  if (!active) {
    host.classList.add("hidden");
    return;
  }
  const panel = el("div", "tt-desc-panel");
  const terms = el("div", "tt-desc-col");
  const coords = el("div", "tt-coord-col");
  if (describe.busy) {
    terms.appendChild(el("div", "tt-note", "Asking CLAP…"));
  } else if (describe.message) {
    terms.appendChild(el("div", "tt-note tt-note-error", describe.message));
  } else if (!describe.results.length) {
    terms.appendChild(
      el("div", "tt-note", "No CLAP match for this track yet."),
    );
  } else {
    for (const r of describe.results) {
      const row = el("div", "tt-describe-row", r.text);
      row.dataset.searchTerm = r.text;
      row.title = "CLAP search for tracks like this";
      terms.appendChild(row);
    }
  }
  for (const c of describe.coordinates) coords.appendChild(coordinateRow(c));
  panel.appendChild(terms);
  panel.appendChild(coords);
  host.replaceChildren(panel);
  host.classList.remove("hidden");
}

/** One coordinate row: pos label, bipolar track with marker, neg label. */
function coordinateRow(c: CoordinateResult): HTMLElement {
  const pct = Math.round(c.percentile * 100);
  const leansPos = c.pole === c.pos;
  const row = el("div", "tt-coord");
  row.title =
    `${c.axis} · ${pct}% of the reference library is below this track ` +
    `(axis reliability ${c.reliability.toFixed(2)})`;
  const pos = el("span", `tt-coord-pole${leansPos ? " lean" : ""}`, c.pos);
  const neg = el("span", `tt-coord-pole${leansPos ? "" : " lean"}`, c.neg);
  const track = el("span", "tt-coord-track");
  const marker = el("i", "tt-coord-marker");
  marker.style.left = `${pct}%`;
  track.appendChild(marker);
  row.appendChild(pos);
  row.appendChild(track);
  row.appendChild(neg);
  return row;
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

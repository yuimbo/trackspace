// ─── Hover tooltip ("hockey card") ────────────────────────────
//
// The tooltip is a small card rather than a block of text: a header (folder +
// title), a BPM/key meta row, structure chips (the audition sections), the genre
// group (cluster + genre-spanning badge), and two stat-bar areas — user tags and
// audio features. `buildTooltip` is pure (unit-testable) and `renderTooltip` is the
// only DOM code, so the canvas view stays free of layout.

import {
  type Model,
  type PreviewSection,
  type Track,
  type TrackAudioFeatures,
  LIBROSA_FEATURE_AXIS_IDS,
  isAudioFeatureAxisId,
} from "../model";
import { clusterColor } from "./toast";
import { displayFolderPath } from "./path-presenter";
import { structureChips, type StructureChip } from "./structure";

/** One stat row: a 0–1 bar plus its human readout. */
export interface TooltipBar {
  label: string;
  value: number;
  readout: string;
}

export interface TooltipGenre {
  /** Cluster label, or ``null`` when the track has no cluster assignment. */
  name: string | null;
  /** Swatch colour, or ``null`` when there is no cluster. */
  color: string | null;
  spanning: boolean;
}

export interface TooltipCard {
  /** Identity of every rendered input; the canvas re-renders only when it changes. */
  key: string;
  folder: string;
  title: string;
  meta: string[];
  chips: StructureChip[];
  genre: TooltipGenre | null;
  tags: TooltipBar[];
  features: TooltipBar[];
  /** Section currently auditioned by the hover preview (highlighted chip). */
  activeSection: PreviewSection | null;
}

const SPANNING_HINT =
  "Intro and drop excerpts disagree (embedding dispersion ≥ 0.25) — the track straddles genres";

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
    rows.push({
      label: m.axisDisplayName(k),
      value: clamp01(v),
      readout: featureReadout(k, v, track),
    });
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
 * in tests; the canvas only renders the result.
 */
export function buildTooltip(track: Track, m: Model): TooltipCard {
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
        };

  const tags = tagRows(track);
  const features = audioFeatureRows(track, m);

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
  ].join("\u0001");

  return {
    key,
    folder,
    title,
    meta,
    chips,
    genre,
    tags,
    features,
    activeSection,
  };
}

/* ── DOM rendering ─────────────────────────────────────────── */

function el(tag: string, className?: string, text?: string): HTMLElement {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
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
    section.appendChild(row);
  }
  return section;
}

/** Render a built card into *host*, replacing any previous content. */
export function renderTooltip(host: HTMLElement, card: TooltipCard): void {
  const root = el("div", "tt-card");
  root.appendChild(el("div", "tt-folder", card.folder));
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
    root.appendChild(genre);
  }

  if (card.tags.length) root.appendChild(statSection("Tags", card.tags));
  if (card.features.length)
    root.appendChild(statSection("Features", card.features, "tt-features"));

  host.replaceChildren(root);
}

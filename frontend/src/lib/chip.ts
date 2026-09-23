// ─── Chip ────────────────────────────────────────────────────
//
// The one pill shared by the hover card and the view-modifier stack: an optional
// leading folder/tag glyph, an optional uppercase caption, a value, and an optional
// parent→own colour fill (a solid hue or a folder gradient). Any elaboration — a
// full path, a weak projection, a multi-folder list — belongs in the hover
// ``title`` so the pill itself stays short.

import { el } from "./dom";
import { hslToRgb } from "./colors";

const SVG_NS = "http://www.w3.org/2000/svg";
/** Feather-style folder / tag glyphs, stroked in the chip's contrast colour. */
const FOLDER_GLYPH = "M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z";
const TAG_GLYPH = "M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z";

export interface ChipParts {
  /** The chip's main text. */
  value: string;
  /** Leading glyph, stroked in the chip's contrast colour. */
  icon?: "folder" | "tag";
  /** Small uppercase caption before the value (e.g. "Tempo"). */
  label?: string;
  /** Colour stops: a solid fill or a parent→own gradient; `[]` = neutral. */
  colors?: string[];
  /** Hover elaboration, so the pill itself stays short. */
  title?: string | null;
}

/** Folder or tag glyph, stroked in `currentColor` so the chip sets the contrast. */
export function chipIcon(kind: "folder" | "tag"): SVGSVGElement {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("class", `chip-icon chip-icon-${kind}`);
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
export function tagFill(colors: string[]): string | null {
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
export function tagTextColor(colors: string[]): string {
  const luminances = colors.map((c) => {
    const m = /hsl\(\s*([\d.]+)\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%\s*\)/.exec(c);
    return m ? hslLuminance(+m[1], +m[2], +m[3]) : 0.5;
  });
  // 0.179 is where dark and light text have equal contrast against the fill.
  const avg = luminances.reduce((a, b) => a + b, 0) / luminances.length;
  return avg > 0.179 ? "#0d1219" : "#f4f7fb";
}

/** Build one chip. */
export function chip(parts: ChipParts): HTMLElement {
  const node = el("span", "chip");
  const colors = parts.colors ?? [];
  const fill = tagFill(colors);
  if (fill) {
    node.classList.add("chip-filled");
    node.style.background = fill;
    node.style.color = tagTextColor(colors);
  }
  if (parts.icon) node.appendChild(chipIcon(parts.icon));
  if (parts.label) node.appendChild(el("span", "chip-label", parts.label));
  node.appendChild(el("span", "chip-value", parts.value));
  if (parts.title) node.title = parts.title;
  return node;
}

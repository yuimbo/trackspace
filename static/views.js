"use strict";

// ─── Constants ───────────────────────────────────────────────
const PAD          = 40;
const DOT_R        = 4;
const DOT_R_SEL    = 6;
const HIT_RADIUS   = 10;
const CLUSTER_CELL = 16;
const TX_PAD       = 14;   // screen padding around selection bounding box
const TX_KNOB_HIT  = 9;    // hit radius for transform handles

// ─── Dir coloring ────────────────────────────────────────────
export function dirColor(folderPath) {
  if (!folderPath) return "hsl(350,60%,55%)";
  let h = 0;
  for (let i = 0; i < folderPath.length; i++)
    h = (h * 31 + folderPath.charCodeAt(i)) & 0x3ffff;
  return `hsl(${h % 360},65%,60%)`;
}

// ─── Toasts ──────────────────────────────────────────────────
let _$toasts;
export function toast(msg, type = "info") {
  _$toasts ??= document.getElementById("toast-container");
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  _$toasts.appendChild(el);
  setTimeout(() => { el.classList.add("out"); setTimeout(() => el.remove(), 400); }, 3200);
}

// ─── Loading overlay ─────────────────────────────────────────
let _lc = 0, _$loading;
export function showLoad() {
  _$loading ??= document.getElementById("loading-overlay");
  _lc++;
  _$loading.classList.remove("hidden");
}
export function hideLoad() {
  _$loading ??= document.getElementById("loading-overlay");
  if (--_lc <= 0) { _lc = 0; _$loading.classList.add("hidden"); }
}

// ═════════════════════════════════════════════════════════════
//  CanvasView
// ═════════════════════════════════════════════════════════════
export class CanvasView {
  constructor(model) {
    this.model   = model;
    this.$canvas = document.getElementById("viewport");
    this.ctx     = this.$canvas.getContext("2d");
    this.$tip    = document.getElementById("tooltip");

    this.positions          = [];      // [{idx, sx, sy}]  rebuilt every draw
    this.hoveredIdx         = -1;
    this.hoveredFolderPrefix = null;  // null | "" (root) | "rel/path" – set by TreeView hover
    this.tipReady    = false;
    this.tipTimer    = null;
    this.lassoPoints = [];
    this.boxSel      = null;   // {x0,y0,x1,y1} in screen coords while box-selecting
    this._rafId      = 0;

    // Transform box state (rebuilt each draw)
    this.txHandles = [];         // [{type, sx, sy, cursor}]
    this._txBox    = null;       // {sl, sr, st, sb} in screen coords
    this._txWorld  = null;       // {minX, maxX, minY, maxY} in world coords
  }

  /* ── Coordinate transforms ──────────────────────────────── */

  w2s(wx, wy) {
    const w = this.$canvas.clientWidth, h = this.$canvas.clientHeight;
    const iw = w - 2 * PAD, ih = h - 2 * PAD, vp = this.model.vp;
    return [
      PAD + (wx - vp.ox) * vp.zoom * iw,
      (h - PAD) - (wy - vp.oy) * vp.zoom * ih,
    ];
  }

  s2w(sx, sy) {
    const w = this.$canvas.clientWidth, h = this.$canvas.clientHeight;
    const iw = w - 2 * PAD, ih = h - 2 * PAD, vp = this.model.vp;
    return [
      (sx - PAD) / (vp.zoom * iw) + vp.ox,
      ((h - PAD) - sy) / (vp.zoom * ih) + vp.oy,
    ];
  }

  /* ── Sizing ─────────────────────────────────────────────── */

  resize() {
    const dpr = devicePixelRatio || 1;
    this.$canvas.width  = this.$canvas.clientWidth  * dpr;
    this.$canvas.height = this.$canvas.clientHeight * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  scheduleDraw() {
    cancelAnimationFrame(this._rafId);
    this._rafId = requestAnimationFrame(() => this.draw());
  }

  /* ── Viewport pan clamping ──────────────────────────────── */

  _clampVP() {
    const vp = this.model.vp;
    const w  = this.$canvas.clientWidth, h = this.$canvas.clientHeight;
    const iw = w - 2 * PAD, ih = h - 2 * PAD;
    if (iw <= 0 || ih <= 0) return;
    // Keep the screen centre inside [0, 1] in world space.
    // s2w(w/2, h/2) = [(w/2-PAD)/(zoom*iw) + ox, (h/2-PAD)/(zoom*ih) + oy]
    const cxOff = (w / 2 - PAD) / (vp.zoom * iw);
    const cyOff = (h / 2 - PAD) / (vp.zoom * ih);
    vp.ox = Math.max(-cxOff, Math.min(1 - cxOff, vp.ox));
    vp.oy = Math.max(-cyOff, Math.min(1 - cyOff, vp.oy));
  }

  /* ── Hit testing ────────────────────────────────────────── */

  hitTest(sx, sy) {
    let best = -1, bestD = HIT_RADIUS;
    for (const p of this.positions) {
      const d = Math.hypot(p.sx - sx, p.sy - sy);
      if (d < bestD) { bestD = d; best = p.idx; }
    }
    return best;
  }

  hitTestTransform(sx, sy) {
    // Corner / edge handles take priority
    for (const h of this.txHandles) {
      if (Math.abs(sx - h.sx) < TX_KNOB_HIT && Math.abs(sy - h.sy) < TX_KNOB_HIT)
        return h;
    }
    // Interior = move
    const b = this._txBox;
    if (b && sx > b.sl && sx < b.sr && sy > b.st && sy < b.sb)
      return { type: "move", cursor: "move" };
    return null;
  }

  /* ── Main draw ──────────────────────────────────────────── */

  draw() {
    this._clampVP();
    const m      = this.model;
    const tracks = m.tracks;
    const w = this.$canvas.clientWidth, h = this.$canvas.clientHeight;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, w, h);
    this.positions  = [];
    this.txHandles  = [];
    this._txBox     = null;
    this._txWorld   = null;

    if (tracks.length === 0) {
      ctx.fillStyle = "#444";
      ctx.font = "14px monospace";
      ctx.textAlign = "center";
      ctx.fillText("No tracks loaded", w / 2, h / 2);
      ctx.textAlign = "start";
      this._drawLasso();
      return;
    }

    this._drawGrid(w, h);
    this._drawAxes(w, h);
    this._drawScatter(tracks);
    this._drawFolderGlow(tracks);
    this._drawTransformBox();
    this._drawClusters();
    this._drawBoxSel();
    this._drawLasso();
    this._updateTooltip(tracks);
  }

  /* ── Grid + tick labels (adaptive to zoom) ──────────────── */

  _drawGrid(w, h) {
    const ctx = this.ctx;
    const m   = this.model;

    // Visible world range
    const [wxMin] = this.s2w(0, 0);
    const [wxMax] = this.s2w(w, 0);
    const [, wyMin] = this.s2w(0, h);
    const [, wyMax] = this.s2w(0, 0);

    // Screen coords of the [0,1]×[0,1] world boundary
    const [sLeft]      = this.w2s(0, 0);
    const [sRight]     = this.w2s(1, 0);
    const [, sBottom]  = this.w2s(0, 0);   // world y=0 → large screen y
    const [, sTop]     = this.w2s(0, 1);   // world y=1 → small screen y

    ctx.save();

    // ── Dark overlay outside [0,1] bounds ────────────────────
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    if (sLeft   > 0) ctx.fillRect(0,      0, sLeft,          h);
    if (sRight  < w) ctx.fillRect(sRight, 0, w - sRight,     h);
    if (sBottom < h) ctx.fillRect(sLeft,  sBottom, sRight - sLeft, h - sBottom);
    if (sTop    > 0) ctx.fillRect(sLeft,  0,       sRight - sLeft, sTop);

    // Thin border around the [0,1] area
    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.lineWidth = 1;
    ctx.strokeRect(sLeft, sTop, sRight - sLeft, sBottom - sTop);

    // Choose tick spacing: target ~60px between ticks
    const pickStep = (worldRange, pxRange) => {
      const ideal = (worldRange / pxRange) * 60;
      const steps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1];
      for (const s of steps) if (s >= ideal) return s;
      return 1;
    };

    const stepX = pickStep(wxMax - wxMin, w);
    const stepY = pickStep(wyMax - wyMin, h);
    const startX = Math.floor(wxMin / stepX) * stepX;
    const startY = Math.floor(wyMin / stepY) * stepY;

    const EPS = 1e-9;
    const inRange = v => v > -EPS && v < 1 + EPS;

    // Adaptive grid lines — confined to [0, 1]
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 1;
    for (let v = startX; v <= wxMax + stepX * 0.5; v += stepX) {
      if (!inRange(v)) continue;
      const [sx] = this.w2s(v, 0);
      ctx.beginPath(); ctx.moveTo(sx, sTop); ctx.lineTo(sx, sBottom); ctx.stroke();
    }
    for (let v = startY; v <= wyMax + stepY * 0.5; v += stepY) {
      if (!inRange(v)) continue;
      const [, sy] = this.w2s(0, v);
      ctx.beginPath(); ctx.moveTo(sLeft, sy); ctx.lineTo(sRight, sy); ctx.stroke();
    }

    // 0.1-step reference lines — always slightly thicker, confined to [0, 1]
    ctx.strokeStyle = "rgba(255,255,255,0.10)";
    ctx.lineWidth = 1.5;
    const REF = 0.1;
    const startXRef = Math.floor(Math.max(wxMin, 0) / REF) * REF;
    const startYRef = Math.floor(Math.max(wyMin, 0) / REF) * REF;
    for (let v = startXRef; v <= Math.min(wxMax, 1) + REF * 0.5; v += REF) {
      if (!inRange(v)) continue;
      const [sx] = this.w2s(v, 0);
      ctx.beginPath(); ctx.moveTo(sx, sTop); ctx.lineTo(sx, sBottom); ctx.stroke();
    }
    for (let v = startYRef; v <= Math.min(wyMax, 1) + REF * 0.5; v += REF) {
      if (!inRange(v)) continue;
      const [, sy] = this.w2s(0, v);
      ctx.beginPath(); ctx.moveTo(sLeft, sy); ctx.lineTo(sRight, sy); ctx.stroke();
    }

    // 0.5 origin lines — absolute centre reference, always drawn across the full [0,1] area
    ctx.strokeStyle = "rgba(255,255,255,0.22)";
    ctx.lineWidth = 1.5;
    const [sx05] = this.w2s(0.5, 0);
    ctx.beginPath(); ctx.moveTo(sx05, sTop); ctx.lineTo(sx05, sBottom); ctx.stroke();
    const [, sy05] = this.w2s(0, 0.5);
    ctx.beginPath(); ctx.moveTo(sLeft, sy05); ctx.lineTo(sRight, sy05); ctx.stroke();

    // Tick labels along bottom edge (X axis) — only within [0, 1]
    ctx.fillStyle = "rgba(255,255,255,0.3)";
    ctx.font = "9px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let v = startX; v <= wxMax + stepX * 0.5; v += stepX) {
      if (!inRange(v)) continue;
      const [sx] = this.w2s(v, 0);
      if (sx < 4 || sx > w - 4) continue;
      ctx.fillText(v.toFixed(stepX < 0.1 ? 2 : 1), sx, h - 14);
    }

    // Tick labels along left edge (Y axis) — only within [0, 1]
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let v = startY; v <= wyMax + stepY * 0.5; v += stepY) {
      if (!inRange(v)) continue;
      const [, sy] = this.w2s(0, v);
      if (sy < 4 || sy > h - 4) continue;
      ctx.fillText(v.toFixed(stepY < 0.1 ? 2 : 1), 32, sy);
    }

    // Axis names
    ctx.fillStyle = "rgba(255,255,255,0.25)";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    if (m.axisX) ctx.fillText(m.axisX, w / 2, h - 2);
    if (m.axisY) {
      ctx.save();
      ctx.translate(8, h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(m.axisY, 0, 0);
      ctx.restore();
    }

    ctx.restore();
  }

  _drawAxes() { /* merged into _drawGrid */ }

  /* ── Scatter plot (filter-aware) ────────────────────────── */

  _drawScatter(tracks) {
    const m = this.model;
    const tx = m.axisX, ty = m.axisY;
    let anyFiltered = false;
    tracks.forEach((t, i) => {
      if (!m.passesFilter(t)) { anyFiltered = true; return; }
      const wx = tx ? (t.tags[tx] ?? 0.5) : 0.5;
      const wy = ty ? (t.tags[ty] ?? 0.5) : 0.5;
      const [sx, sy] = this.w2s(wx, wy);
      this.positions.push({ idx: i, sx, sy });
      this._dot(sx, sy, t, i);
    });
    if (anyFiltered) {
      const ctx = this.ctx;
      const w = this.$canvas.clientWidth, h = this.$canvas.clientHeight;
      ctx.save();
      ctx.fillStyle = "rgba(255,170,50,0.55)";
      ctx.font = "10px monospace";
      ctx.textAlign = "right";
      ctx.fillText("filtered", w - 6, h - 6);
      ctx.restore();
    }
  }

  /* ── Folder-hover glow pass ────────────────────────────── */

  _inHoveredFolder(track) {
    const prefix = this.hoveredFolderPrefix;
    if (prefix === null) return false;
    const f = track.folder ?? "";
    if (prefix === "") return true;   // root node → all tracks
    return f === prefix || f.startsWith(prefix + "/");
  }

  _drawFolderGlow(tracks) {
    if (this.hoveredFolderPrefix === null) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.shadowBlur = 14;
    ctx.lineWidth  = 2;
    for (const pos of this.positions) {
      const track = tracks[pos.idx];
      if (!this._inHoveredFolder(track)) continue;
      const color = dirColor(track.folder ?? "");
      ctx.beginPath();
      ctx.arc(pos.sx, pos.sy, DOT_R + 5, 0, Math.PI * 2);
      ctx.strokeStyle  = color;
      ctx.shadowColor  = color;
      ctx.globalAlpha  = 0.75;
      ctx.stroke();
    }
    ctx.restore();
  }

  /* ── Single dot ─────────────────────────────────────────── */

  _dot(sx, sy, track, idx) {
    const sel = this.model.selected.has(track.path);
    const hov = this.hoveredIdx === idx;
    const r   = sel ? DOT_R_SEL : hov ? DOT_R + 1.5 : DOT_R;
    const ctx = this.ctx;

    let color;
    if (sel)      color = "#ffdd57";
    else if (hov) color = "#ff7eb3";
    else          color = dirColor(track.folder ?? "");

    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    if (sel) {
      ctx.strokeStyle = "#ffffffaa";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  /* ── Transform bounding box ─────────────────────────────── */

  _drawTransformBox() {
    const m = this.model;
    if (!m.selected.size) return;
    if (!m.axisX && !m.axisY) return;

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const path of m.selected) {
      const t = m.trackByPath(path);
      if (!t || !m.passesFilter(t)) continue;
      const wx = m.axisX ? (t.tags[m.axisX] ?? 0.5) : 0.5;
      const wy = m.axisY ? (t.tags[m.axisY] ?? 0.5) : 0.5;
      if (wx < minX) minX = wx; if (wx > maxX) maxX = wx;
      if (wy < minY) minY = wy; if (wy > maxY) maxY = wy;
    }
    if (!isFinite(minX)) return;

    this._txWorld = { minX, maxX, minY, maxY };

    // Screen coords of world corners
    // w2s: high world-Y → small screen-Y (top), low world-Y → large screen-Y (bottom)
    const [sx0] = this.w2s(minX, 0);
    const [sx1] = this.w2s(maxX, 0);
    const [, sy0] = this.w2s(0, minY);  // screen bottom (large sy)
    const [, sy1] = this.w2s(0, maxY);  // screen top   (small sy)

    const sl = sx0 - TX_PAD;
    const sr = sx1 + TX_PAD;
    const st = sy1 - TX_PAD;
    const sb = sy0 + TX_PAD;

    // Enforce minimum box size
    const cx = (sl + sr) / 2, cy = (st + sb) / 2;
    const bsl = cx - Math.max((sr - sl) / 2, 16);
    const bsr = cx + Math.max((sr - sl) / 2, 16);
    const bst = cy - Math.max((sb - st) / 2, 16);
    const bsb = cy + Math.max((sb - st) / 2, 16);

    this._txBox = { sl: bsl, sr: bsr, st: bst, sb: bsb };

    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = "rgba(255,221,87,0.35)";
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 4]);
    ctx.strokeRect(bsl, bst, bsr - bsl, bsb - bst);
    ctx.setLineDash([]);

    const hasRange = (maxX - minX) >= 0.005 || (maxY - minY) >= 0.005;
    const mx = (bsl + bsr) / 2, my = (bst + bsb) / 2;

    this.txHandles = [];
    if (hasRange) {
      if (m.axisX && m.axisY) {
        this.txHandles.push(
          { type: "nw", sx: bsl, sy: bst, cursor: "nw-resize" },
          { type: "ne", sx: bsr, sy: bst, cursor: "ne-resize" },
          { type: "sw", sx: bsl, sy: bsb, cursor: "sw-resize" },
          { type: "se", sx: bsr, sy: bsb, cursor: "se-resize" },
        );
      }
      if (m.axisY)
        this.txHandles.push(
          { type: "n", sx: mx, sy: bst, cursor: "n-resize" },
          { type: "s", sx: mx, sy: bsb, cursor: "s-resize" },
        );
      if (m.axisX)
        this.txHandles.push(
          { type: "e", sx: bsr, sy: my, cursor: "e-resize" },
          { type: "w", sx: bsl, sy: my, cursor: "w-resize" },
        );
    }

    // Draw handle squares
    for (const h of this.txHandles) {
      ctx.fillStyle = "rgba(255,221,87,0.9)";
      ctx.fillRect(h.sx - 4, h.sy - 4, 8, 8);
      ctx.strokeStyle = "rgba(0,0,0,0.55)";
      ctx.lineWidth = 1;
      ctx.strokeRect(h.sx - 4, h.sy - 4, 8, 8);
    }
    ctx.restore();
  }

  /* ── Cluster badges ─────────────────────────────────────── */

  _drawClusters() {
    if (this.positions.length < 2) return;
    const grid = new Map();
    for (const p of this.positions) {
      const key = `${Math.round(p.sx / CLUSTER_CELL)},${Math.round(p.sy / CLUSTER_CELL)}`;
      if (!grid.has(key)) grid.set(key, []);
      grid.get(key).push(p);
    }

    const ctx = this.ctx;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    for (const cluster of grid.values()) {
      if (cluster.length < 2) continue;
      const cx = cluster.reduce((s, p) => s + p.sx, 0) / cluster.length;
      const cy = cluster.reduce((s, p) => s + p.sy, 0) / cluster.length;
      const n  = cluster.length;
      const br = n > 9 ? 9 : 7;
      const bx = cx + DOT_R + 1;
      const by = cy - DOT_R - 1;

      ctx.beginPath();
      ctx.arc(bx, by, br, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(15,15,30,0.88)";
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.28)";
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = "#fff";
      ctx.font = `bold ${n > 9 ? 7 : 8}px monospace`;
      ctx.fillText(String(n), bx, by);
    }

    ctx.textAlign = "start";
    ctx.textBaseline = "alphabetic";
  }

  /* ── Box selection overlay ──────────────────────────────── */

  _drawBoxSel() {
    const bs = this.boxSel;
    if (!bs) return;
    const x = Math.min(bs.x0, bs.x1);
    const y = Math.min(bs.y0, bs.y1);
    const w = Math.abs(bs.x1 - bs.x0);
    const h = Math.abs(bs.y1 - bs.y0);
    if (w < 2 && h < 2) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = "rgba(255,221,87,0.7)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(255,221,87,0.06)";
    ctx.fillRect(x, y, w, h);
    ctx.restore();
  }

  /* ── Lasso overlay ──────────────────────────────────────── */

  _drawLasso() {
    const pts = this.lassoPoints;
    if (pts.length < 2) return;
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.closePath();
    ctx.strokeStyle = "rgba(255,221,87,0.7)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(255,221,87,0.06)";
    ctx.fill();
  }

  /* ── Tooltip ────────────────────────────────────────────── */

  _updateTooltip(tracks) {
    if (this.hoveredIdx < 0 || !this.tipReady) {
      this.$tip.classList.add("hidden");
      return;
    }
    const track = tracks[this.hoveredIdx];
    const pos   = this.positions.find(p => p.idx === this.hoveredIdx);
    if (!track || !pos) { this.$tip.classList.add("hidden"); return; }

    const tags   = Object.entries(track.tags).map(([k, v]) => `${k}: ${v.toFixed(2)}`).join("  ");
    const folder = track.folder ? `[${track.folder}]  ` : "";
    this.$tip.textContent = folder + track.filename + (tags ? "  ·  " + tags : "");
    this.$tip.classList.remove("hidden");

    const r  = this.$canvas.getBoundingClientRect();
    let tx   = r.left + pos.sx + 14;
    let ty   = r.top  + pos.sy - 10;
    if (tx + 280 > window.innerWidth) tx = r.left + pos.sx - 280;
    this.$tip.style.left = tx + "px";
    this.$tip.style.top  = ty + "px";
  }

  clearHover() {
    clearTimeout(this.tipTimer);
    this.tipReady   = false;
    this.hoveredIdx = -1;
    this.$tip.classList.add("hidden");
  }
}

// ═════════════════════════════════════════════════════════════
//  TreeView
// ═════════════════════════════════════════════════════════════
export class TreeView {
  constructor(model, container) {
    this.model = model;
    this.$el   = container;
    this.pendingRename = null;
    this._clickTimer   = null;

    this.onPickFolder      = null;
    this.onSelectFolder    = null;
    this.onCreateSubfolder = null;
    this.onRenameFolder    = null;
    this.onHoverFolder     = null;
    this.onHoverFolderEnd  = null;
  }

  render() {
    this.$el.innerHTML = "";
    if (this.model.folderTree)
      this._buildNode(this.model.folderTree, this.$el, 0);
  }

  _buildNode(node, parent, depth) {
    const hasKids = node.children?.length > 0;
    let expanded  = depth < 2;

    const row = document.createElement("div");
    row.className = "folder-row";
    row.style.paddingLeft = depth * 14 + "px";
    row.addEventListener("mouseenter", () => this.onHoverFolder?.(node.path));
    row.addEventListener("mouseleave", () => this.onHoverFolderEnd?.());

    const arrow = document.createElement("span");
    arrow.className = "folder-arrow";
    arrow.textContent = hasKids ? (expanded ? "▾" : "▸") : "\u2003";
    row.appendChild(arrow);

    const swatch = document.createElement("span");
    swatch.className = "folder-swatch";
    swatch.style.background = dirColor(node.path === "." ? "" : node.path);
    row.appendChild(swatch);

    const label = document.createElement("span");
    label.className = "folder-label";
    label.textContent = node.name;
    label.dataset.path = node.path;
    row.appendChild(label);

    const selBtn = document.createElement("span");
    selBtn.className = "folder-sel-btn";
    selBtn.title = "Select all tracks in this folder";
    selBtn.textContent = "◉";
    selBtn.addEventListener("click", e => {
      e.stopPropagation();
      this.onSelectFolder?.(node.path);
    });
    row.appendChild(selBtn);

    const addBtn = document.createElement("span");
    addBtn.className = "folder-add-btn";
    addBtn.title = "Create subfolder here";
    addBtn.textContent = "+";
    addBtn.addEventListener("click", e => {
      e.stopPropagation();
      this.onCreateSubfolder?.(node.path === "." ? "" : node.path);
    });
    row.appendChild(addBtn);

    if (node.path !== ".") {
      const renBtn = document.createElement("span");
      renBtn.className = "folder-rename-btn";
      renBtn.title = "Rename folder";
      renBtn.textContent = "✎";
      renBtn.addEventListener("click", e => {
        e.stopPropagation();
        this._startRename(row, label, node);
      });
      row.appendChild(renBtn);
    }

    parent.appendChild(row);

    let box = null;
    if (hasKids) {
      box = document.createElement("div");
      box.style.display = expanded ? "" : "none";
      parent.appendChild(box);
      node.children.forEach(c => this._buildNode(c, box, depth + 1));
    }

    label.addEventListener("click", e => {
      e.stopPropagation();
      clearTimeout(this._clickTimer);
      this._clickTimer = setTimeout(() => {
        this.$el.querySelectorAll(".folder-label.active").forEach(el => el.classList.remove("active"));
        label.classList.add("active");
        this.onPickFolder?.(node.path === "." ? "" : node.path);
      }, 240);
    });

    if (node.path !== ".") {
      label.addEventListener("dblclick", e => {
        e.stopPropagation();
        clearTimeout(this._clickTimer);
        this._startRename(row, label, node);
      });
    }

    if (hasKids) {
      arrow.style.cursor = "pointer";
      arrow.addEventListener("click", e => {
        e.stopPropagation();
        expanded = !expanded;
        arrow.textContent = expanded ? "▾" : "▸";
        box.style.display = expanded ? "" : "none";
      });
    }

    if (this.pendingRename === node.path) {
      this.pendingRename = null;
      requestAnimationFrame(() => this._startRename(row, label, node));
    }
  }

  _startRename(row, labelEl, node) {
    const input = document.createElement("input");
    input.type = "text";
    input.className = "folder-rename-input";
    input.value = node.name;

    let done = false;
    labelEl.replaceWith(input);
    input.select();
    input.focus();

    const commit = async () => {
      if (done) return;
      done = true;
      const newName = input.value.trim();
      if (!newName || newName === node.name) { input.replaceWith(labelEl); return; }
      const ok = await this.onRenameFolder?.(node, newName);
      if (!ok) { done = false; input.replaceWith(labelEl); }
    };
    const cancel = () => {
      if (done) return;
      done = true;
      input.replaceWith(labelEl);
    };
    input.addEventListener("keydown", e => {
      if (e.key === "Enter")  { e.preventDefault(); commit(); }
      if (e.key === "Escape") { e.preventDefault(); cancel(); }
    });
    input.addEventListener("blur", commit);
  }
}

// ═════════════════════════════════════════════════════════════
//  TagPanelView
// ═════════════════════════════════════════════════════════════
export class TagPanelView {
  constructor(model, container) {
    this.model = model;
    this.$el   = container;

    this.onAxisChange   = null;
    this.onFilterChange = null;
  }

  render() {
    this.$el.innerHTML = "";
    const m = this.model;
    for (const tag of m.tags) {
      const li = document.createElement("li");
      this._buildTagRow(li, tag);
      this.$el.appendChild(li);
    }
  }

  _buildTagRow(li, tag) {
    const m = this.model;
    if (!m.tagHasValuesInView(tag)) li.classList.add("tag-no-values");

    const name = document.createElement("span");
    name.className = "tag-name";
    name.textContent = tag;
    li.appendChild(name);

    [["X", "axisX"], ["Y", "axisY"]].forEach(([lbl, key]) => {
      const btn = document.createElement("button");
      btn.className = "axis-btn" + (m[key] === tag ? " active" : "");
      btn.textContent = lbl;
      btn.addEventListener("click", () => this.onAxisChange?.(key, tag));
      li.appendChild(btn);
    });

    const rng = m.filterRanges[tag] || [0, 1];
    const { slider, resetBtn } = this._makeDualRangeSlider(
      rng[0], rng[1],
      (range) => this.onFilterChange?.(tag, range),
      ()      => this.onFilterChange?.(tag, [0, 1])
    );
    li.appendChild(slider);
    li.appendChild(resetBtn);
  }

  _makeDualRangeSlider(lo, hi, onChange, onReset) {
    const wrap = document.createElement("div");
    wrap.className = "range-slider";

    const track = document.createElement("div");
    track.className = "range-track";
    wrap.appendChild(track);

    const fill = document.createElement("div");
    fill.className = "range-fill";
    track.appendChild(fill);

    const sLo = document.createElement("input");
    const sHi = document.createElement("input");
    for (const s of [sLo, sHi]) {
      Object.assign(s, { type: "range", min: "0", max: "1", step: "0.01" });
      s.className = "range-thumb";
      wrap.appendChild(s);
    }
    sLo.value = String(lo);
    sHi.value = String(hi);

    const sync = () => {
      const a = Math.min(+sLo.value, +sHi.value);
      const b = Math.max(+sLo.value, +sHi.value);
      fill.style.left  = (a * 100) + "%";
      fill.style.width = ((b - a) * 100) + "%";
      sLo.style.zIndex = +sLo.value > +sHi.value ? "3" : "2";
    };
    sync();

    const fire = () => onChange([
      Math.min(+sLo.value, +sHi.value),
      Math.max(+sLo.value, +sHi.value),
    ]);
    sLo.addEventListener("input", () => { sync(); fire(); });
    sHi.addEventListener("input", () => { sync(); fire(); });

    const resetBtn = document.createElement("button");
    resetBtn.className = "range-reset-btn";
    resetBtn.textContent = "↺";
    resetBtn.title = "Reset filter range";
    resetBtn.addEventListener("click", e => {
      e.stopPropagation();
      sLo.value = "0"; sHi.value = "1";
      sync(); onReset();
    });

    return { slider: wrap, resetBtn };
  }
}

// ═════════════════════════════════════════════════════════════
//  PropertiesView  – selection list with ghost-deselect
// ═════════════════════════════════════════════════════════════
export class PropertiesView {
  constructor(model, container) {
    this.model = model;
    this.$el   = container;
    // path → {timerId, track}
    this._pendingRestore = new Map();
    this._rangeAnchor    = null;   // path used as shift-select range start
    this._sortedPaths    = [];     // sorted order used for range calc
    this._listSel        = new Set(); // sub-selection within the list
    this._selHash        = "";

    this.onDeselect    = null;  // (path) =>
    this.onRestore     = null;  // (path) =>
    this.onFocusRange  = null;  // (Set<path>) => keep only these selected
    this.onHoverTrack  = null;  // (path) =>
    this.onHoverEnd    = null;  // () =>
  }

  /* ── Sort helpers ───────────────────────────────────────── */

  _sortKey(t) {
    if (!t) return "\x7f";
    const a  = (t.artist  || "").toLowerCase().trim();
    const ti = (t.title   || t.filename || "").toLowerCase().trim();
    return a ? `${a}\x00${ti}` : `\x7f${ti}`;
  }

  _sortedSelected() {
    const m = this.model;
    return [...m.selected]
      .map(p => m.trackByPath(p))
      .filter(Boolean)
      .sort((a, b) => this._sortKey(a) < this._sortKey(b) ? -1 : 1)
      .map(t => t.path);
  }

  /* ── Render ─────────────────────────────────────────────── */

  render() {
    this.$el.innerHTML = "";
    const m      = this.model;
    const sorted = this._sortedSelected();
    this._sortedPaths = sorted;

    // Reset list sub-selection when the viewport selection changes
    const hash = [...m.selected].sort().join("|");
    if (hash !== this._selHash) {
      this._selHash = hash;
      this._listSel.clear();
      this._rangeAnchor = null;
    }

    const ghosts = [...this._pendingRestore.entries()].filter(([p]) => !m.selected.has(p));

    if (sorted.length === 0 && ghosts.length === 0) {
      const empty = document.createElement("div");
      empty.className = "props-empty";
      empty.textContent = "No selection";
      this.$el.appendChild(empty);
      return;
    }

    // ── Active selection ────────────────────────────────────
    for (const path of sorted) {
      const track    = m.trackByPath(path);
      const isListSel = this._listSel.has(path);
      const row      = document.createElement("div");
      row.className  = "props-track-row" + (isListSel ? " list-sel" : "");

      // ── Info area (click/shift-click for list sub-selection) ──
      const info = document.createElement("div");
      info.className = "props-track-info";
      info.title = path;

      const primary = document.createElement("span");
      primary.className = "props-track-primary";
      primary.textContent = (track?.title || track?.filename) ?? path;
      info.appendChild(primary);

      if (track?.artist) {
        const secondary = document.createElement("span");
        secondary.className = "props-track-secondary";
        secondary.textContent = track.artist;
        info.appendChild(secondary);
      }

      info.addEventListener("mouseenter", () => this.onHoverTrack?.(path));
      info.addEventListener("mouseleave", () => this.onHoverEnd?.());

      info.addEventListener("click", e => {
        if (e.shiftKey && this._rangeAnchor && this._sortedPaths.length) {
          // Extend list sub-selection range — does NOT change m.selected
          const ai = this._sortedPaths.indexOf(this._rangeAnchor);
          const bi = this._sortedPaths.indexOf(path);
          if (ai >= 0 && bi >= 0) {
            const lo = Math.min(ai, bi), hi = Math.max(ai, bi);
            for (const p of this._sortedPaths.slice(lo, hi + 1)) this._listSel.add(p);
          }
        } else if (e.ctrlKey || e.metaKey) {
          // Toggle individual item in list sub-selection
          if (this._listSel.has(path)) this._listSel.delete(path);
          else { this._listSel.add(path); this._rangeAnchor = path; }
        } else {
          // Single click: select only this item
          this._listSel.clear();
          this._listSel.add(path);
          this._rangeAnchor = path;
        }
        this.render();
        e.stopPropagation();
      });

      row.appendChild(info);

      // ── Focus-select button (⊙) ──
      const focusBtn = document.createElement("button");
      focusBtn.className = "props-focus-btn";
      focusBtn.textContent = "⊙";
      focusBtn.title = "Focus select — keep only list-selected (or this track)";
      focusBtn.addEventListener("click", e => {
        e.stopPropagation();
        const targets = this._listSel.size > 0 ? new Set(this._listSel) : new Set([path]);
        this._rangeAnchor = path;
        this.onFocusRange?.(targets);
      });
      row.appendChild(focusBtn);

      // ── Deselect button (×) ──
      const delBtn = document.createElement("button");
      delBtn.className = "props-deselect-btn";
      delBtn.textContent = "×";
      delBtn.title = "Deselect (click ↩ ghost to restore)";
      delBtn.addEventListener("click", e => {
        e.stopPropagation();
        const t = m.trackByPath(path);
        const timerId = setTimeout(() => {
          this._pendingRestore.delete(path);
          this.render();
        }, 3000);
        this._pendingRestore.set(path, { timerId, track: t });
        if (this._rangeAnchor === path) this._rangeAnchor = null;
        this._listSel.delete(path);
        this.onDeselect?.(path);
      });
      row.appendChild(delBtn);

      this.$el.appendChild(row);
    }

    // ── Ghost entries (recently deselected, re-selectable) ──
    if (ghosts.length > 0) {
      const sep = document.createElement("div");
      sep.className = "props-sep";
      this.$el.appendChild(sep);

      for (const [path, { track }] of ghosts) {
        const row = document.createElement("div");
        row.className = "props-ghost-row";
        row.title = "Click to restore to selection";

        const icon = document.createElement("span");
        icon.className = "props-ghost-icon";
        icon.textContent = "↩";
        row.appendChild(icon);

        const info = document.createElement("div");
        info.className = "props-ghost-info";

        const primary = document.createElement("span");
        primary.className = "props-track-primary";
        primary.textContent = (track?.title || track?.filename) ?? path;
        info.appendChild(primary);

        if (track?.artist) {
          const secondary = document.createElement("span");
          secondary.className = "props-track-secondary";
          secondary.textContent = track.artist;
          info.appendChild(secondary);
        }

        row.appendChild(info);
        row.addEventListener("click", () => {
          const entry = this._pendingRestore.get(path);
          if (entry) { clearTimeout(entry.timerId); this._pendingRestore.delete(path); }
          this.onRestore?.(path);
        });
        this.$el.appendChild(row);
      }
    }
  }
}

// ═════════════════════════════════════════════════════════════
//  BatchView  – transform (scale/translate) sliders per tag
// ═════════════════════════════════════════════════════════════
export class BatchView {
  constructor(model, container) {
    this.model     = model;
    this.$el       = container;
    this._snapshots = new Map();   // tag → Map<path, origValue>
    this._selHash   = "";
    this._dragging  = false;

    this.onApply    = null;  // (tag, Map<path, newValue>) =>
    this.onDragEnd  = null;  // () =>
  }

  // Called on "tags-dirty" (in-flight viewport drag / txform): drop stale snapshots
  // so the sliders rebuild from the current in-memory tag values, then re-render.
  renderDirty() {
    if (this._dragging) return;
    this._snapshots.clear();
    this.render();
  }

  render() {
    if (this._dragging) return;

    const m    = this.model;
    const hash = [...m.selected].sort().join("|");

    if (hash !== this._selHash) {
      this._selHash = hash;
      this._snapshots.clear();
    }

    this.$el.innerHTML = "";

    if (!m.selected.size) return;

    if (!m.tags.length) return;

    const grid = document.createElement("div");
    grid.className = "batch-tag-grid";

    for (const tag of m.tags) {
      const snap    = this._getOrCreateSnapshot(tag);
      const vals    = [...snap.values()];
      const hasVals = vals.length > 0;
      const origLo  = hasVals ? Math.min(...vals) : 0.5;
      const origHi  = hasVals ? Math.max(...vals) : 0.5;
      const origAvg = hasVals ? vals.reduce((a, b) => a + b, 0) / vals.length : 0.5;
      const narrow  = !hasVals || (origHi - origLo) < 0.05;

      const label = document.createElement("span");
      label.className = "batch-tag-label" + (hasVals ? "" : " batch-tag-label--unset");
      label.textContent = tag;
      label.title = hasVals ? tag : tag + " (unset)";

      const slider = narrow
        ? this._makeSingleSlider(tag, origAvg, !hasVals)
        : this._makeTransformSlider(tag, origLo, origHi);

      grid.appendChild(label);
      grid.appendChild(slider);
    }

    this.$el.appendChild(grid);
  }

  /* ── Snapshot helpers ───────────────────────────────────── */

  _getOrCreateSnapshot(tag) {
    if (!this._snapshots.has(tag)) {
      const m = this.model;
      const snap = new Map();
      for (const path of m.selected) {
        const t = m.trackByPath(path);
        if (t && t.tags[tag] !== undefined) snap.set(path, t.tags[tag]);
      }
      this._snapshots.set(tag, snap);
    }
    return this._snapshots.get(tag);
  }

  _clearSnapshot(tag) { this._snapshots.delete(tag); }

  /* ── Single translate slider (narrow / uniform range) ───── */

  _makeSingleSlider(tag, initVal, isUnset = false) {
    const outer = document.createElement("div");
    outer.className = "txslider";

    const trackEl = document.createElement("div");
    trackEl.className = "txslider-track";
    outer.appendChild(trackEl);

    const knob = document.createElement("div");
    knob.className = "txslider-knob";
    if (isUnset) knob.style.opacity = "0.45";
    outer.appendChild(knob);

    let val = initVal;

    const update = () => {
      val = Math.max(0, Math.min(1, val));
      knob.style.left = `calc(${val * 100}% - 6px)`;
    };
    update();

    const fireTransform = () => {
      const snap    = this._getOrCreateSnapshot(tag);
      const updates = new Map();
      if (!snap.size) {
        // Tag unset on all selected tracks – assign val to every selected track
        for (const path of this.model.selected) updates.set(path, val);
      } else {
        const snapAvg = [...snap.values()].reduce((a, b) => a + b, 0) / snap.size;
        const delta   = val - snapAvg;
        for (const [path, origVal] of snap)
          updates.set(path, Math.max(0, Math.min(1, origVal + delta)));
      }
      this.onApply?.(tag, updates);
    };

    knob.addEventListener("pointerdown", e => {
      if (e.button !== 0) return;
      e.stopPropagation();
      e.preventDefault();
      knob.setPointerCapture(e.pointerId);
      this._dragging = true;

      const startX   = e.clientX;
      const startVal = val;
      const W        = outer.getBoundingClientRect().width || 100;

      knob.onpointermove = e => {
        val = Math.max(0, Math.min(1, startVal + (e.clientX - startX) / W));
        update();
        fireTransform();
      };
      knob.onpointerup = () => {
        knob.onpointermove = null;
        knob.onpointerup   = null;
        this._dragging = false;
        this._clearSnapshot(tag);
        this.onDragEnd?.();
      };
    });

    return outer;
  }

  /* ── Transform slider (dual-knob scale/translate) ───────── */

  _makeTransformSlider(tag, origLo, origHi) {
    const outer = document.createElement("div");
    outer.className = "txslider";

    const trackEl = document.createElement("div");
    trackEl.className = "txslider-track";
    outer.appendChild(trackEl);

    const fill = document.createElement("div");
    fill.className = "txslider-fill";
    trackEl.appendChild(fill);

    const loKnob = document.createElement("div");
    loKnob.className = "txslider-knob";
    const hiKnob = document.createElement("div");
    hiKnob.className = "txslider-knob";
    outer.appendChild(loKnob);
    outer.appendChild(hiKnob);

    // Slider always represents the full 0–1 range.
    // lo/hi knobs start at the actual data min/max of the selection.
    let lo = origLo, hi = origHi;

    const update = () => {
      lo = Math.max(0, Math.min(1, lo));
      hi = Math.max(0, Math.min(1, hi));
      if (hi < lo) hi = lo;
      fill.style.left   = (lo * 100) + "%";
      fill.style.right  = ((1 - hi) * 100) + "%";
      loKnob.style.left = `calc(${lo * 100}% - 6px)`;
      hiKnob.style.left = `calc(${hi * 100}% - 6px)`;
    };
    update();

    const fireTransform = () => {
      const snap = this._getOrCreateSnapshot(tag);
      const vals  = [...snap.values()];
      if (!vals.length) return;
      const sLo = Math.min(...vals);
      const sHi = Math.max(...vals);
      const range = sHi - sLo;
      const targetRange = hi - lo;

      const updates = new Map();
      for (const [path, origVal] of snap) {
        let nv = range < 0.0001
          ? (lo + hi) / 2
          : lo + (origVal - sLo) / range * targetRange;
        updates.set(path, Math.max(0, Math.min(1, nv)));
      }
      this.onApply?.(tag, updates);
    };

    const startDrag = (el, isKnob, isLo) => {
      el.addEventListener("pointerdown", e => {
        if (e.button !== 0) return;
        e.stopPropagation();
        e.preventDefault();
        el.setPointerCapture(e.pointerId);
        this._dragging = true;

        const startX   = e.clientX;
        const startLo  = lo, startHi = hi;
        const span     = hi - lo;
        let   W        = outer.getBoundingClientRect().width || 100;

        el.onpointermove = e => {
          const delta = (e.clientX - startX) / W;
          if (!isKnob) {
            lo = Math.max(0, Math.min(1 - span, startLo + delta));
            hi = lo + span;
          } else if (isLo) {
            lo = Math.max(0, Math.min(hi, startLo + delta));
          } else {
            hi = Math.max(lo, Math.min(1, startHi + delta));
          }
          update();
          fireTransform();
        };

        el.onpointerup = () => {
          el.onpointermove = null;
          el.onpointerup   = null;
          this._dragging = false;
          this._clearSnapshot(tag);
          this.onDragEnd?.();
        };
      });
    };

    fill.style.cursor = "ew-resize";
    startDrag(fill,   false, null);
    startDrag(loKnob, true,  true);
    startDrag(hiKnob, true,  false);

    return outer;
  }
}

// ═════════════════════════════════════════════════════════════
//  StatusView
// ═════════════════════════════════════════════════════════════
export class StatusView {
  constructor() {
    this.$el = document.getElementById("status-text");
  }

  update(model) {
    if (model.selected.size)
      this.$el.textContent = `${model.selected.size} selected  ·  ${model.tracks.length} tracks`;
    else
      this.$el.textContent = `${model.tracks.length} tracks in /${model.folder}`;
  }
}

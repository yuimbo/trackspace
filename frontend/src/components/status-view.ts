import type { Model } from "../model";

const PANEL_HIDE_MS = 140;

function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

export class StatusView {
  $el: HTMLElement;
  private _panel: HTMLElement;
  private _hideTimer = 0;
  private _boundPosition: () => void;
  private _model!: Model;
  private _hoverAnchor: HTMLElement | null = null;

  constructor() {
    this.$el = document.getElementById("status-text")!;
    this._panel = document.createElement("div");
    this._panel.id = "status-detail-panel";
    this._panel.className = "status-detail-panel";
    this._panel.setAttribute("role", "tooltip");
    this._panel.hidden = true;
    document.body.appendChild(this._panel);

    this._boundPosition = () => this._repositionPanel();

    window.addEventListener("resize", this._boundPosition);

    this.$el.addEventListener("mouseover", (e) => {
      const el = (e.target as HTMLElement).closest<HTMLElement>(
        ".status-hoverable",
      );
      if (!el || !this.$el.contains(el)) return;
      if (this._hoverAnchor === el) return;
      this._hoverAnchor = el;
      this._clearHideTimer();
      const kind = el.dataset.detailKind as "library" | "embed" | undefined;
      if (!kind) return;
      const html =
        kind === "library"
          ? this._libraryPanelHtml(this._model)
          : this._embedPanelHtml(this._model);
      this._showPanel(el, html, kind);
    });

    this.$el.addEventListener("mouseout", (e) => {
      const from = e.target as Node;
      const rel = e.relatedTarget as Node | null;
      const anchor = (from as HTMLElement).closest?.(".status-hoverable");
      if (!anchor || !this.$el.contains(anchor)) return;
      if (rel && (anchor === rel || anchor.contains(rel))) return;
      if (rel && this._panel.contains(rel)) return;
      this._hoverAnchor = null;
      this._scheduleHide();
    });

    this.$el.addEventListener("focusin", (e) => {
      const el = (e.target as HTMLElement).closest<HTMLElement>(
        ".status-hoverable",
      );
      if (!el || !this.$el.contains(el)) return;
      this._clearHideTimer();
      const kind = el.dataset.detailKind as "library" | "embed" | undefined;
      if (!kind) return;
      this._hoverAnchor = el;
      const html =
        kind === "library"
          ? this._libraryPanelHtml(this._model)
          : this._embedPanelHtml(this._model);
      this._showPanel(el, html, kind);
    });

    this.$el.addEventListener("focusout", (e) => {
      const el = (e.target as HTMLElement).closest<HTMLElement>(
        ".status-hoverable",
      );
      const rel = e.relatedTarget as Node | null;
      if (!el) return;
      if (rel && (el === rel || el.contains(rel))) return;
      if (rel && this._panel.contains(rel)) return;
      this._hoverAnchor = null;
      this._scheduleHide();
    });

    this._panel.addEventListener("mouseenter", () => this._clearHideTimer());
    this._panel.addEventListener("mouseleave", () => this._scheduleHide());

    document.addEventListener(
      "keydown",
      (e) => {
        if (e.key === "Escape" && !this._panel.hidden) {
          this._clearHideTimer();
          this._hoverAnchor = null;
          this._panel.hidden = true;
          delete this._panel.dataset.openKind;
        }
      },
      true,
    );
  }

  private _clearHideTimer(): void {
    if (this._hideTimer) {
      clearTimeout(this._hideTimer);
      this._hideTimer = 0;
    }
  }

  private _scheduleHide(): void {
    this._clearHideTimer();
    this._hideTimer = window.setTimeout(() => {
      this._hideTimer = 0;
      this._hoverAnchor = null;
      this._panel.hidden = true;
      delete this._panel.dataset.openKind;
      window.removeEventListener("scroll", this._boundPosition, true);
    }, PANEL_HIDE_MS);
  }

  private _repositionPanel(): void {
    const anchor = this._panel.dataset.anchorId
      ? document.getElementById(this._panel.dataset.anchorId)
      : null;
    if (!anchor || this._panel.hidden) return;
    const r = anchor.getBoundingClientRect();
    const pw = this._panel.offsetWidth;
    const ph = this._panel.offsetHeight;
    let left = r.left + r.width / 2 - pw / 2;
    left = Math.max(8, Math.min(left, window.innerWidth - pw - 8));
    const top = Math.max(8, r.top - ph - 8);
    this._panel.style.left = `${left}px`;
    this._panel.style.top = `${top}px`;
  }

  private _showPanel(
    anchor: HTMLElement,
    html: string,
    kind: string,
  ): void {
    this._clearHideTimer();
    const id = anchor.id || `status-hover-${kind}`;
    if (!anchor.id) anchor.id = id;
    this._panel.dataset.anchorId = id;
    this._panel.dataset.openKind = kind;
    this._panel.innerHTML = html;
    this._panel.hidden = false;
    window.addEventListener("scroll", this._boundPosition, true);
    requestAnimationFrame(() => this._repositionPanel());
  }

  private _libraryPanelHtml(model: Model): string {
    const p = model.libraryLoadProgress!;
    const pct =
      p.total > 0 ? Math.round((100 * p.done) / p.total) : 0;
    const last =
      p.lastPath || p.lastTitle
        ? `<div class="status-detail-kv"><span>Last file</span><span class="status-detail-mono">${esc(p.lastPath || p.lastTitle || "")}</span></div>${
            p.lastPath && p.lastTitle
              ? `<div class="status-detail-kv"><span>Title</span><span>${esc(p.lastTitle)}</span></div>`
              : ""
          }`
        : "";
    return `
<div class="status-detail-title">Library scan</div>
<div class="status-detail-kv"><span>Progress</span><span>${p.done} / ${p.total} (${pct}%)</span></div>
${last}
<div class="status-detail-note">Each track loads ID3 tags and a Chromaprint fingerprint. SQLite caches results by path and file modification time; cache hits avoid opening the MP3.</div>`;
  }

  private _embedPanelHtml(model: Model): string {
    const p = model.embeddingProgress!;
    const pct =
      p.total > 0 ? Math.round((100 * p.done) / p.total) : 0;
    const srcs = "CLAP, EffNet, audio features (tempo·energy·dance)";
    let last = "";
    if (p.lastPath) {
      const ok =
        p.lastOk === false
          ? '<span class="status-detail-bad">failed</span>'
          : '<span class="status-detail-ok">ok</span>';
      const fail =
        p.lastFailures?.length &&
        `<div class="status-detail-kv"><span>Missing layers</span><span>${esc(p.lastFailures.join(", "))}</span></div>`;
      last = `<div class="status-detail-kv"><span>Last track</span><span class="status-detail-mono">${esc(p.lastPath)}</span></div>
<div class="status-detail-kv"><span>Batch item</span><span>${ok}</span></div>${fail || ""}`;
    }
    return `
<div class="status-detail-title">Embedding generation</div>
<div class="status-detail-kv"><span>Progress</span><span>${p.done} / ${p.total} (${pct}%)</span></div>
<div class="status-detail-kv"><span>Active sources</span><span>${esc(srcs)}</span></div>
${last}
<div class="status-detail-note">The server runs CLAP per track and batches EffNet and classical features. The UI refreshes the projection periodically (PCA) while generation continues.</div>`;
  }

  update(model: Model): void {
    this._model = model;

    const reopenKind = this._panel.hidden ? "" : this._panel.dataset.openKind;

    this.$el.replaceChildren();

    const addSep = () => {
      const sep = document.createElement("span");
      sep.className = "status-sep";
      sep.textContent = "  ·  ";
      this.$el.appendChild(sep);
    };

    const base = document.createElement("span");
    base.className = "status-base";
    if (model.selected.size) {
      base.textContent = `${model.selected.size} selected  ·  ${model.tracks.length} tracks`;
    } else {
      base.textContent = `${model.tracks.length} tracks`;
    }
    this.$el.appendChild(base);

    if (model.libraryLoadProgress) {
      addSep();
      const a = document.createElement("span");
      a.className = "status-hoverable";
      a.dataset.detailKind = "library";
      a.tabIndex = 0;
      a.setAttribute("aria-describedby", "status-detail-panel");
      const lp = model.libraryLoadProgress;
      a.textContent = `loading ${lp.done}/${lp.total}`;
      this.$el.appendChild(a);
    }

    if (model.viewMode === "embeddings") {
      addSep();
      const mode = document.createElement("span");
      mode.textContent = "Embedding Space (t-SNE) — CLAP+EffNet+audio";
      this.$el.appendChild(mode);
    }

    if (model.embeddingsGenerating && model.embeddingProgress) {
      addSep();
      const a = document.createElement("span");
      a.className = "status-hoverable";
      a.dataset.detailKind = "embed";
      a.tabIndex = 0;
      a.setAttribute("aria-describedby", "status-detail-panel");
      const ep = model.embeddingProgress;
      a.textContent = `generating ${ep.done}/${ep.total}`;
      this.$el.appendChild(a);
    }

    if (reopenKind === "library" && model.libraryLoadProgress) {
      const el = this.$el.querySelector(
        '[data-detail-kind="library"]',
      ) as HTMLElement | null;
      if (el)
        this._showPanel(el, this._libraryPanelHtml(model), "library");
    } else if (
      reopenKind === "embed" &&
      model.embeddingsGenerating &&
      model.embeddingProgress
    ) {
      const el = this.$el.querySelector(
        '[data-detail-kind="embed"]',
      ) as HTMLElement | null;
      if (el) this._showPanel(el, this._embedPanelHtml(model), "embed");
    } else if (reopenKind) {
      this._panel.hidden = true;
      delete this._panel.dataset.openKind;
    }
  }
}

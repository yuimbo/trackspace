import type { Model, QueueStatus } from "../model";

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
      const kind = el.dataset.detailKind as
        | "library"
        | "embed"
        | "queue"
        | undefined;
      if (!kind) return;
      let html: string;
      if (kind === "library") html = this._libraryPanelHtml(this._model);
      else if (kind === "embed") html = this._embedPanelHtml(this._model);
      else html = this._queuePanelHtml(this._model);
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
      const kind = el.dataset.detailKind as
        | "library"
        | "embed"
        | "queue"
        | undefined;
      if (!kind) return;
      this._hoverAnchor = el;
      let html: string;
      if (kind === "library") html = this._libraryPanelHtml(this._model);
      else if (kind === "embed") html = this._embedPanelHtml(this._model);
      else html = this._queuePanelHtml(this._model);
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
    const srcs = "MAEST genre logits, MAEST embedding, rhythm, CLAP, EffNet";
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

  private _queuePanelHtml(model: Model): string {
    const q: QueueStatus | null = model.queueStatus;
    if (!q) {
      return `
<div class="status-detail-title">Analysis queue</div>
<div class="status-detail-note">No analysis queued.</div>`;
    }
    const rows: string[] = [];
    for (const kind of q.order) {
      const k = q.kinds[kind];
      if (!k || k.total === 0) continue;
      const pct = Math.round((100 * k.done) / k.total);
      const failed =
        k.failed > 0
          ? ` · <span class="status-detail-bad">${k.failed} failed</span>`
          : "";
      rows.push(
        `<div class="status-detail-kv"><span>${esc(k.label)}</span><span>${k.done} / ${k.total} (${pct}%)${failed}</span></div>`,
      );
    }
    let newest: { path: string; at: number } | null = null;
    for (const kind of q.order) {
      const last = q.kinds[kind]?.last;
      if (!last || !last.path) continue;
      if (!newest || last.at > newest.at) {
        newest = { path: last.path, at: last.at };
      }
    }
    const lastRow = newest
      ? `<div class="status-detail-kv"><span>Last</span><span class="status-detail-mono">${esc(newest.path)}</span></div>`
      : "";
    const maestError = q.models?.maest_error
      ? `<div class="status-detail-kv"><span>MAEST</span><span class="status-detail-bad">${esc(q.models.maest_error)}</span></div>`
      : "";
    return `
<div class="status-detail-title">Analysis queue</div>
${rows.join("\n")}
${lastRow}
${maestError}
<div class="status-detail-note">Analysis work is persisted in a SQLite queue, so progress survives a restart and interrupted jobs resume automatically. Failed jobs are retried a few times before being listed.</div>`;
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
      mode.textContent = "Embedding Space (t-SNE) — MAEST+rhythm";
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

    if (model.queueStatus && model.queueStatus.outstanding > 0) {
      const qs = model.queueStatus;
      let label = "";
      let finished = 0;
      let total = 0;
      for (const kind of qs.order) {
        const k = qs.kinds[kind];
        if (k && k.outstanding > 0) {
          label = k.label;
          finished = k.finished;
          total = k.total;
          break;
        }
      }
      if (label) {
        addSep();
        const a = document.createElement("span");
        a.className = "status-hoverable";
        a.dataset.detailKind = "queue";
        a.tabIndex = 0;
        a.setAttribute("aria-describedby", "status-detail-panel");
        a.textContent = `${label.charAt(0).toLowerCase()}${label.slice(1)} ${finished}/${total}`;
        this.$el.appendChild(a);
      }
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
    } else if (
      reopenKind === "queue" &&
      model.queueStatus &&
      model.queueStatus.outstanding > 0
    ) {
      const el = this.$el.querySelector(
        '[data-detail-kind="queue"]',
      ) as HTMLElement | null;
      if (el) this._showPanel(el, this._queuePanelHtml(model), "queue");
    } else if (reopenKind) {
      this._panel.hidden = true;
      delete this._panel.dataset.openKind;
    }
  }
}

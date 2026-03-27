import type { Model } from "../model";


export class StatusView {
  $el: HTMLElement;

  constructor() {
    this.$el = document.getElementById("status-text")!;
  }

  update(model: Model): void {
    const parts: string[] = [];
    if (model.selected.size) {
      parts.push(`${model.selected.size} selected  ·  ${model.tracks.length} tracks`);
    } else {
      parts.push(`${model.tracks.length} tracks`);
    }
    if (model.libraryLoadProgress) {
      const { done, total } = model.libraryLoadProgress;
      parts.push(`loading ${done}/${total}`);
    }
    if (model.viewMode === "embeddings") {
      const srcs = model.activeSources.map((s) =>
        s === "clap" ? "CLAP" : s === "effnet" ? "EffNet" : "Audio",
      ).join("+");
      parts.push(`Embedding Space (${model.projectionMethod.toUpperCase()}) [${srcs}]`);
      if (model.embeddingsGenerating && model.embeddingProgress) {
        parts.push(`generating ${model.embeddingProgress.done}/${model.embeddingProgress.total}`);
      }
    }
    this.$el.textContent = parts.join("  ·  ");
  }
}

import type { Model } from "../model";


export class StatusView {
  $el: HTMLElement;

  constructor() {
    this.$el = document.getElementById("status-text")!;
  }

  update(model: Model): void {
    if (model.selected.size)
      this.$el.textContent = `${model.selected.size} selected  ·  ${model.tracks.length} tracks`;
    else
      this.$el.textContent = `${model.tracks.length} tracks in /${model.folder}`;
  }
}

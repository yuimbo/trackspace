import "./trackspace.css";

import Alpine from "alpinejs";
import htmx from "htmx.org";

import { Model } from "./model";
import {
  CanvasView,
  TagPanelView,
  PropertiesView,
  BatchView,
  StatusView,
  SearchView,
} from "./components";
import { Controller } from "./controller";
import { renderShortcutList } from "./hotkeys";

window.htmx = htmx;

Alpine.start();

const model = new Model();
model.loadLS();

const canvas = new CanvasView(model);
const tagPanel = new TagPanelView(model, document.getElementById("tag-list")!);
const featuresPanel = new TagPanelView(
  model,
  document.getElementById("feature-list")!,
  "features",
);
const props = new PropertiesView(
  model,
  document.getElementById("properties-list")!,
);
const batch = new BatchView(model, document.getElementById("batch-list")!);
const status = new StatusView();
status.update(model);
const search = new SearchView(
  model,
  document.getElementById("search-section")!,
);

const ctrl = new Controller(
  model,
  canvas,
  tagPanel,
  featuresPanel,
  props,
  batch,
  status,
  search,
);
void ctrl.init();

renderShortcutList(document.getElementById("shortcut-list")!);

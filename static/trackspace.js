"use strict";

import { Model }                                                                        from "./model.js";
import { CanvasView, TreeView, TagPanelView, PropertiesView, BatchView, StatusView }    from "./views.js";
import { Controller }                                                                   from "./controller.js";

const model = new Model();
model.loadLS();

const canvas   = new CanvasView(model);
const tree     = new TreeView(model, document.getElementById("folder-tree"));
const tagPanel = new TagPanelView(model, document.getElementById("tag-list"));
const props    = new PropertiesView(model, document.getElementById("properties-list"));
const batch    = new BatchView(model, document.getElementById("batch-list"));
const status   = new StatusView();

const ctrl = new Controller(model, canvas, tree, tagPanel, props, batch, status);
ctrl.init();

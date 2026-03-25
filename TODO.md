# Trackspace – Implementation Plan

## Phase 1: Core Backend
- [x] **tags.py** – ID3 private tag read/write
  - [x] Read all `TXXX:trackspace:*` frames from an mp3
  - [x] Write/update a single tag value (float 0-1) into `TXXX:trackspace:<tagname>`
  - [x] Delete a tag from a file
  - [x] Rename a tag on a file
  - [x] Batch-read: given a list of paths, return `{path: {tag: value, …}, …}`
- [x] **app.py** – Flask routes
  - [x] `GET /api/folders?root=<path>` – list subfolders (tree)
  - [x] `GET /api/tracks?folder=<path>&recursive=<bool>` – list tracks with tag data
  - [x] `GET /api/tags` – list all known tag names across loaded tracks
  - [x] `POST /api/tags/rename` – rename a tag across all files
  - [x] `POST /api/tags/delete` – delete a tag from all files
  - [x] `POST /api/tracks/tags` – batch-update tag values for selected tracks
  - [x] `GET /api/audio/<path>` – stream an mp3 for hover-preview

## Phase 2: Frontend – Layout & Navigation
- [x] **index.html** – page shell, layout containers, overlays
- [x] **Sidebar: Folder tree** – collapsible folder browser, click to filter
- [x] **Sidebar: Tag list** – show all tags, mode-dependent controls
- [x] **Main viewport** – canvas for 2D scatter / grid of tracks

## Phase 3: Frontend – Viewport & Interaction
- [x] **Viewport rendering** – draw tracks as dots positioned by two selected tags (X/Y)
- [x] **Axis selector** – assign any tag to X or Y from the tag list (Viewtool A)
- [x] **Filter sliders** – per-tag range sliders for filtering (Viewtool B, mutually exclusive with A)
- [x] **Pan & zoom** – alt+drag to pan, scroll to zoom (centered on cursor)
- [x] **Lasso select** – shift+drag freeform polygon to select tracks
- [x] **Drag-to-transform** – move selected tracks in viewport → update their tag values
- [x] **Hover preview** – on hover, play from ~40% of the track (toggleable via checkbox / H key)
- [x] **Hit testing** – find nearest track to cursor for selection & hover
- [x] **Keyboard shortcuts** – H (preview), 0 (reset view), Esc (deselect), ⌘A (select all)

## Phase 4: Polish & Edge Cases
- [x] Debounce / throttle tag writes (500ms batched flush)
- [x] Handle files with no tags gracefully (default to 0.5 on each axis)
- [x] Persist UI state (folder, axes, viewport, view mode, filters) in localStorage
- [x] Error toasts for failed API calls
- [x] Loading spinner overlay for folder/track loading
- [x] Tooltip on hover showing filename + tag values

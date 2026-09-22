# Trackspace

Organise music in a 2D tag space. Tags are stored as private ID3 frames (`TXXX:trackspace:*`) directly in your mp3 files.

## Setup

```bash
cd tools/trackspace

# Python deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Frontend deps
cd frontend && npm install && cd ..
```

The UI is **Vite + TypeScript** with **Alpine.js** (e.g. shortcuts modal) and **HTMX** (folder tree HTML from Flask `GET /partials/folder-tree`).

## Dev mode (hot-reload)

```bash
cd frontend
npm run dev:all
```

Opens Vite at [http://localhost:5173](http://localhost:5173). `/api` and `/partials` are proxied to Flask on port 5111.

## Production

```bash
# Build frontend
cd frontend && npm run build && cd ..

# Run server
source .venv/bin/activate
python -m backend.app
```

**`python -m backend.app` serves the prebuilt `frontend/dist`.** A change to any
`frontend/src` file is not picked up until `npm run build` runs again — including a
change to the shape of a wire payload the backend emits, which will break the served
bundle until it is rebuilt. For frontend work use `npm run dev:all` above, which serves
the source directly.

Then open [http://127.0.0.1:5111](http://127.0.0.1:5111).

**Options:**

```
python -m backend.app /path/to/music # optional initial music root
python -m backend.app -p 8080        # custom port
python -m backend.app --host 0.0.0.0 # expose on LAN
```

## Docs

- `AGENTS.md` — entry point and core invariants (start here for contributing).
- `architecture.md` — system structure, frontend MVC, backend packages, HTTP API.
- `ui_conventions.md` — frontend interaction rules (canvas, tags, folders, hotkeys).
- `audio_pipeline.md` — fingerprinting → embeddings → projection → clustering, and the job queue.
- `TODO.md` — deferred work and the product roadmap.

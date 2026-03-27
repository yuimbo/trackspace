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
python app.py /path/to/music
```

Then open [http://127.0.0.1:5111](http://127.0.0.1:5111).

**Options:**

```
python app.py -p 8080        # custom port
python app.py --host 0.0.0.0 # expose on LAN
```

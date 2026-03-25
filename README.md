# Trackspace

Organise music in a 2D tag space. Tags are stored as private ID3 frames (`TXXX:trackspace:*`) directly in your mp3 files.

## Setup

```bash
cd tools/trackspace
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Launch

```bash
source .venv/bin/activate
python app.py /path/to/music
```

Then open [http://127.0.0.1:5111](http://127.0.0.1:5111).

**Options:**

```
python app.py /path/to/music -p 8080        # custom port
python app.py /path/to/music --host 0.0.0.0 # expose on LAN
```

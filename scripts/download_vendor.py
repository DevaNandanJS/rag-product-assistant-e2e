"""Download pinned vendor dependencies (marked.min.js and purify.min.js)."""
import urllib.request
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parent.parent / "frontend" / "vendor"
VENDOR_DIR.mkdir(parents=True, exist_ok=True)

LIBS = {
    "marked.min.js": "https://cdn.jsdelivr.net/npm/marked@15.0.12/marked.min.js",
    "purify.min.js": "https://cdn.jsdelivr.net/npm/dompurify@3.2.6/dist/purify.min.js",
}

def download():
    for name, url in LIBS.items():
        dest = VENDOR_DIR / name
        print(f"Downloading {name} from {url}...")
        urllib.request.urlretrieve(url, dest)
        size = dest.stat().st_size
        print(f"  -> Saved {dest} ({size} bytes)")

if __name__ == "__main__":
    download()

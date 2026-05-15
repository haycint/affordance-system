"""
Launcher for the human-operator monitoring frontend (multi-page HTML).

Usage:
    python monitor_launcher.py --port 9000              # demo=false
    python monitor_launcher.py --port 9000 --demo true  # demo=true

* Serves the static files under frontend/monitor/.
* On first run, downloads three.js + OrbitControls into vendor/ so the
  pages work without internet access. Override the source with --three-src
  (e.g. point to an internal mirror).
* /config.js exposes window.DEMO_MODE.
"""

import argparse
import os
import sys
import urllib.request
from http.server import HTTPServer, SimpleHTTPRequestHandler

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor")
VENDOR = os.path.join(ROOT, "vendor")

DEFAULT_CDN = "https://unpkg.com/three@0.160.0"
VENDOR_FILES = [
    ("build/three.module.js",                    "three.module.js"),
    ("examples/jsm/controls/OrbitControls.js",   "OrbitControls.js"),
]


def ensure_vendor(base_url: str):
    os.makedirs(VENDOR, exist_ok=True)
    for rel, local in VENDOR_FILES:
        out = os.path.join(VENDOR, local)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            continue
        url = f"{base_url}/{rel}"
        print(f"[monitor] downloading {url} → {out}")
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
            # rewrite imports inside OrbitControls so they resolve to the
            # local three.module.js without an import-map
            if local == "OrbitControls.js":
                data = data.replace(
                    b"from 'three'", b"from './three.module.js'")
                data = data.replace(
                    b'from "three"', b'from "./three.module.js"')
            with open(out, "wb") as f:
                f.write(data)
        except Exception as e:
            print(f"[monitor] WARNING: failed to fetch {url}: {e}", file=sys.stderr)
            print(f"[monitor]   place the file at {out} manually and re-run.",
                  file=sys.stderr)


def make_handler(demo: bool):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=ROOT, **kw)

        def log_message(self, fmt, *args):
            sys.stderr.write("[monitor] " + (fmt % args) + "\n")

        def do_GET(self):
            if self.path == "/config.js":
                body = f"window.DEMO_MODE = {str(bool(demo)).lower()};\n"
                self.send_response(200)
                self.send_header("Content-Type", "application/javascript")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body.encode())
                return
            if self.path in ("/", ""):
                self.path = "/index.html"
            return super().do_GET()
    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--demo", default="false",
                   help="'true' / 'false' — enables the GT comparison panel.")
    p.add_argument("--three-src", default=DEFAULT_CDN,
                   help="Base URL to fetch three.js from. Override with an "
                        "internal mirror if unpkg is unreachable.")
    p.add_argument("--skip-vendor", action="store_true",
                   help="Skip the vendor download step.")
    args = p.parse_args()
    demo = str(args.demo).lower() in ("1", "true", "yes", "on")

    if not args.skip_vendor:
        ensure_vendor(args.three_src.rstrip("/"))

    print(f"[monitor] serving {ROOT} on http://{args.host}:{args.port} "
          f"(demo={demo})")
    HTTPServer((args.host, args.port), make_handler(demo)).serve_forever()


if __name__ == "__main__":
    main()

"""
Minimal HTTP server exposing the tumor-marker classifier as a JSON API
and serving a single-page HTML form, using only the Python standard
library.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

from cancer_dx.model import ALL_MARKERS, predict_class

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"


class Handler(BaseHTTPRequestHandler):
    server_version = "cancer-dx/0.1"

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming convention)
        if self.path in ("/", "/index.html"):
            html = INDEX_HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        else:
            self._send_json(404, {"detail": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/predict":
            self._send_json(404, {"detail": "Not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length) if length else b"{}"

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            self._send_json(400, {"detail": "Request body must be valid JSON"})
            return

        try:
            markers = {m: float(payload[m]) for m in ALL_MARKERS}
        except (KeyError, TypeError, ValueError):
            self._send_json(
                400,
                {"detail": f"Request must include numeric values for: {', '.join(ALL_MARKERS)}"},
            )
            return

        name = payload.get("name") or "Patient"

        try:
            predicted, ranked = predict_class(markers)
        except ValueError as exc:
            self._send_json(400, {"detail": str(exc)})
            return

        self._send_json(
            200,
            {
                "name": name,
                "predicted_class": predicted,
                "ranked": [{"cancer_class": c, "probability": p} for c, p in ranked],
            },
        )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Keep server logs terse and readable in a terminal demo.
        print(f"[cancer-dx] {self.address_string()} - {format % args}")


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"cancer-dx running at http://{host}:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
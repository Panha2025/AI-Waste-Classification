"""Minimal public readiness endpoint; never returns secret values."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler


def health_payload():
    # Liveness only: don't load TensorFlow or expose deployment configuration.
    return {"status": "ok"}


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = health_payload()
        status = 200
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        return

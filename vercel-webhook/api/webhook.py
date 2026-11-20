"""Vercel Python function endpoint for Telegram's HTTPS webhook."""
from __future__ import annotations

import asyncio
import json
from http.server import BaseHTTPRequestHandler

from api._logic import MAX_WEBHOOK_BODY_BYTES, process_webhook


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = self.headers.get("Content-Length", "0")
        try:
            body_size = int(length)
        except ValueError:
            body_size = MAX_WEBHOOK_BODY_BYTES + 1
        if body_size <= 0 or body_size > MAX_WEBHOOK_BODY_BYTES:
            return self._json(413, {"ok": False, "error": "payload_too_large"})
        try:
            payload = json.loads(self.rfile.read(body_size))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._json(400, {"ok": False, "error": "invalid_json"})
        status, response = asyncio.run(process_webhook(self.headers, payload))
        return self._json(status, response)

    def do_GET(self):
        return self._json(405, {"ok": False, "error": "method_not_allowed"})

    def _json(self, status: int, payload: dict):
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        # Do not log request data or Bot API URLs.
        return

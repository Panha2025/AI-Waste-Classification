#!/usr/bin/env python3
"""Set or inspect the Telegram webhook without printing the bot token."""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from urllib.parse import urlsplit

import httpx

logging.getLogger("httpx").setLevel(logging.WARNING)


def _credentials():
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not configured in this shell.")
    return token, secret


def _request(token: str, method: str, payload: dict | None = None) -> dict:
    try:
        response = httpx.post(
            f"https://api.telegram.org/bot{token}/{method}",
            json=payload,
            timeout=15.0,
        )
        response.raise_for_status()
        result = response.json()
        if not result.get("ok"):
            raise RuntimeError
        return result.get("result") or {}
    except Exception:
        # HTTP exceptions can include the request URL, which contains the token.
        raise SystemExit(f"Telegram {method} request failed; details were suppressed for safety.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    set_parser = subparsers.add_parser("set", help="register a deployed HTTPS webhook")
    set_parser.add_argument("url", help="public HTTPS URL ending in /api/webhook")
    subparsers.add_parser("info", help="inspect current webhook status")
    args = parser.parse_args()
    token, secret = _credentials()

    if args.action == "set":
        parsed = urlsplit(args.url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or not parsed.path.rstrip("/").endswith("/api/webhook")
        ):
            raise SystemExit("Use a public HTTPS URL ending in /api/webhook.")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", secret):
            raise SystemExit("TELEGRAM_WEBHOOK_SECRET must be 1–256 letters, digits, underscores, or hyphens.")
        _request(token, "setWebhook", {
            "url": args.url,
            "secret_token": secret,
            "allowed_updates": ["message"],
        })
        print("Telegram webhook registered. The token and secret were not displayed.")
    else:
        info = _request(token, "getWebhookInfo")
        print(f"Webhook configured: {bool(info.get('url'))}")
        if info.get("url"):
            print(f"Webhook URL: {info['url']}")
        print(f"Pending updates: {int(info.get('pending_update_count', 0))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

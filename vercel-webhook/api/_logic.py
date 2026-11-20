"""Telegram webhook logic shared by the two Vercel Python functions."""
from __future__ import annotations

import asyncio
import hmac
import os
from html import escape
from pathlib import Path
from threading import Lock
from typing import Mapping

import httpx

from waste_ai.config import MAX_IMAGE_BYTES
from waste_ai.inference import WasteClassifier, format_result

TELEGRAM_HEADER = "x-telegram-bot-api-secret-token"
LOW_CONFIDENCE_THRESHOLD = 0.70
MAX_WEBHOOK_BODY_BYTES = 1_000_000
DEDUP_TTL_SECONDS = 86_400
_classifier: WasteClassifier | None = None
_classifier_lock = Lock()


def _get_classifier() -> WasteClassifier:
    global _classifier
    if _classifier is None:
        with _classifier_lock:
            if _classifier is None:
                _classifier = WasteClassifier()
    return _classifier


def _environment(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def _reply_for_text(text: str) -> str:
    command = text.strip().split(maxsplit=1)[0].split("@", 1)[0].lower() if text.strip() else ""
    if command == "/start":
        return (
            "<b>👋 Welcome to Waste Identifier!</b>\n"
            "Get a quick AI prediction for a waste item:\n\n"
            "1. Take a photo of one waste item.\n"
            "2. Send it here.\n"
            "3. Receive a prediction and a short suggestion.\n\n"
            "Please check local guidelines before disposing of an item."
        )
    if command == "/help":
        return (
            "<b>📷 How to use Waste Identifier</b>\n"
            "1. Take a clear photo of one waste item.\n"
            "2. Send the photo to the bot.\n"
            "3. Read the prediction, confidence, and suggestion.\n\n"
            "Please send a photo of one item. AI predictions can be wrong, and local recycling rules vary."
        )
    if command == "/about":
        return (
            "ITM 360 — binary waste classification research prototype.\n"
            "The AI predicts Recyclable or Non-Recyclable directly from the image.\n"
            "Training labels come from source folders and are provisional, not human-verified.\n"
            "Predictions are estimates; local recycling acceptance varies."
        )
    return "📷 Please send a photo of a waste item so I can classify it."


async def _telegram_call(client: httpx.AsyncClient, token: str, method: str, data: dict):
    response = await client.post(f"https://api.telegram.org/bot{token}/{method}", json=data)
    response.raise_for_status()
    body = response.json()
    if not body.get("ok"):
        # Do not relay Telegram API descriptions; they can contain request details.
        raise RuntimeError("Telegram Bot API request failed")
    return body.get("result")


async def _claim_update(client: httpx.AsyncClient, env: Mapping[str, str], update_id: int) -> bool:
    redis_url = env.get("UPSTASH_REDIS_REST_URL", "").strip().rstrip("/")
    redis_token = env.get("UPSTASH_REDIS_REST_TOKEN", "").strip()
    if not redis_url or not redis_token:
        raise RuntimeError("Durable update de-duplication is not configured")
    response = await client.post(
        redis_url,
        headers={"Authorization": f"Bearer {redis_token}"},
        json=["SET", f"wasteai:telegram:update:{update_id}", "1", "NX", "EX", DEDUP_TTL_SECONDS],
    )
    response.raise_for_status()
    result = response.json()
    if "error" in result:
        raise RuntimeError("Durable update de-duplication request failed")
    return result.get("result") == "OK"


async def _photo_bytes(client: httpx.AsyncClient, token: str, photo: dict) -> bytes:
    file_size = photo.get("file_size")
    if file_size is not None and int(file_size) > MAX_IMAGE_BYTES:
        raise ValueError("Please send an image smaller than 10 MB.")
    file_info = await _telegram_call(client, token, "getFile", {"file_id": photo["file_id"]})
    file_path = file_info.get("file_path") if isinstance(file_info, dict) else None
    if not file_path or file_path.startswith("/") or ".." in Path(file_path).parts:
        raise ValueError("Telegram did not return a valid image file.")
    response = await client.get(f"https://api.telegram.org/file/bot{token}/{file_path}")
    response.raise_for_status()
    data = response.content
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Please send an image smaller than 10 MB.")
    return data


async def _send_update_reply(client: httpx.AsyncClient, token: str, update: dict, classifier=None):
    message = update.get("message") or update.get("edited_message")
    if not isinstance(message, dict) or not isinstance(message.get("chat"), dict):
        return
    chat_id = message["chat"].get("id")
    if chat_id is None:
        return

    if isinstance(message.get("text"), str):
        text = _reply_for_text(message["text"])
    else:
        photos = message.get("photo")
        document = message.get("document")
        if isinstance(photos, list) and photos:
            photo = photos[-1]
        elif (
            isinstance(document, dict)
            and document.get("mime_type") in {"image/jpeg", "image/png", "image/webp"}
        ):
            photo = document
        else:
            text = "📷 Please send a photo of a waste item so I can classify it."
            await _telegram_call(client, token, "sendMessage", {
                "chat_id": chat_id, "text": text, "parse_mode": "HTML"
            })
            return

        try:
            image = await _photo_bytes(client, token, photo)
            model = classifier or await asyncio.to_thread(_get_classifier)
            prediction = await asyncio.to_thread(model.predict_bytes, image)
            text = format_result(prediction, LOW_CONFIDENCE_THRESHOLD, telegram_html=True)
        except ValueError as exc:
            text = f"Sorry, I couldn't process that image: {escape(str(exc))}"
        except Exception:
            text = "Sorry, I couldn't classify that photo. Please try again with a clear image."

    await _telegram_call(client, token, "sendMessage", {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "reply_to_message_id": message.get("message_id"),
        "allow_sending_without_reply": True,
    })


async def process_webhook(
    headers: Mapping[str, str],
    update: dict,
    *,
    env: Mapping[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
    classifier=None,
    claim_update=None,
) -> tuple[int, dict]:
    """Validate, atomically claim an update, and answer it at most once."""
    settings = _environment(env)
    provided_secret = next((v for k, v in headers.items() if k.lower() == TELEGRAM_HEADER), "")
    expected_secret = settings.get("TELEGRAM_WEBHOOK_SECRET", "")
    if not expected_secret or not hmac.compare_digest(str(provided_secret), expected_secret):
        return 403, {"ok": False, "error": "forbidden"}
    token = settings.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return 503, {"ok": False, "error": "not_ready"}
    if not isinstance(update, dict) or not isinstance(update.get("update_id"), int):
        return 400, {"ok": False, "error": "invalid_update"}
    if len(str(update["update_id"])) > 30:
        return 400, {"ok": False, "error": "invalid_update"}

    own_client = client is None
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(25.0, connect=5.0))
    try:
        if own_client:
            await http.__aenter__()
        try:
            claim = claim_update or _claim_update
            claimed = await claim(http, settings, update["update_id"])
        except Exception:
            # Fail closed: never answer unless the idempotency key was stored.
            return 503, {"ok": False, "error": "deduplication_unavailable"}
        if not claimed:
            return 200, {"ok": True, "duplicate": True}
        try:
            await _send_update_reply(http, token, update, classifier=classifier)
        except Exception:
            # Telegram may retry; the stored update key prevents a duplicate reply.
            return 502, {"ok": False, "error": "processing_failed"}
        return 200, {"ok": True}
    finally:
        if own_client:
            await http.__aexit__(None, None, None)

import asyncio
import json

from api._logic import _claim_update, process_webhook
from api.health import health_payload
from waste_ai.inference import Prediction


ENV = {
    "TELEGRAM_BOT_TOKEN": "unit-test-token",
    "TELEGRAM_WEBHOOK_SECRET": "unit_test_secret",
    "UPSTASH_REDIS_REST_URL": "https://redis.example.test",
    "UPSTASH_REDIS_REST_TOKEN": "unit-test-redis-token",
}
HEADERS = {"X-Telegram-Bot-Api-Secret-Token": ENV["TELEGRAM_WEBHOOK_SECRET"]}


class Response:
    def __init__(self, body, content=b""):
        self.body = body
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


class MockTelegramClient:
    def __init__(self, fail_send=False):
        self.calls = []
        self.fail_send = fail_send

    async def post(self, url, *, json=None, headers=None):
        self.calls.append(("POST", url, json, headers))
        if url == ENV["UPSTASH_REDIS_REST_URL"]:
            return Response({"result": "OK"})
        if url.endswith("/getFile"):
            return Response({"ok": True, "result": {"file_path": "photos/sample.jpg"}})
        if url.endswith("/sendMessage"):
            if self.fail_send:
                raise RuntimeError("simulated Telegram send failure")
            return Response({"ok": True, "result": {"message_id": 99}})
        raise AssertionError(f"Unexpected Telegram request path: {url.split('/')[-1]}")

    async def get(self, url):
        self.calls.append(("GET", url, None, None))
        return Response({}, content=b"fake-image-bytes")


class FakeClassifier:
    def __init__(self, prediction=None, error=None):
        self.prediction = prediction or Prediction("recyclable", 0.939, "hidden-model", (.061, .939))
        self.error = error
        self.received = []

    def predict_bytes(self, data):
        self.received.append(data)
        if self.error:
            raise self.error
        return self.prediction


def allow_claim(*args):
    async def claim(*_args):
        return args[0] if args else True
    return claim


def message_update(update_id=100, text=None, photo=None, document=None):
    msg = {"message_id": update_id, "chat": {"id": 55}}
    if text is not None:
        msg["text"] = text
    if photo is not None:
        msg["photo"] = photo
    if document is not None:
        msg["document"] = document
    return {"update_id": update_id, "message": msg}


def sent_text(client):
    call = next(c for c in client.calls if c[0] == "POST" and c[1].endswith("/sendMessage"))
    return call[2]


def test_rejects_missing_or_invalid_telegram_secret_before_processing():
    client = MockTelegramClient()
    status, body = asyncio.run(process_webhook({}, message_update(text="/start"), env=ENV, client=client))
    assert (status, body["error"]) == (403, "forbidden")
    assert client.calls == []


def test_health_is_a_lightweight_liveness_check():
    assert health_payload() == {"status": "ok"}


def test_missing_durable_dedup_configuration_fails_closed():
    client = MockTelegramClient()
    env = {"TELEGRAM_BOT_TOKEN": ENV["TELEGRAM_BOT_TOKEN"],
           "TELEGRAM_WEBHOOK_SECRET": ENV["TELEGRAM_WEBHOOK_SECRET"]}
    status, body = asyncio.run(process_webhook(HEADERS, message_update(text="/start"), env=env, client=client))
    assert (status, body["error"]) == (503, "deduplication_unavailable")
    assert client.calls == []


def test_atomic_redis_claim_uses_update_id_and_secret_header():
    client = MockTelegramClient()
    claimed = asyncio.run(_claim_update(client, ENV, 4321))
    assert claimed
    _, url, payload, headers = client.calls[0]
    assert url == ENV["UPSTASH_REDIS_REST_URL"]
    assert payload == ["SET", "wasteai:telegram:update:4321", "1", "NX", "EX", 86400]
    assert headers == {"Authorization": f"Bearer {ENV['UPSTASH_REDIS_REST_TOKEN']}"}


def test_duplicate_update_is_acknowledged_without_sending_another_reply():
    client = MockTelegramClient()
    status, body = asyncio.run(process_webhook(
        HEADERS, message_update(text="/start"), env=ENV, client=client,
        claim_update=allow_claim(False),
    ))
    assert (status, body) == (200, {"ok": True, "duplicate": True})
    assert not any(c[0] == "POST" and c[1].endswith("/sendMessage") for c in client.calls)


def test_send_failure_is_not_duplicated_when_telegram_retries_update():
    claims = iter([True, False])
    async def claim(*_args):
        return next(claims)

    update = message_update(text="/start")
    client = MockTelegramClient(fail_send=True)
    first = asyncio.run(process_webhook(
        HEADERS, update, env=ENV, client=client, claim_update=claim,
    ))
    second = asyncio.run(process_webhook(
        HEADERS, update, env=ENV, client=client, claim_update=claim,
    ))
    assert first[0] == 502
    assert second == (200, {"ok": True, "duplicate": True})
    assert sum(c[1].endswith("/sendMessage") for c in client.calls) == 1


def test_start_and_help_dispatch_html_messages():
    for update_id, command, expected in (
        (1, "/start", "<b>👋 Welcome to Waste Identifier!</b>"),
        (2, "/help", "<b>📷 How to use Waste Identifier</b>"),
        (3, "/about", "ITM 360 — binary waste classification research prototype."),
    ):
        client = MockTelegramClient()
        status, body = asyncio.run(process_webhook(
            HEADERS, message_update(update_id, command), env=ENV, client=client,
            claim_update=allow_claim(True),
        ))
        assert status == 200 and body["ok"]
        sent = sent_text(client)
        assert sent["text"].startswith(expected)
        assert sent["parse_mode"] == "HTML"
        if command == "/start":
            assert "1. Take a photo" in sent["text"]
        elif command == "/help":
            assert "3. Read the prediction" in sent["text"]


def test_photo_is_downloaded_from_telegram_and_classified_with_real_formatter():
    client = MockTelegramClient()
    classifier = FakeClassifier()
    update = message_update(photo=[{"file_id": "photo-id", "file_size": 123}])
    status, body = asyncio.run(process_webhook(
        HEADERS, update, env=ENV, client=client, classifier=classifier,
        claim_update=allow_claim(True),
    ))
    assert status == 200 and body["ok"]
    assert classifier.received == [b"fake-image-bytes"]
    assert any(c[1].endswith("/getFile") for c in client.calls)
    assert any(c[0] == "GET" and "/file/botunit-test-token/" in c[1] for c in client.calls)
    sent = sent_text(client)
    assert sent["text"].startswith("<b>♻️ Recyclable</b>")
    assert "AI Confidence: 93.9%" in sent["text"]
    assert "hidden-model" not in sent["text"]
    assert sent["parse_mode"] == "HTML"


def test_non_recyclable_and_low_confidence_formatting():
    for label, confidence, title in (
        ("non_recyclable", .912, "<b>🗑️ Non-Recyclable</b>"),
        ("recyclable", .519, "<b>⚠️ Possibly Recyclable</b>"),
        ("non_recyclable", .519, "<b>⚠️ Possibly Non-Recyclable</b>"),
    ):
        client = MockTelegramClient()
        classifier = FakeClassifier(Prediction(label, confidence, "hidden-model", (.5, .5)))
        update = message_update(photo=[{"file_id": "photo-id", "file_size": 123}])
        status, _ = asyncio.run(process_webhook(
            HEADERS, update, env=ENV, client=client, classifier=classifier,
            claim_update=allow_claim(True),
        ))
        assert status == 200
        sent = sent_text(client)["text"]
        assert title in sent
        assert f"AI Confidence: {confidence:.1%}" in sent


def test_unsupported_message_gets_photo_prompt():
    client = MockTelegramClient()
    status, _ = asyncio.run(process_webhook(
        HEADERS, message_update(text="hello"), env=ENV, client=client,
        claim_update=allow_claim(True),
    ))
    assert status == 200
    assert sent_text(client)["text"] == "📷 Please send a photo of a waste item so I can classify it."


def test_invalid_document_does_not_download_and_sends_photo_prompt():
    client = MockTelegramClient()
    update = message_update(document={"file_id": "doc-id", "mime_type": "application/pdf"})
    status, _ = asyncio.run(process_webhook(
        HEADERS, update, env=ENV, client=client, claim_update=allow_claim(True),
    ))
    assert status == 200
    assert sent_text(client)["text"].startswith("📷 Please send")
    assert not any(c[0] == "GET" for c in client.calls)


def test_invalid_photo_errors_are_html_escaped_and_generic_errors_are_friendly():
    for error, expected in (
        (ValueError("Invalid <photo> & data"), "Invalid &lt;photo&gt; &amp; data"),
        (RuntimeError("internal detail"), "Sorry, I couldn't classify that photo."),
    ):
        client = MockTelegramClient()
        classifier = FakeClassifier(error=error)
        update = message_update(photo=[{"file_id": "photo-id", "file_size": 123}])
        status, _ = asyncio.run(process_webhook(
            HEADERS, update, env=ENV, client=client, classifier=classifier,
            claim_update=allow_claim(True),
        ))
        sent = sent_text(client)["text"]
        assert status == 200
        assert expected in sent
        assert "internal detail" not in sent


def test_bad_update_is_rejected_without_external_calls():
    client = MockTelegramClient()
    status, body = asyncio.run(process_webhook(
        HEADERS, {"message": {}}, env=ENV, client=client,
    ))
    assert (status, body["error"]) == (400, "invalid_update")
    assert client.calls == []

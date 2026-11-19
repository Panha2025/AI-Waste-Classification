import asyncio
from io import BytesIO

from waste_ai import bot
from waste_ai.inference import Prediction


class FakeFile:
    file_size = 4
    mime_type = "image/jpeg"

    def __init__(self, payload=b"fake"):
        self.payload = payload

    async def get_file(self):
        return self

    async def download_to_memory(self, out):
        out.write(self.payload)


class FakeMessage:
    document = None

    def __init__(self, media):
        self.photo = [media]
        self.replies = []

    async def reply_text(self, text):
        self.replies.append(text)


class FakeClassifier:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def predict_bytes(self, payload):
        self.calls.append(payload)
        if self.error:
            raise self.error
        return Prediction("recyclable", .91, "test-model", (.09,.91))


class FakeUpdate:
    def __init__(self, message):
        self.effective_message = message


def test_telegram_photo_handler_repeats_predictions_without_commands(monkeypatch):
    classifier = FakeClassifier()
    monkeypatch.setattr(bot, "CLASSIFIER", classifier)
    message = FakeMessage(FakeFile())
    update = FakeUpdate(message)
    asyncio.run(bot.handle_image(update, None))
    asyncio.run(bot.handle_image(update, None))
    assert len(classifier.calls) == 2
    assert all("♻️ Recyclable" in reply for reply in message.replies)
    assert all("local acceptance varies" in reply.lower() for reply in message.replies)


def test_telegram_photo_handler_replies_gracefully_on_bad_image(monkeypatch):
    classifier = FakeClassifier(ValueError("Invalid or unsupported image"))
    monkeypatch.setattr(bot, "CLASSIFIER", classifier)
    message = FakeMessage(FakeFile(b"bad"))
    asyncio.run(bot.handle_image(FakeUpdate(message), None))
    assert "couldn't process" in message.replies[0]

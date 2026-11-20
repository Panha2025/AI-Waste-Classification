from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/manage_webhook.py"
sys.path.insert(0, str(SCRIPT.parent))
import manage_webhook


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_register_script_sends_token_and_secret_only_in_request(monkeypatch, capsys):
    calls = []
    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return Response({"ok": True, "result": True})

    monkeypatch.setattr(manage_webhook.httpx, "post", fake_post)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "set", "https://example.vercel.app/api/webhook"])
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "never-print-this-token")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "never-print-this-secret")

    assert manage_webhook.main() == 0
    output = capsys.readouterr().out
    assert "registered" in output
    assert "never-print-this-token" not in output
    assert "never-print-this-secret" not in output
    assert "never-print-this-token" in calls[0][0]
    assert calls[0][1]["json"]["secret_token"] == "never-print-this-secret"
    assert calls[0][1]["json"]["allowed_updates"] == ["message"]


def test_inspection_prints_only_public_status_and_not_token(monkeypatch, capsys):
    def fake_post(url, **kwargs):
        return Response({"ok": True, "result": {
            "url": "https://example.vercel.app/api/webhook", "pending_update_count": 3,
        }})

    monkeypatch.setattr(manage_webhook.httpx, "post", fake_post)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "info"])
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "never-print-this-token")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "secret")
    assert manage_webhook.main() == 0
    output = capsys.readouterr().out
    assert "Webhook configured: True" in output
    assert "Pending updates: 3" in output
    assert "never-print-this-token" not in output


def test_registration_rejects_non_https_or_wrong_path_without_api_call(monkeypatch):
    def fail_if_called(*args, **kwargs):
        pytest.fail("Telegram API should not be called for an invalid URL")

    monkeypatch.setattr(manage_webhook.httpx, "post", fail_if_called)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "set", "http://example.com/wrong"])
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "valid_secret")
    with pytest.raises(SystemExit, match="public HTTPS"):
        manage_webhook.main()


def test_telegram_request_errors_do_not_expose_api_url_or_token(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise RuntimeError("https://api.telegram.org/botsecret-token/getWebhookInfo")

    monkeypatch.setattr(manage_webhook.httpx, "post", fail)
    with pytest.raises(SystemExit) as exc:
        manage_webhook._request("secret-token", "getWebhookInfo")
    output = str(exc.value) + capsys.readouterr().out
    assert "secret-token" not in output
    assert "api.telegram.org" not in output

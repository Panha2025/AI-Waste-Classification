# Waste Identifier — Vercel webhook package

This directory is an isolated webhook deployment variant. It does not replace the project's local long-polling bot and does not modify the university submission ZIP. It uses the existing selected binary MobileNetV2 model and the same inference code, mapping, preprocessing, 0.33 recyclable threshold, 0.70 low-confidence threshold, and compact Telegram HTML responses.

## Vercel Hobby feasibility

Vercel currently documents Python Functions as beta, with Python 3.12, 3.13, and 3.14 available. This package pins Python 3.13 to match the available Linux CPU TensorFlow 2.21 wheel. Hobby functions have 2 GB RAM and 1 vCPU, and up to 300 seconds when Fluid Compute is enabled. Standard Python function bundles are limited to 500 MB uncompressed; Large Functions support up to 5 GB in public beta. This TensorFlow deployment is likely above 500 MB: the project's installed macOS TensorFlow package is about 1.1 GB (not a Linux bundle measurement). The package opts into the model only and excludes unrelated project data; enable `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` in Vercel if the build requests it. Large Functions require Fluid Compute. [TensorFlow install guidance](https://www.tensorflow.org/install/pip)

The 2 GB runtime memory limit is the key risk. The model is loaded lazily on the first photo request and cached per warm instance, but TensorFlow's actual Linux cold-start and peak memory must be verified in a Vercel preview deployment. A local Mac test cannot prove the Linux memory behavior. A 60-second function limit is configured, within both the legacy Hobby maximum and the current Fluid Compute maximum.

Vercel's function request body limit is 4.5 MB. Telegram sends the update metadata to the webhook, not the photo bytes; the function downloads the photo from Telegram after checking the reported size. Photo bytes are limited to 10 MB, matching the local classifier.

References: [Python runtime and bundle rules](https://vercel.com/docs/functions/runtimes/python), [Function memory, duration, and payload limits](https://vercel.com/docs/functions/limitations), [Large Functions announcement](https://vercel.com/changelog/vercel-functions-can-now-be-up-to-5-gb-in-package-size).

## Contents

- `api/webhook.py` — HTTPS Telegram webhook function at `/api/webhook`.
- `api/health.py` — readiness check at `/api/health`.
- `api/_logic.py` — secret validation, durable update de-duplication, Telegram API calls, and response dispatch.
- `waste_ai/` — minimal inference package copied from the project.
- `artifacts/` — selected model and manifest only.
- `scripts/manage_webhook.py` — safe `set` and `info` commands; does not print the bot token.
- `tests/` — tests with mocked Telegram and Redis HTTP calls; no messages are sent.

## Deploy without activating the webhook yet

1. Import **only this directory** as the Vercel project root. Do not import the parent project root. The package contains no datasets, `.env`, or bot token.
2. Add these environment variables in Vercel Project Settings → Environment Variables (Production and Preview as needed):
   - `TELEGRAM_BOT_TOKEN` — the private BotFather token.
   - `TELEGRAM_WEBHOOK_SECRET` — a random 1–256 character string using only letters, digits, `_`, or `-`.
   - `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` — credentials for an Upstash Redis database used for atomic update de-duplication. Upstash currently lists a free tier (250 MB and 500,000 commands/month); the Redis database is a separate service and its limits/terms can change. Without this store, the webhook fails closed and sends no reply; serverless memory is not shared reliably across cold starts or instances. [Current pricing](https://upstash.com/pricing/redis)
3. Deploy a Preview first. If Vercel reports the standard bundle limit, set `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` in the Vercel project environment and redeploy. This opts into Vercel's Large Functions beta; no paid plan is selected here.
4. Visit `https://<deployment-host>/api/health`. It should return `{"status":"ok"}`. This is a liveness check; it deliberately does not expose credential/configuration state or load TensorFlow.
5. Do not register the webhook until the preview and health check are confirmed. Registration changes the bot's active update delivery from local polling to HTTPS webhook. Local long polling and webhook delivery cannot safely compete.

The deployment package intentionally does not register or activate a webhook. Vercel's Hobby plan is for personal, non-commercial use; review Vercel's current [plan terms](https://vercel.com/pricing) for your usage.

## Register or inspect the webhook later

After you approve activation, provide the token and secret to the local shell without putting them in source or command history. In zsh:

```zsh
read -s "TELEGRAM_BOT_TOKEN?Telegram token: "; echo
export TELEGRAM_BOT_TOKEN
read -s "TELEGRAM_WEBHOOK_SECRET?Webhook secret: "; echo
export TELEGRAM_WEBHOOK_SECRET
python scripts/manage_webhook.py set https://<deployment-host>/api/webhook
python scripts/manage_webhook.py info
```

The script never prints token values or full Bot API URLs. `set` sends `allowed_updates=["message"]` and the configured secret header value to Telegram; `info` prints the configured public webhook URL and pending update count. It has no delete or unset command.

## Local checks

Use Python 3.13 for the Vercel-matching dependency target. The included smoke test was run in the project's local Python 3.11 environment with TensorFlow 2.21:

```sh
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
python scripts/model_smoke_test.py
```

Tests mock Telegram and Redis. The model smoke test loads the included model and runs a synthetic image through the real preprocessing/inference path; its predicted class is a pipeline check, not an evaluation result. No `.env` is needed for tests.

## Update de-duplication and delivery semantics

Before replying, the webhook atomically claims Telegram's `update_id` in Redis with a one-day expiry. Retries that reach another warm/cold function instance are acknowledged without another reply. The claim is deliberately fail-closed if Redis is unavailable. Telegram's send-message API and Redis do not share a transaction, so exactly-once delivery cannot be guaranteed: a failure after claiming an update can suppress a later retry, favoring no duplicate replies over guaranteed delivery. Monitor Vercel function logs and Telegram webhook status after activation.

## Scope and limitations

- This is a serverless webhook, not a continuously running polling process. It has no public user-facing web app; only webhook and health routes exist.
- Telegram photo bytes are fetched from Telegram's Bot API and processed in memory; the deployment does not save images.
- Labels remain provisional source-folder labels; no Cambodian local evaluation is claimed.
- This package includes the selected 18.3 MB `.keras` model, not source datasets or training artifacts.
- A Vercel build, cloud model load, cloud memory measurement, webhook registration, and live Telegram test have not been run.

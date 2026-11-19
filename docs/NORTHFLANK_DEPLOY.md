# Northflank Free Sandbox deployment

This deployment runs the existing binary Telegram bot as one long-polling CPU service. It has
no HTTP server, public port, training data, or training runtime. The container includes only the
bot inference modules, deployment dependencies, selected model, and model manifest.

## Included model and settings

- Model: `artifacts/models/binary_mobilenetv2_finetuned.keras` (18 MB locally)
- `MODEL_PATH`: `/app/artifacts/models/binary_mobilenetv2_finetuned.keras`
- Binary class order: `non_recyclable`, `recyclable`
- Recyclable probability threshold: `0.33` from the selected model manifest
- CPU runtime: Python 3.11, TensorFlow CPU 2.21.0, one intra/inter-op thread
- Start command: `python -m waste_ai.bot`
- No `EXPOSE` instruction or Northflank port is required; long polling makes outbound HTTPS
  requests to Telegram.

The deployment image uses `requirements-deploy.txt`, not the full development requirements.
The bot does not import scikit-learn at runtime; its import is lazy in the evaluation-only
metrics function. Docker is not required for local unit tests, but a local Docker build is a
useful extra check if Docker Desktop is installed.

## Prepare the private GitHub repository

1. Use the existing repository `Panha2025/AI-Waste-Classification`; do not create another one.
   It is currently public, so all committed files will be publicly visible. From the project root,
   initialize Git if needed, add the project files, commit, and push the `main` branch:

   ```bash
   git init -b main
   git add .
   git status --short
   git commit -m "Prepare WasteAI Telegram bot deployment"
   git remote add origin https://github.com/Panha2025/AI-Waste-Classification.git
   git push -u origin main
   ```

   Confirm that the selected model and manifest are included, and `.env`, datasets, virtual
   environments, backup models, and other trained models are absent. Then commit and connect
   the GitHub repository URL as `origin` before pushing `main`. The repository must include
   `Dockerfile`, `.dockerignore`,
   `requirements-deploy.txt`, `waste_ai/`, `artifacts/model_manifest.json`, and
   `artifacts/models/binary_mobilenetv2_finetuned.keras`.
3. `.gitignore` excludes `.env`, downloaded datasets, virtual environments, processed data,
   training outputs, backup models, and other models. It allows only the selected binary model
   and active manifest under `artifacts/`.
4. Confirm before pushing that `git status --short` lists the selected model and active manifest
   but does not list `.env`, any dataset folder, or another `.keras` model. Do not use `git add
   -f` on ignored datasets or secrets.

The selected model is about 18 MB. Keep only this selected model and the small presentation
artifacts in the repository; the ignore rules exclude training data, other models, and backups.

## Deploy on Northflank

1. Create or open a Northflank project on the **Free Sandbox** plan. Do not choose BYOC, GPU,
   or a pay-as-you-go team/project.
2. Connect GitHub under the account/team's Git integrations and grant access to this private
   repository only.
3. Create one continuously running **combined service** from the repository and target branch.
4. Choose **Dockerfile** as the build type, with build context `/` and Dockerfile `/Dockerfile`.
5. Under environment variables/runtime variables, add `TELEGRAM_BOT_TOKEN` with the token from
   @BotFather. Mark/store it as a secret in Northflank; never commit or echo it. Optionally set
   `CONFIDENCE_THRESHOLD=0.70` for the warning message cutoff (this is separate from the model's
   0.33 class-decision threshold).
6. Do not add ports or public networking. Keep one replica/instance and long polling; do not
   create a second service instance using the same Telegram token.
7. Select the smallest CPU resource that can run TensorFlow within your Sandbox allowance. The
   current Northflank pricing page lists `nf-compute-50` at 0.5 vCPU / 1 GiB and $12/month; the
   Sandbox currently includes up to $20/month in Northflank Cloud resource usage. Check the live
   billing/usage page before deploying because plan quotas can change. Do not select a resource
   that would exceed your free allowance.
8. Deploy. In service logs, confirm the startup line names `binary_mobilenetv2_finetuned` and
   send the bot a test photo in Telegram. Logs should never contain the token.

## Memory and limits

TensorFlow CPU on Linux x86-64 with Python 3.11 is supported by TensorFlow's published wheels.
The model file itself is small (18 MB), but TensorFlow's runtime and image inference use much
more memory than the weights. The Northflank 1 GiB development plan is the lowest-cost Sandbox
fit; it may be tight. This workspace has no Docker executable, so the final Linux container
build could not be measured here. A local macOS CPU smoke run peaked at about 565 MiB RSS with
TensorFlow's thread counts limited to one; Linux x86-64 can differ. Watch Northflank's deployment
logs and memory metric on first start. If the service is OOM-killed or restarts, the next 2 GiB
plan currently exceeds the $20/month Sandbox resource allowance; stop the service instead of
silently selecting paid resources, and revisit hosting or runtime optimization.

The model was trained on source-folder-derived provisional labels, none human-verified, and no
Cambodian local test set exists. Predictions are research estimates and local collection rules
may differ.

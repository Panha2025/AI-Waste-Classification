# WasteAI — ITM 360 AI Waste Classification Telegram Bot

**AUPP • ITM-360 • Group 4 • Professor Pin Kuntha • Final submission: Nov 29, 2026**

> **Current experiment:** Direct binary prediction — Recyclable or Non-Recyclable — from a
> new two-output model. Targets are mapped from audited source folders; none are human-verified
> recycling decisions. See the binary report and metrics before interpreting model scores.

**Binary run completed Nov 19, 2026:** 7,664 images were used after excluding 101 unresolved
near-duplicate candidates upstream and one visually ambiguous TrashNet image. After
validation-only fine-tuning and threshold selection, the held-out test result against
provisional source labels was 89.83% accuracy, 88.82% balanced accuracy, and 0.745 macro F1;
Non-Recyclable recall rose from 42.0% to 87.7%, with Recyclable recall at 90.0%. Labels are
not human-verified. No Cambodian local images or Telegram token are available, so local
generalization and live polling remain untested. See `artifacts/BINARY_EXPERIMENT_REPORT.md`.

## What is included?

- Image audit: source/category counts, damaged files, duplicate exclusions, cleaning log,
  per-source visuals, and group-aware public train/validation/test splits.
- Direct two-class **MobileNetV2** and Custom CNN models, plus the preserved six-class
  MobileNetV2 backup.
- Real validation and held-out evaluation: accuracy, per-class precision/recall/F1, macro F1,
  confusion matrices, learning curves, source/split counts, actual CPU/RAM/GPU environment.
- Model selection based on **validation macro F1**, not the test set.
- Telegram bot: `/start`, `/help`, `/about`, automatic photo/image uploads, safe in-memory image
  handling, predicted label, confidence, low-confidence warning, error handling.
- Local CLI predictor, experiment-generated research report template, handoff checklist, tests.

## 1. Setup (Mac, Windows, or Linux)

Recommended: **Python 3.11**, in a new environment (Apple Silicon supported by TensorFlow pip).

```bash
cd WasteAI_Telegram_Final_Project
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/doctor.py
```

Windows PowerShell: `py -3.11 -m venv .venv` and `.venv\Scripts\Activate.ps1`.
If `python3.11` is unavailable, install Python 3.11 first.

**Apple Silicon:** TensorFlow's standard macOS installation supports CPU inference/training. For
optional GPU acceleration Apple documents `pip install tensorflow-metal` for supported combinations.
Check actual compatibility for your TensorFlow version; do not assume GPU is installed/working.
Large training jobs may be faster on a Colab GPU, then copy the `.keras` model and metrics back.

## 2. Existing public datasets and binary label policy

The staged sources are `data/raw/public/trashnet/` and
`data/raw/public/garbage_classification_v2/`. For the binary experiment, source folders
cardboard, paper, glass, metal, and plastic map provisionally to `recyclable`; `trash` maps
provisionally to `non_recyclable`. Dataset folder categories are not item-level judgments:
visible condition and local collection rules are not established by a material label. The
split manifest preserves the source label and provenance. No binary label has been human
verified. The prior six-class model and source labels remain preserved as a separate backup.

Dataset source, version, and license details remain in `data/processed/dataset_audit.md` and
`data/processed/dataset_review_addendum.md`. Cite counts from the sources actually used.

## 3. Prepare and audit all images

```bash
python -m waste_ai.data
```

Outputs:

- `data/processed/source_summary.csv` — number original/removed/used per dataset
- `data/processed/cleaning_log.csv` — each excluded image and reason
- `data/processed/split_manifest.csv` — real paths/labels/sources/capture groups/assigned splits
- `data/processed/split_counts.csv` — exact count per source/class/split
- `data/processed/dataset_examples.png` — sample images for professor
- `data/processed/representative_spot_check.md` — limited visual QA notes and flagged example
- `data/processed/summary.json` — comprehensive dataset statistics + warnings

The preparation process removes exact duplicates, excludes unresolved near-duplicate candidates,
and splits at capture-group level. It targets 70/15/15 train/validation/test; real proportions
may differ because groups stay intact. This experiment uses public data only and makes no claim
of Cambodian local-test performance.

> Limit: perceptual duplicate detection catches near-identical images, not every
> semantic duplicate or every burst-photo relationship. Inspect data manually and group
> photos captured together within a session folder. The cleaning log must be reviewed.

## 4. Train the prioritized model

```bash
python -m waste_ai.binary_data
python -m waste_ai.binary_training --model mobilenetv2 --epochs 8 \
  --batch-size 32 --patience 2 --weights imagenet
```

This creates a direct two-output softmax model. When available, the training code copies
the frozen ImageNet MobileNetV2 backbone from the preserved six-class model; it does not use
the old model's six-way head or predictions. Balanced class weights are computed from the
training split only. For a from-scratch baseline on the same binary split:

```bash
python -m waste_ai.binary_training --model custom_cnn --epochs 8 \
  --batch-size 32 --patience 2 --weights none
python -m waste_ai.binary_finetune --epochs 6 --batch-size 32 --top-layers 20 \
  --non-recyclable-weight-multiplier 2.0
python -m waste_ai.binary_select_evaluate
```

Training and validation outputs, duration, hardware, and model files are saved separately
under `artifacts/metrics/binary_*` and `artifacts/models/binary_*.keras`.

Training outputs actual `.keras` models in `artifacts/models/`, training settings and
hardware data in `artifacts/metrics/*_train.json`, and history plots. Training uses only
train and validation; the held-out sets are not inspected for selection.

## 5. Evaluate and select a model

```bash
python -m waste_ai.binary_evaluate
```

This compares the baseline and fine-tuned MobileNetV2 on validation only, requiring at least
80% validation Recyclable recall while selecting by balanced accuracy, then evaluates only
the selected model on the held-out test. It saves threshold sweeps, per-class results,
predictions and a confusion matrix. The original binary weights are preserved in
`artifacts/backups/`; the six-class manifest is `artifacts/model_manifest_six_class.json`.

This public-only experiment has no Cambodian local-test images; do not claim local
generalization from the public test metrics.

## 6. Run a local prediction

```bash
python -m waste_ai.predict /path/to/your/image.jpg
```

## 7. Start the Telegram bot

1. Open Telegram and create a new bot through the official **@BotFather** account.
2. Copy `.env.example` to `.env` and enter your token **only in your private local `.env`**.
3. Run the bot:

```bash
cp .env.example .env
python -m waste_ai.bot
```

Send the bot a photo. It loads the selected two-output binary model, predicts Recyclable or
Non-Recyclable directly, and returns confidence, a low-confidence warning, and brief guidance.
Local acceptance varies; this provisional model is not a disposal guarantee.

**Security:** Do not commit or share `.env`; do not paste the token into ChatGPT/Codex/GitHub.
If a bot token was ever exposed publicly, revoke it via BotFather and replace it. This bot
stores no uploaded photo on disk; however **Telegram processes/hosts messages under its
own privacy policies**. The running bot process must stay online to reply; production deployment
can use a Python worker/service supporting long polling rather than a static Vercel site.
For the prepared Northflank CPU container and private-repository deployment steps, see
[`docs/NORTHFLANK_DEPLOY.md`](docs/NORTHFLANK_DEPLOY.md).

## 8. Presentation evidence

The current binary results, validation threshold comparison, selected test metrics, hardware,
and limitations are documented in `artifacts/BINARY_EXPERIMENT_REPORT.md`. The older
`artifacts/DRAFT_FINAL_REPORT.md` is the prior six-class experiment report and should be
treated as historical evidence, not the current binary result.

See `docs/METHODOLOGY.md`, `docs/PROFESSOR_FEEDBACK_CHECKLIST.md`,
`docs/TEAM_CONTRIBUTIONS.md`, and `docs/CODEX_HANDOFF.md`.

## 9. Test code

```bash
python -m pytest -q
python scripts/doctor.py
```

Automated checks include six-class and binary metrics, binary split leakage checks,
model preprocessing and Telegram-handler behavior.
Without a private bot token, they cannot verify a live exchange with Telegram's servers.

## Important limitation

This is a two-category **image classification research prototype**, not an object detector or a
general-purpose conversational chatbot. It works best with a clear photo of ONE item. If
multiple items are pictured it cannot locate/classify each one. Model confidence does not
prove a source label is correct or that an item is accepted by local recycling services.

The measured test set is public data from the same dataset sources as training. It does not
measure performance on Cambodian waste. Labels are mapped from source folders, not manually
verified. Test accuracy matches the always-Recyclable baseline due to class imbalance; use the
per-class results and weak-label limitations in the binary report.

## Historical binary review artifacts

The earlier 600-image human-review queue remains unreviewed and separate. Binary training
uses the new source-mapped split manifest at `data/processed/binary_split_manifest.csv`.
The six-class source labels, model, and previous report remain preserved for comparison.

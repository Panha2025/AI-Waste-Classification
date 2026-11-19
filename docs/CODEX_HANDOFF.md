# WasteAI handoff — current binary objective

The current research target is direct two-class image classification:
**Recyclable** or **Non-Recyclable**. The active Telegram bot loads the binary model listed
in `artifacts/model_manifest.json`. The old six-class MobileNetV2 remains a preserved backup
and its predictions are not translated into binary decisions.

## Current status

- Binary data: 7,664 images after excluding 101 unresolved near-duplicate candidates upstream
  and one visually ambiguous TrashNet example. Source-folder mappings are provisional; zero
  binary labels are human-verified.
- Direct binary MobileNetV2 and Custom CNN models have been trained and evaluated on the same
  capture-group-aware split. The selected model is fine-tuned MobileNetV2 at a 0.33
  recyclable-probability threshold. Test metrics: 89.83% accuracy, 88.82% balanced accuracy,
  0.745 macro F1; Non-Recyclable recall 87.65%, Recyclable recall 90.0%. These measure agreement
  with provisional source-folder labels, not verified recycling outcomes.
- Automated checks passed: run `.venv/bin/python -m pytest -q` and
  `.venv/bin/python scripts/doctor.py` to verify the current environment.
- No Cambodian local test images are available. Do not claim local generalization.
- No private `.env` or Telegram token is configured. Live bot polling is not verified.
- The extracted `data/waste_classification_v2/` supplement has been audited but was not added
  to the model: uploader license terms could not be verified from the downloaded files or
  readable primary metadata, and representative `N` samples include common bottle, can, and
  paper items. Its report, counts, exact/near-duplicate lists, and spot-check contact sheet are
  under `data/processed/waste_classification_v2_*` and `artifacts/waste_classification_v2_spotcheck.jpg`.
  Preserve the existing split and selected model until reuse rights and a suitable residual-waste
  label policy are established.

## Reproduce the binary experiment

```bash
python -m waste_ai.binary_data
python -m waste_ai.binary_training --model mobilenetv2 --epochs 8 --batch-size 32 --patience 2 --weights imagenet
python -m waste_ai.binary_training --model custom_cnn --epochs 8 --batch-size 32 --patience 2 --weights none
python -m waste_ai.binary_finetune --epochs 6 --batch-size 32 --top-layers 20 --non-recyclable-weight-multiplier 2.0
python -m waste_ai.binary_select_evaluate
```

`binary_select_evaluate` compares models and thresholds on validation, then evaluates only the
selected model on the held-out test set. Do not use the earlier `binary_evaluate` script for
current selection; it reflects the initial threshold experiment.

Use `data/processed/binary_split_manifest.csv` and `binary_summary.json` for provenance and
splits. See `artifacts/BINARY_EXPERIMENT_REPORT.md` and `docs/METHODOLOGY.md` before reporting
metrics. These metrics measure agreement with source-folder-derived targets, not manually
verified recycling decisions. Keep the 600-image review queue and original datasets intact.

## Live Telegram test

Copy `.env.example` to a private `.env`, set `TELEGRAM_BOT_TOKEN` locally, and run
`.venv/bin/python -m waste_ai.bot`. Never paste or print the token. Model inference and a
simulated Telegram photo-handler workflow have been exercised with held-out images; no real
Telegram exchange has been made.

## Research items still needing human evidence

- Review the source-mapped labels and actual item condition; do not present them as ground truth.
- Provide factual team contributions in `docs/TEAM_CONTRIBUTIONS.md`.
- Add independently labeled local/Cambodian images only if the team later resumes that scope.
- Obtain explicit dataset-image reuse terms before incorporating any supplemental data.

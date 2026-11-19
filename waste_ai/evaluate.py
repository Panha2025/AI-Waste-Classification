"""Evaluate held-out combined and separate local test sets, select deployment model by validation F1."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from waste_ai.config import ARTIFACTS, METRICS, MODELS, LABELS, CLASS_TO_INDEX, IMAGE_SIZE, ensure_dirs
from waste_ai.metrics import summarize_predictions
from waste_ai.training import SUPPORTED, load_manifest, make_dataset


def evaluate_one(model_name, data, batch_size=16):
    import tensorflow as tf
    model_path = MODELS / f"{model_name}.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not trained: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    results = {"model": model_name, "model_path": str(model_path),
               "evaluated_utc": datetime.now(timezone.utc).isoformat(), "sets": {}}
    for split in ("test", "local_test"):
        frame = data[data.split == split]
        if frame.empty:
            results["sets"][split] = {"n_images": 0, "warning": "No images in this split"}
            continue
        probabilities = model.predict(make_dataset(frame, batch_size), verbose=0)
        truth = frame["label"].map(CLASS_TO_INDEX).to_numpy(dtype=int)
        results["sets"][split] = summarize_predictions(truth, probabilities)
        raw = pd.DataFrame({"path": frame.path.to_numpy(), "source": frame.source.to_numpy(),
                            "true_label": frame.label.to_numpy(),
                            "predicted_label": [LABELS[i] for i in probabilities.argmax(axis=1)],
                            **{f"probability_{label}": probabilities[:, i]
                               for i, label in enumerate(LABELS)}})
        raw.to_csv(METRICS / f"{model_name}_{split}_predictions.csv", index=False)
        plot_confusion(model_name, split, results["sets"][split]["confusion_matrix"])
    (METRICS / f"{model_name}_evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf8")
    return results


def plot_confusion(name, split, matrix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=13)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.tick_params(axis="x", labelrotation=25)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{name} — {split}")
    fig.tight_layout()
    fig.savefig(METRICS / f"{name}_{split}_confusion.png", dpi=160)
    plt.close(fig)


def choose_best_by_validation(names):
    candidates = []
    for name in names:
        record = METRICS / f"{name}_train.json"
        if record.exists():
            j = json.loads(record.read_text(encoding="utf8"))
            candidates.append((float(j["validation_metrics"]["macro_f1"]), name))
    if not candidates:
        raise FileNotFoundError("No training metrics found; train models first")
    # Test scores do not affect selection. Alphabetical tie-break is deterministic.
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1], candidates[0][0]


def main():
    p = argparse.ArgumentParser(description="Compare trained models on untouched test sets")
    p.add_argument("--all", action="store_true", help="Evaluate all available trained models")
    p.add_argument("--model", choices=SUPPORTED)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    ensure_dirs()
    data = load_manifest()
    if args.model:
        names = [args.model]
    elif args.all:
        names = [n for n in SUPPORTED if (MODELS / f"{n}.keras").exists()]
    else:
        names = [n for n in SUPPORTED if (MODELS / f"{n}.keras").exists()]
    if not names:
        p.error("No trained .keras models found. Run python -m waste_ai.training --model all")
    results = [evaluate_one(n, data, args.batch_size) for n in names]
    best, score = choose_best_by_validation(names)
    manifest = {
        "selected_model": best, "model_path": str((MODELS / f"{best}.keras").resolve()),
        "selection_method": "highest validation macro F1; held-out test results NOT used to select",
        "validation_macro_f1": score,
        "image_size": IMAGE_SIZE, "labels": LABELS,
        "output_type": "six-class softmax probabilities",
    }
    (ARTIFACTS / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    table = []
    for r in results:
        row = {"model": r["model"]}
        for split, metrics in r["sets"].items():
            row[f"{split}_n"] = metrics.get("n_images", 0)
            for metric in ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
                           "macro_f1", "weighted_f1"):
                row[f"{split}_{metric}"] = metrics.get(metric)
        table.append(row)
    pd.DataFrame(table).to_csv(METRICS / "model_comparison.csv", index=False)
    print(pd.DataFrame(table).to_string(index=False))
    print(f"\nDeployment model: {best} (selection based on validation macro F1 = {score:.4f})")
    print("Results, confusion matrices, and comparison CSV saved in artifacts/metrics/")


if __name__ == "__main__":
    main()

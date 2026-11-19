"""Evaluate binary models on the held-out source-label test split and select by validation F1."""
from __future__ import annotations

import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from waste_ai.binary_data import BINARY_LABELS, OUTPUT as MANIFEST
from waste_ai.binary_training import LABEL_TO_INDEX, make_binary_dataset
from waste_ai.config import ARTIFACTS, METRICS, MODELS
from waste_ai.metrics import summarize_predictions


def _plot_confusion(name, matrix):
    fig, ax = plt.subplots(figsize=(5.8, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(2), ["Non-Recyclable", "Recyclable"])
    ax.set_yticks(range(2), ["Non-Recyclable", "Recyclable"])
    ax.set_xlabel("Predicted source-derived class")
    ax.set_ylabel("Held-out source-derived class")
    ax.set_title(f"{name} — binary test confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center",
                    color="white" if matrix[i][j] > max(map(max, matrix)) / 2 else "black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(METRICS / f"binary_{name}_test_confusion.png", dpi=160)
    plt.close(fig)


def evaluate_all(batch_size=32):
    import tensorflow as tf
    frame = pd.read_csv(MANIFEST)
    test = frame[frame.split == "test"].copy()
    val = frame[frame.split == "val"].copy()
    if set(test.binary_label) != set(BINARY_LABELS) or set(val.binary_label) != set(BINARY_LABELS):
        raise ValueError("Validation and held-out test splits must include both binary labels")
    test_ds = make_binary_dataset(test, batch_size)
    val_ds = make_binary_dataset(val, batch_size)
    truth = test.binary_label.map(LABEL_TO_INDEX).to_numpy(dtype=int)
    val_truth = val.binary_label.map(LABEL_TO_INDEX).to_numpy(dtype=int)
    rows, comparison = [], []
    for name in ("mobilenetv2", "custom_cnn"):
        path = MODELS / f"binary_{name}.keras"
        train_record_path = METRICS / f"binary_{name}_train.json"
        if not path.is_file() or not train_record_path.is_file():
            continue
        model = tf.keras.models.load_model(path, compile=False)
        probs = np.asarray(model.predict(test_ds, verbose=0), dtype=float)
        val_probs = np.asarray(model.predict(val_ds, verbose=0), dtype=float)
        if probs.shape != (len(test), 2):
            raise ValueError(f"{name} returned invalid binary output shape {probs.shape}")
        # Choose a binary decision threshold using validation only; never tune on test.
        thresholds = np.linspace(0.01, 0.99, 99)
        threshold_metrics = [(float(threshold),
                              summarize_predictions(val_truth, val_probs[:, 1], float(threshold)))
                             for threshold in thresholds]
        threshold, validation_metrics = max(
            threshold_metrics,
            key=lambda item: (item[1]["macro_f1"], item[1]["balanced_accuracy"],
                              -abs(item[0] - 0.5)))
        metrics = summarize_predictions(truth, probs[:, 1], threshold)
        record = {
            "model": name, "model_path": str(path), "class_order": list(BINARY_LABELS),
            "decision_threshold_recyclable_probability": threshold,
            "test_label_provenance": "provisional source-folder mapping; not human-verified",
            "validation_at_selected_threshold": validation_metrics, "test": metrics,
            "local_test": {"n_images": 0, "warning": "No local images"},
        }
        (METRICS / f"binary_{name}_evaluation.json").write_text(json.dumps(record, indent=2), encoding="utf8")
        predicted = (probs[:, 1] >= threshold).astype(int)
        output = test[["path", "source", "label", "binary_label", "split", "capture_group"]].copy()
        output["predicted_binary_label"] = [BINARY_LABELS[i] for i in predicted]
        output["confidence"] = probs[np.arange(len(probs)), predicted]
        output["p_non_recyclable"] = probs[:, 0]
        output["p_recyclable"] = probs[:, 1]
        output.to_csv(METRICS / f"binary_{name}_test_predictions.csv", index=False)
        _plot_confusion(name, metrics["confusion_matrix"])
        train_record = json.loads(train_record_path.read_text(encoding="utf8"))
        validation_f1 = float(validation_metrics["macro_f1"])
        row = {"model": name, "test_n": metrics["n_images"], "test_accuracy": metrics["accuracy"],
               "test_balanced_accuracy": metrics["balanced_accuracy"],
               "test_macro_precision": metrics["macro_precision"],
               "test_macro_recall": metrics["macro_recall"], "test_macro_f1": metrics["macro_f1"],
               "validation_macro_f1": validation_f1,
               "selected_recyclable_threshold": threshold}
        comparison.append(row)
        rows.append((name, validation_f1, threshold, metrics))
    if not comparison:
        raise FileNotFoundError("No trained binary model found")
    pd.DataFrame(comparison).to_csv(METRICS / "binary_model_comparison.csv", index=False)
    selected = max(rows, key=lambda row: row[1])
    manifest = {
        "selected_model": f"binary_{selected[0]}",
        "model_path": str(MODELS / f"binary_{selected[0]}.keras"),
        "selection_method": "highest validation macro F1; held-out test not used for selection",
        "validation_macro_f1": selected[1], "binary_threshold": selected[2],
        "image_size": [224, 224],
        "labels": list(BINARY_LABELS), "output_type": "two-class softmax probabilities",
        "label_provenance": "source-folder-derived provisional labels; none human-verified",
    }
    default_manifest = ARTIFACTS / "model_manifest.json"
    if default_manifest.exists():
        prior = json.loads(default_manifest.read_text(encoding="utf8"))
        if prior.get("selected_model") in ("mobilenetv2", "custom_cnn", "efficientnetb0"):
            shutil.copy2(default_manifest, ARTIFACTS / "model_manifest_six_class.json")
    default_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf8")
    (ARTIFACTS / "binary_model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    for name, val_f1, threshold, metrics in rows:
        print(f"{name}: test accuracy={metrics['accuracy']:.4f}; balanced_accuracy="
              f"{metrics['balanced_accuracy']:.4f}; macro_f1={metrics['macro_f1']:.4f}; "
              f"threshold={threshold:.2f}; "
              f"validation_macro_f1={val_f1:.4f}")
    print(f"Selected direct binary model: binary_{selected[0]}")
    return manifest


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate binary models on the held-out test split")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    evaluate_all(args.batch_size)


if __name__ == "__main__":
    main()

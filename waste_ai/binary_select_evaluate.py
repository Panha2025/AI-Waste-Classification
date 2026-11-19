"""Select a binary model/threshold on validation, then evaluate that winner once on test."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from waste_ai.binary_data import BINARY_LABELS, OUTPUT as MANIFEST
from waste_ai.binary_training import LABEL_TO_INDEX, make_binary_dataset
from waste_ai.config import ARTIFACTS, METRICS, MODELS
from waste_ai.metrics import summarize_predictions

CANDIDATES = {
    "binary_mobilenetv2": MODELS / "binary_mobilenetv2.keras",
    "binary_mobilenetv2_finetuned": MODELS / "binary_mobilenetv2_finetuned.keras",
}
MIN_RECYCLABLE_RECALL = 0.80


def _best_validation_threshold(truth, probabilities):
    records = []
    for threshold in np.linspace(0.01, 0.99, 99):
        metrics = summarize_predictions(truth, probabilities[:, 1], float(threshold))
        records.append({"threshold": float(threshold), **{
            "balanced_accuracy": metrics["balanced_accuracy"], "macro_f1": metrics["macro_f1"],
            "non_recyclable_precision": metrics["per_class"]["non_recyclable"]["precision"],
            "non_recyclable_recall": metrics["per_class"]["non_recyclable"]["recall"],
            "recyclable_precision": metrics["per_class"]["recyclable"]["precision"],
            "recyclable_recall": metrics["per_class"]["recyclable"]["recall"],
        }})
    all_rows = pd.DataFrame(records)
    acceptable = all_rows[all_rows.recyclable_recall >= MIN_RECYCLABLE_RECALL]
    relaxed = False
    if acceptable.empty:
        acceptable = all_rows[all_rows.recyclable_recall >= 0.75]
        relaxed = True
    if acceptable.empty:
        acceptable = all_rows
        relaxed = True
    best = acceptable.sort_values(
        ["balanced_accuracy", "macro_f1", "non_recyclable_recall", "recyclable_recall"],
        ascending=False, kind="stable").iloc[0]
    all_rows["meets_recyclable_recall_floor"] = all_rows.recyclable_recall >= MIN_RECYCLABLE_RECALL
    return float(best.threshold), best.to_dict(), all_rows, relaxed


def _save_confusion(matrix):
    fig, ax = plt.subplots(figsize=(5.8, 5))
    image = ax.imshow(matrix, cmap="Blues")
    labels = ("Non-Recyclable", "Recyclable")
    ax.set_xticks(range(2), labels)
    ax.set_yticks(range(2), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Source-mapped test label")
    ax.set_title("Selected binary model — untouched test split")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center",
                    color="white" if matrix[i][j] > max(map(max, matrix)) / 2 else "black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(METRICS / "binary_final_confusion.png", dpi=160)
    plt.close(fig)


def _save_validation_tradeoff():
    fig, ax = plt.subplots(figsize=(9, 5))
    styles = {"balanced_accuracy": "-", "non_recyclable_recall": "--",
              "recyclable_recall": ":"}
    for model_name, path in CANDIDATES.items():
        sweep_path = METRICS / f"{model_name}_validation_threshold_sweep.csv"
        if not sweep_path.is_file():
            continue
        sweep = pd.read_csv(sweep_path)
        for metric, style in styles.items():
            ax.plot(sweep.threshold, sweep[metric], linestyle=style,
                    label=f"{model_name.removeprefix('binary_')} {metric}")
        # Dashed vertical markers are filled by the best-threshold records below.
        selected_csv = METRICS / "binary_validation_model_comparison.csv"
        if selected_csv.is_file():
            selected_table = pd.read_csv(selected_csv)
            match = selected_table[selected_table.model == model_name]
            if not match.empty:
                threshold = float(match.iloc[0].threshold)
                ax.axvline(threshold, alpha=0.25, color="black")
    ax.axhline(MIN_RECYCLABLE_RECALL, color="gray", linewidth=1, alpha=0.5,
               label="Recyclable recall floor")
    ax.set_xlabel("Recyclable probability threshold")
    ax.set_ylabel("Validation score")
    ax.set_ylim(0, 1)
    ax.set_title("Validation-only threshold trade-offs")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(METRICS / "binary_validation_threshold_tradeoff.png", dpi=160)
    plt.close(fig)


def select_then_test(batch_size=32):
    import tensorflow as tf
    frame = pd.read_csv(MANIFEST)
    val = frame[frame.split == "val"].copy()
    test = frame[frame.split == "test"].copy()
    val_truth = val.binary_label.map(LABEL_TO_INDEX).to_numpy(dtype=int)
    test_truth = test.binary_label.map(LABEL_TO_INDEX).to_numpy(dtype=int)
    val_ds = make_binary_dataset(val, batch_size)
    validation_comparison = []
    thresholds = {}
    relaxed_flags = {}

    # Only validation data is used in this loop.
    for name, path in CANDIDATES.items():
        if not path.is_file():
            continue
        model = tf.keras.models.load_model(path, compile=False)
        val_probs = np.asarray(model.predict(val_ds, verbose=0), dtype=float)
        if val_probs.shape != (len(val), 2):
            raise ValueError(f"{name} returned unexpected validation shape {val_probs.shape}")
        threshold, selected_row, sweep, relaxed = _best_validation_threshold(val_truth, val_probs)
        sweep.insert(0, "model", name)
        thresholds[name] = threshold
        relaxed_flags[name] = relaxed
        validation_comparison.append({"model": name, **selected_row,
                                      "threshold": threshold,
                                      "recyclable_recall_floor_relaxed": relaxed})
        sweep.to_csv(METRICS / f"{name}_validation_threshold_sweep.csv", index=False)
        del model
    if not validation_comparison:
        raise FileNotFoundError("No binary model candidates found")
    validation_df = pd.DataFrame(validation_comparison).sort_values(
        ["balanced_accuracy", "macro_f1", "non_recyclable_recall", "recyclable_recall"],
        ascending=False, kind="stable")
    validation_df.to_csv(METRICS / "binary_validation_model_comparison.csv", index=False)
    _save_validation_tradeoff()
    selected_name = str(validation_df.iloc[0].model)
    selected_path = CANDIDATES[selected_name]
    selected_threshold = thresholds[selected_name]
    relaxed = relaxed_flags[selected_name]
    # Load the winner fresh and touch test data only now, after model and threshold selection.
    selected_model = tf.keras.models.load_model(selected_path, compile=False)
    test_ds = make_binary_dataset(test, batch_size)
    test_probs = np.asarray(selected_model.predict(test_ds, verbose=0), dtype=float)
    if test_probs.shape != (len(test), 2):
        raise ValueError(f"Selected model returned unexpected test shape {test_probs.shape}")
    metrics = summarize_predictions(test_truth, test_probs[:, 1], selected_threshold)
    result = {
        "model": selected_name, "model_path": str(selected_path),
        "class_order": list(BINARY_LABELS),
        "decision_threshold_recyclable_probability": selected_threshold,
        "threshold_selection": "validation balanced accuracy, tie-break macro F1 and non-recyclable recall, subject to recyclable recall >= 0.80",
        "recyclable_recall_floor_relaxed": relaxed,
        "test_label_provenance": "provisional source-folder mapping; no binary label human-verified",
        "test": metrics, "local_test": {"n_images": 0, "warning": "No local images"},
    }
    (METRICS / "binary_final_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    predicted = (test_probs[:, 1] >= selected_threshold).astype(int)
    predictions = test[["path", "source", "label", "binary_label", "split", "capture_group"]].copy()
    predictions["predicted_binary_label"] = [BINARY_LABELS[idx] for idx in predicted]
    predictions["confidence"] = test_probs[np.arange(len(test)), predicted]
    predictions["p_non_recyclable"] = test_probs[:, 0]
    predictions["p_recyclable"] = test_probs[:, 1]
    predictions.to_csv(METRICS / "binary_final_test_predictions.csv", index=False)
    _save_confusion(metrics["confusion_matrix"])
    selected_validation_metrics = {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in validation_df.iloc[0].to_dict().items()
    }
    selected_manifest = {
        "selected_model": selected_name, "model_path": str(selected_path),
        "selection_method": "highest validation balanced accuracy subject to >=0.80 recyclable recall; tie-break macro F1 and non-recyclable recall; test not used",
        "validation_threshold": selected_threshold,
        "validation_metrics": selected_validation_metrics,
        "binary_threshold": selected_threshold,
        "image_size": [224, 224], "labels": list(BINARY_LABELS),
        "output_type": "direct two-class softmax probabilities",
        "label_provenance": "source-folder-derived provisional labels; none human-verified",
    }
    manifest_text = json.dumps(selected_manifest, indent=2)
    (ARTIFACTS / "model_manifest.json").write_text(manifest_text, encoding="utf8")
    (ARTIFACTS / "binary_model_manifest.json").write_text(manifest_text, encoding="utf8")
    final_summary = {
        "selected_model": selected_name, "validation_candidates": validation_df.to_dict("records"),
        "test_evaluated_model_only": selected_name,
        "test_metrics": metrics,
        "test_accuracy_majority_baseline": float((test_truth == 1).mean()),
    }
    (ARTIFACTS / "binary_final_selection.json").write_text(json.dumps(final_summary, indent=2), encoding="utf8")
    print("Validation-only comparison:")
    print(validation_df.to_string(index=False))
    print(f"\nOnly the selected model was evaluated on the held-out test set: {selected_name}")
    print(json.dumps({"threshold": selected_threshold, **metrics}, indent=2))
    return selected_manifest, result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Select on validation, then test only the selected binary model")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    select_then_test(args.batch_size)


if __name__ == "__main__":
    main()

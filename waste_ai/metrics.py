"""Pure prediction and evaluation helpers; no invented scores."""
from __future__ import annotations
import numpy as np
from waste_ai.config import BINARY_LABELS, LABELS


def classify_probability(probability, threshold=0.5):
    """Legacy binary sigmoid helper retained for old reports/tests."""
    p = float(probability)
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Expected a finite sigmoid probability between 0 and 1")
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1")
    cls = int(p >= threshold)
    confidence = p if cls == 1 else 1 - p
    return BINARY_LABELS[cls], confidence


def classify_probabilities(probabilities):
    """Return top six-way category and its model probability."""
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    if values.shape != (len(LABELS),) or not np.isfinite(values).all():
        raise ValueError(f"Expected {len(LABELS)} finite class probabilities")
    if (values < 0).any() or (values > 1).any() or values.sum() <= 0:
        raise ValueError("Class probabilities must lie in [0,1] and have positive sum")
    index = int(values.argmax())
    return LABELS[index], float(values[index])


def classify_binary_probabilities(probabilities, threshold=0.5):
    """Classify a direct two-output softmax as non-recyclable or recyclable."""
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    if values.shape != (len(BINARY_LABELS),) or not np.isfinite(values).all():
        raise ValueError(f"Expected {len(BINARY_LABELS)} finite binary class probabilities")
    if (values < 0).any() or (values > 1).any() or values.sum() <= 0:
        raise ValueError("Class probabilities must lie in [0,1] and have positive sum")
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1")
    index = int(values[1] >= threshold)
    return BINARY_LABELS[index], float(values[index])


def summarize_predictions(y_true, probabilities, threshold=0.5):
    # Keep evaluation-only scikit-learn out of the bot's import path and runtime memory.
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                 balanced_accuracy_score, f1_score, precision_score, recall_score)
    true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim == 1:
        class_names = BINARY_LABELS
        labels = [0, 1]
        predicted = (probs >= threshold).astype(int)
    elif probs.ndim == 2 and probs.shape[1] in (len(BINARY_LABELS), len(LABELS)):
        class_names = BINARY_LABELS if probs.shape[1] == len(BINARY_LABELS) else LABELS
        labels = list(range(len(class_names)))
        if not np.isfinite(probs).all():
            raise ValueError("Predictions must be finite")
        predicted = probs.argmax(axis=1)
    else:
        raise ValueError(f"Expected a vector of sigmoid scores or an (n,{len(LABELS)}) probability matrix")
    if len(true) != len(predicted):
        raise ValueError("Mismatched predictions and ground truth")
    if not len(true):
        return {"n_images": 0, "warning": "No images in this evaluation set"}
    return {
        "n_images": int(len(true)),
        "class_labels": list(class_names),
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "macro_precision": float(precision_score(true, predicted, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true, predicted, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, predicted, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(true, predicted, labels=labels).tolist(),
        "per_class": classification_report(true, predicted, labels=labels, target_names=class_names,
                                           zero_division=0, output_dict=True),
    }

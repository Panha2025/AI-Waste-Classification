"""Load the saved selected model once; classify uploaded photos."""
from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import threading

import numpy as np
from PIL import Image, ImageOps

from waste_ai.config import ARTIFACTS, IMAGE_SIZE, LABELS, BINARY_LABELS, MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS
from waste_ai.metrics import classify_probabilities, classify_binary_probabilities


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    model: str
    probabilities: tuple[float, ...]


def resolve_model_path():
    explicit = os.getenv("MODEL_PATH")
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        metadata = ARTIFACTS / "model_manifest.json"
        if not metadata.exists():
            raise FileNotFoundError("No selected model. Run training and evaluation first.")
        entry = json.loads(metadata.read_text(encoding="utf8"))
        path = Path(entry["model_path"]).expanduser()
        if not path.is_absolute():
            path = (ARTIFACTS.parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Trained model not found at {path}")
    return path


class WasteClassifier:
    def __init__(self, model_path=None):
        import tensorflow as tf
        self.model_path = Path(model_path).resolve() if model_path else resolve_model_path()
        self.labels = tuple(LABELS)
        self.binary_threshold = 0.5
        manifest_path = ARTIFACTS / "model_manifest.json"
        if manifest_path.exists():
            metadata = json.loads(manifest_path.read_text(encoding="utf8"))
            saved_path = Path(metadata.get("model_path", "")).expanduser()
            # A manifest may retain its training-machine absolute path while the same
            # selected model is mounted at a container path. Match the selected model
            # identity too, so its binary class order and calibrated threshold travel.
            if (saved_path.resolve() == self.model_path or
                    metadata.get("selected_model") == self.model_path.stem):
                model_labels = metadata.get("labels")
                if model_labels:
                    self.labels = tuple(model_labels)
                self.binary_threshold = float(metadata.get("binary_threshold", 0.5))
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        self._lock = threading.Lock()  # prevent concurrent invocation of non-threadsafe model components

    def predict_bytes(self, data: bytes):
        if not data or len(data) > MAX_IMAGE_BYTES:
            raise ValueError("Image is empty or exceeds 10 MB. Send a smaller photo.")
        try:
            with Image.open(BytesIO(data)) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image exceeds 20 megapixels")
                image.load()
                processed = ImageOps.exif_transpose(image).convert("RGB")
                processed = ImageOps.pad(processed, IMAGE_SIZE, method=Image.Resampling.BILINEAR,
                                         color=(0, 0, 0), centering=(0.5, 0.5))
                array = np.asarray(processed, dtype=np.float32)[None, ...]
        except (OSError, Image.DecompressionBombError) as e:
            raise ValueError("Invalid or unsupported image") from e
        with self._lock:
            result = self.model.predict(array, verbose=0)
        values = np.asarray(result, dtype=float).reshape(-1)
        labels = getattr(self, "labels", tuple(LABELS))
        if values.shape != (len(labels),):
            raise ValueError(f"Model returned {len(values)} classes; expected {len(labels)}")
        if labels == BINARY_LABELS:
            label, confidence = classify_binary_probabilities(
                values, getattr(self, "binary_threshold", 0.5))
        elif labels == LABELS:
            label, confidence = classify_probabilities(values)
        else:
            raise ValueError(f"Unsupported model class mapping: {labels}")
        return Prediction(label, confidence, self.model_path.stem, tuple(float(x) for x in values))


def format_result(prediction: Prediction, uncertain_threshold: float = 0.70):
    confidence = prediction.confidence
    if prediction.label in BINARY_LABELS:
        label_text = ("♻️ Recyclable" if prediction.label == "recyclable"
                      else "🗑️ Non-Recyclable")
        guidance = ("If it is clean and dry, place it in your recycling stream only if your local service accepts it."
                    if prediction.label == "recyclable" else
                    "Dispose of it with residual waste unless your local service offers a separate route.")
        confidence_note = ("⚠️ Low-confidence result. Please verify the item manually.\n"
                           if confidence < uncertain_threshold else "")
        return ("Binary waste classification\n"
                f"Prediction: {label_text}\n"
                f"Model confidence: {confidence:.1%}\n"
                f"Model: {prediction.model}\n\n"
                f"{confidence_note}Guidance: {guidance}\n\n"
                "This is a provisional dataset-based estimate. Local acceptance varies by location.")
    guidance = {
        "cardboard": "Keep it dry and clean; flatten it if your local collection accepts cardboard.",
        "glass": "Check whether your local service accepts this glass type; handle broken glass safely.",
        "metal": "Empty and rinse metal containers if accepted by your local collection service.",
        "paper": "Keep paper clean and dry; food- or grease-soiled paper may not be accepted.",
        "plastic": "Acceptance depends on plastic type and local collection rules; check before sorting.",
        "trash": "This dataset category represents residual/mixed trash; separate identifiable recyclables where appropriate.",
    }
    confidence_note = ("⚠️ Low-confidence result. Verify the category manually.\n"
                       if confidence < uncertain_threshold else "")
    return ("Waste category prediction\n"
            f"Predicted category: {prediction.label.title()}\n"
            f"Model confidence: {confidence:.1%}\n"
            f"Model: {prediction.model}\n\n"
            f"{confidence_note}General guidance: {guidance.get(prediction.label, 'Check local sorting rules.')}\n\n"
            "Dataset category prediction only. Confidence is not certainty, and local acceptance varies.")

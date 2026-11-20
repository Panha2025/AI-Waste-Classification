#!/usr/bin/env python3
"""Load the selected model and verify its image inference path with a synthetic image."""
from io import BytesIO
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from waste_ai.inference import WasteClassifier


def main():
    classifier = WasteClassifier()
    buffer = BytesIO()
    Image.new("RGB", (320, 240), color=(115, 135, 95)).save(buffer, format="JPEG")
    result = classifier.predict_bytes(buffer.getvalue())
    assert result.label in {"recyclable", "non_recyclable"}
    assert 0.0 <= result.confidence <= 1.0
    assert classifier.labels == ("non_recyclable", "recyclable")
    assert classifier.binary_threshold == 0.33
    print("Selected model loaded; preprocessing and binary inference passed.")
    print(f"Mapping: {classifier.labels}; recyclable threshold: {classifier.binary_threshold:.2f}")
    print("The synthetic-image prediction is a smoke test, not an evaluation score.")


if __name__ == "__main__":
    main()

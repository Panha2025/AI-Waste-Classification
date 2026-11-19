from io import BytesIO
import numpy as np
from PIL import Image
import pytest

from waste_ai.inference import WasteClassifier, Prediction


class DummyModel:
    def predict(self, array, verbose=0):
        assert array.shape == (1, 224, 224, 3)
        assert array.dtype == np.float32
        return np.asarray([[0.01, 0.93, 0.02, 0.01, 0.02, 0.01]], dtype=np.float32)


class BinaryDummyModel:
    def predict(self, array, verbose=0):
        assert array.shape == (1, 224, 224, 3)
        return np.asarray([[.30, .70]], dtype=np.float32)


def test_photo_inference_with_stub_model_without_tensorflow(tmp_path):
    from threading import Lock
    import pathlib
    classifier = WasteClassifier.__new__(WasteClassifier)
    classifier.model = DummyModel()
    classifier.model_path = tmp_path / "dummy.keras"
    classifier._lock = Lock()
    data = BytesIO()
    Image.new("RGB", (60, 80), color=(130, 220, 190)).save(data, format="PNG")
    result = classifier.predict_bytes(data.getvalue())
    assert result.label == "glass"
    assert result.confidence == pytest.approx(.93, abs=.00001)
    assert result.model == "dummy"


def test_reject_bad_image_before_inference(tmp_path):
    from threading import Lock
    classifier = WasteClassifier.__new__(WasteClassifier)
    classifier.model = DummyModel()
    classifier.model_path = tmp_path / "dummy.keras"
    classifier._lock = Lock()
    with pytest.raises(ValueError, match="Invalid or unsupported image"):
        classifier.predict_bytes(b"this is not an image")
    with pytest.raises(ValueError, match="empty"):
        classifier.predict_bytes(b"")


def test_inference_uses_direct_binary_output_and_selected_threshold(tmp_path):
    from threading import Lock
    classifier = WasteClassifier.__new__(WasteClassifier)
    classifier.model = BinaryDummyModel()
    classifier.model_path = tmp_path / "binary_mobilenetv2.keras"
    classifier.labels = ("non_recyclable", "recyclable")
    classifier.binary_threshold = .65
    classifier._lock = Lock()
    data = BytesIO()
    Image.new("RGB", (60, 80), color=(100, 120, 140)).save(data, format="PNG")
    result = classifier.predict_bytes(data.getvalue())
    assert result.label == "recyclable"
    assert result.confidence == pytest.approx(.70)


def test_deployed_model_path_keeps_selected_manifest_binary_mapping(tmp_path, monkeypatch):
    import json
    import sys
    import types
    import waste_ai.config as config

    model_path = tmp_path / "binary_mobilenetv2_finetuned.keras"
    model_path.touch()
    manifest_path = tmp_path / "artifacts"
    manifest_path.mkdir()
    (manifest_path / "model_manifest.json").write_text(json.dumps({
        "selected_model": "binary_mobilenetv2_finetuned",
        "model_path": "/Users/training-machine/model.keras",
        "labels": ["non_recyclable", "recyclable"],
        "binary_threshold": 0.33,
    }))
    loaded = object()
    fake_tf = types.SimpleNamespace(keras=types.SimpleNamespace(
        models=types.SimpleNamespace(load_model=lambda path, compile=False: loaded)))
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setattr(config, "ARTIFACTS", manifest_path)

    classifier = WasteClassifier(model_path)
    assert classifier.model is loaded
    assert classifier.labels == ("non_recyclable", "recyclable")
    assert classifier.binary_threshold == pytest.approx(0.33)

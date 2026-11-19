from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from waste_ai import data


def test_group_splitting_never_leaks():
    rows = []
    for label in data.LABELS:
        for group in range(12):
            for _ in range(2):
                rows.append({"label": label, "capture_group": f"{label}-{group}"})
    first = data.split_groups_by_label(rows, ("train", "val", "test"), seed=42)
    again = data.split_groups_by_label(rows, ("train", "val", "test"), seed=42)
    assert first == again
    assert set(first.values()) == {"train", "val", "test"}
    for label in data.LABELS:
        assert set(first[f"{label}-{g}"] for g in range(12)) == {"train", "val", "test"}


def test_requires_independent_groups():
    with pytest.raises(data.DatasetError, match="independent capture groups"):
        data.split_groups_by_label([{"label": "cardboard", "capture_group": "x"}],
                                   ("train", "val", "test"))


def test_real_directory_ingestion_with_dummy_test_images(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    processed.mkdir()
    mapping = tmp_path / "label_map.json"
    mapping.write_text(json.dumps({"global": {label: label for label in data.LABELS},
                                  "sources": {}}))
    monkeypatch.setattr(data, "RAW_DATA", raw)
    monkeypatch.setattr(data, "PROCESSED", processed)
    monkeypatch.setattr(data, "LABEL_MAP", mapping)
    # Visually similar but distinct candidates are never auto-removed by the pipeline.
    monkeypatch.setattr(data, "near_same_image", lambda *_args, **_kwargs: pytest.fail("near-duplicate auto-removal attempted"))
    rng = np.random.default_rng(154)
    for label in data.LABELS:
        for n in range(12):
            folder = raw / "public" / "synthetic_test_source" / label
            folder.mkdir(parents=True, exist_ok=True)
            pixels = rng.integers(0, 255, size=(48, 48, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(folder / f"sample_{n}.png")
        for n in range(6):
            folder = raw / "local" / label / f"session_{n}"
            folder.mkdir(parents=True, exist_ok=True)
            pixels = rng.integers(0, 255, size=(48, 48, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(folder / "pic.png")
    # Duplicate plus corrupt image are excluded with documented reasons.
    a = raw / "public" / "synthetic_test_source" / "cardboard" / "sample_0.png"
    (a.parent / "duplicate.png").write_bytes(a.read_bytes())
    (a.parent / "bad.png").write_bytes(b"not an image")
    summary = data.prepare(seed=42)
    frame = pd.read_csv(processed / "split_manifest.csv")
    log = pd.read_csv(processed / "cleaning_log.csv")
    assert summary["total_removed_images"] >= 2
    assert "exact_duplicate" in set(log.reason)
    assert "corrupt_or_unreadable" in set(log.reason)
    assert "local_test" in set(frame.split)
    assert set(frame.split) >= {"train", "val", "test", "local_test"}
    assert all(g.split.nunique() == 1 for _, g in frame.groupby("capture_group"))
    assert (processed / "dataset_examples.png").exists()
    assert (processed / "source_summary.csv").exists()
    assert all(x in summary["class_counts"] for x in data.LABELS)

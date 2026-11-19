import pandas as pd

from waste_ai import binary_data
from waste_ai.metrics import summarize_predictions


def test_binary_manifest_group_splits_and_known_ambiguous_exclusion(tmp_path, monkeypatch):
    rows = []
    for source_label, mapped, source in [
        ("cardboard", "recyclable", "public/trashnet"),
        ("trash", "non_recyclable", "public/trashnet"),
    ]:
        for index in range(20):
            filename = "trash100.jpg" if source_label == "trash" and index == 0 else f"{source_label}{index}.jpg"
            rows.append({
                "path": f"/tmp/{source}/{source_label}/{filename}",
                "source": source,
                "label": source_label,
                "capture_group": f"{source}/{source_label}/{index}",
                "sha256": f"{source_label}-{index}",
            })
    source = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    monkeypatch.setattr(binary_data, "INPUT", source)
    monkeypatch.setattr(binary_data, "OUTPUT", tmp_path / "binary.csv")
    monkeypatch.setattr(binary_data, "PROCESSED", tmp_path)
    prepared, summary = binary_data.prepare_binary_data(seed=17)
    assert summary["n_excluded_known_ambiguous"] == 1
    assert "trash100.jpg" not in " ".join(prepared.path)
    assert set(prepared.binary_label) == {"non_recyclable", "recyclable"}
    assert set(prepared.groupby("capture_group").split.nunique()) == {1}
    assert all(set(group.binary_label) == {"non_recyclable", "recyclable"}
               for _, group in prepared.groupby("split"))


def test_binary_softmax_is_not_treated_as_six_class():
    report = summarize_predictions([0, 1], [[.8, .2], [.1, .9]])
    assert report["n_images"] == 2
    assert report["class_labels"] == ["non_recyclable", "recyclable"]
    assert report["accuracy"] == 1

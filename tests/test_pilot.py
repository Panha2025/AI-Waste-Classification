import pandas as pd

from waste_ai import pilot


def test_stratified_pilot_is_deterministic_balanced_and_quarantines_near_pairs(tmp_path, monkeypatch):
    rows = []
    for label, categories in (("recyclable", ("paper", "plastic")),
                              ("non_recyclable", ("trash",))):
        for source in ("source_a", "source_b"):
            for category in categories:
                for n in range(8):
                    ident = f"{source}-{category}-{n}"
                    rows.append({"source": source, "original_category": category,
                                 "original_path": f"origin/{ident}.jpg", "sha256": ident,
                                 "proposed_binary_class": label,
                                 "staged_path": f"data/raw/{source}/{category}/{ident}.jpg"})
    candidate_path = tmp_path / "candidate.csv"
    near_path = tmp_path / "near.csv"
    pd.DataFrame(rows).to_csv(candidate_path, index=False)
    pd.DataFrame([{"path": "origin/source_a-paper-0.jpg",
                   "candidate_duplicate_of": "origin/source_b-paper-1.jpg"}]).to_csv(near_path, index=False)
    monkeypatch.setattr(pilot, "CANDIDATES", candidate_path)
    monkeypatch.setattr(pilot, "NEAR_REVIEW", near_path)
    monkeypatch.setattr(pilot, "QUEUE", tmp_path / "queue.csv")
    monkeypatch.setattr(pilot, "SUMMARY", tmp_path / "summary.csv")
    monkeypatch.setattr(pilot, "EXCLUDED", tmp_path / "excluded.csv")
    monkeypatch.setattr(pilot, "PROCESSED", tmp_path)

    first, status = pilot.build_pilot_queue(target_per_class=6, seed=42)
    second, _ = pilot.build_pilot_queue(target_per_class=6, seed=42)
    assert first.sha256.tolist() == second.sha256.tolist()
    assert first.proposed_binary_class.value_counts().to_dict() == {"recyclable": 6, "non_recyclable": 6}
    assert first.sha256.nunique() == len(first)
    assert not first.original_path.isin({"origin/source_a-paper-0.jpg", "origin/source_b-paper-1.jpg"}).any()
    assert status["near_duplicate_candidates_excluded"] == 2
    assert set(first.source) == {"source_a", "source_b"}

import pandas as pd
import pytest

from waste_ai import label_review


def test_human_decision_exports_only_explicit_binary_labels(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate_manifest.csv"
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    pd.DataFrame([
        {"sha256": "a" * 64, "staged_path": str(image_a), "source": "s1",
         "original_category": "paper", "proposed_binary_class": "recyclable"},
        {"sha256": "b" * 64, "staged_path": str(image_b), "source": "s2",
         "original_category": "trash", "proposed_binary_class": "non_recyclable"},
    ]).to_csv(candidate, index=False)
    decisions = tmp_path / "reviews.csv"
    verified = tmp_path / "verified.csv"
    monkeypatch.setattr(label_review, "CANDIDATES", candidate)
    monkeypatch.setattr(label_review, "DECISIONS", decisions)
    monkeypatch.setattr(label_review, "VERIFIED_EXPORT", verified)

    label_review._save_decision({"id": "a" * 64, "decision": "uncertain", "note": "damaged"})
    assert pd.read_csv(verified).empty
    label_review._save_decision({"id": "b" * 64, "decision": "non_recyclable", "reviewer": "PK"})
    exported = pd.read_csv(verified)
    assert exported.id.tolist() == ["b" * 64]
    assert exported.review_status.tolist() == ["human_verified"]


def test_review_rejects_unknown_candidate_and_label(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate_manifest.csv"
    pd.DataFrame([{"sha256": "a", "staged_path": "a.jpg", "source": "s",
                   "original_category": "paper", "proposed_binary_class": "recyclable"}]).to_csv(candidate, index=False)
    monkeypatch.setattr(label_review, "CANDIDATES", candidate)
    monkeypatch.setattr(label_review, "DECISIONS", tmp_path / "reviews.csv")
    monkeypatch.setattr(label_review, "VERIFIED_EXPORT", tmp_path / "verified.csv")
    with pytest.raises(ValueError, match="Unknown candidate"):
        label_review._save_decision({"id": "missing", "decision": "recyclable"})
    with pytest.raises(ValueError, match="Invalid review decision"):
        label_review._save_decision({"id": "a", "decision": "banana"})


def test_candidate_queue_marks_folder_labels_provisional_and_ai_unavailable(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate_manifest.csv"
    pd.DataFrame([{"sha256": "a", "staged_path": "data/raw/public/s/paper.jpg", "source": "s",
                   "original_path": "data/source/paper.jpg", "original_category": "paper",
                   "proposed_binary_class": "recyclable"}]).to_csv(candidate, index=False)
    monkeypatch.setattr(label_review, "CANDIDATES", candidate)
    monkeypatch.setattr(label_review, "DECISIONS", tmp_path / "reviews.csv")
    monkeypatch.setattr(label_review, "AI_SUGGESTIONS", tmp_path / "no_ai.csv")
    row = label_review._rows()[0]
    assert row["proposed_binary_class"] == "recyclable"
    assert row["ai_status"] == "unavailable_no_local_model"
    assert row["ai_confidence"] is None


def test_imported_model_suggestion_is_unverified_and_confidence_bounded(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate_manifest.csv"
    pd.DataFrame([{"sha256": "a", "staged_path": "a.jpg", "source": "s",
                   "original_path": "a.jpg", "original_category": "paper",
                   "proposed_binary_class": "recyclable"}]).to_csv(candidate, index=False)
    suggestions = tmp_path / "ai.csv"
    suggestions.write_text("id,ai_suggestion,confidence,reason\na,recyclable,0,object cue\n")
    monkeypatch.setattr(label_review, "CANDIDATES", candidate)
    monkeypatch.setattr(label_review, "DECISIONS", tmp_path / "reviews.csv")
    monkeypatch.setattr(label_review, "AI_SUGGESTIONS", suggestions)
    row = label_review._rows()[0]
    assert row["ai_suggestion"] == "recyclable"
    assert row["ai_confidence"] == 0
    assert row["ai_status"] == "ai_suggested_unverified"


def test_pilot_progress_counts_binary_approvals_uncertain_and_pending(tmp_path, monkeypatch):
    queue_path = tmp_path / "pilot.csv"
    pd.DataFrame({"sha256": ["a", "b", "c", "d"]}).to_csv(queue_path, index=False)
    (tmp_path / "pilot_review_queue_status.json").write_text('{"target_per_class": 2}')
    decisions = [
        {"id": "a", "review_decision": "recyclable"},
        {"id": "b", "review_decision": "uncertain"},
        {"id": "c", "review_decision": "non_recyclable"},
        {"id": "d", "review_decision": ""},
    ]
    monkeypatch.setattr(label_review, "PILOT_QUEUE", queue_path)
    monkeypatch.setattr(label_review, "PROCESSED", tmp_path)
    monkeypatch.setattr(label_review, "_rows", lambda: decisions)
    progress = label_review._pilot_progress()
    assert progress == {"total": 4, "reviewed": 3,
                        "approved": {"recyclable": 1, "non_recyclable": 1},
                        "uncertain": 1, "pending": 1, "target_per_class": 2}

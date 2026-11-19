import numpy as np

from waste_ai.binary_select_evaluate import _best_validation_threshold


def test_validation_threshold_prioritizes_balance_and_keeps_recyclable_recall():
    truth = np.array([0, 0, 1, 1])
    probabilities = np.array([[.70, .30], [.80, .20], [.30, .70], [.20, .80]])
    threshold, metrics, sweep, relaxed = _best_validation_threshold(truth, probabilities)
    assert not relaxed
    assert metrics["balanced_accuracy"] == 1
    assert metrics["non_recyclable_recall"] == 1
    assert metrics["recyclable_recall"] == 1
    assert threshold > .30
    assert sweep.meets_recyclable_recall_floor.any()


def test_validation_threshold_marks_when_recyclable_recall_floor_cannot_be_met():
    truth = np.array([0, 1, 1, 1])
    probabilities = np.array([[.999, .001], [.999, .001], [.999, .001], [.999, .001]])
    _, metrics, _, relaxed = _best_validation_threshold(truth, probabilities)
    assert relaxed
    assert metrics["recyclable_recall"] < .80

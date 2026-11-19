import math
import pytest
from waste_ai.metrics import (classify_probability, classify_probabilities,
                              classify_binary_probabilities, summarize_predictions)
from waste_ai.inference import Prediction, format_result


def test_probability_classification_and_confidence():
    assert classify_probability(0.90) == ("recyclable", pytest.approx(0.90))
    assert classify_probability(0.10) == ("non_recyclable", pytest.approx(0.90))
    assert classify_probability(0.5)[0] == "recyclable"


@pytest.mark.parametrize("p", [-0.1, 1.01, float("nan"), float("inf")])
def test_invalid_probabilities(p):
    with pytest.raises(ValueError):
        classify_probability(p)


def test_confusion_matrix_and_class_wise_metrics():
    truth = [0, 0, 1, 1]
    probabilities = [0.1, 0.8, 0.9, 0.2]
    result = summarize_predictions(truth, probabilities)
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["macro_f1"] == pytest.approx(0.5)
    assert result["balanced_accuracy"] == pytest.approx(0.5)
    assert result["n_images"] == 4
    assert summarize_predictions([], [])["n_images"] == 0


def test_low_confidence_warning_and_normal_message():
    uncertain = format_result(Prediction("plastic", .51, "custom_cnn", (.01,.01,.01,.01,.51,.45)))
    assert "Low-confidence" in uncertain
    assert "51.0%" in uncertain
    strong = format_result(Prediction("glass", .88, "mobilenetv2", (.02,.88,.02,.02,.03,.03)))
    assert "Predicted category: Glass" in strong
    assert "88.0%" in strong


def test_six_class_probabilities_and_metrics():
    assert classify_probabilities([.01, .8, .03, .04, .06, .06]) == ("glass", pytest.approx(.8))
    truth = [0, 1, 2, 3, 4, 5]
    probabilities = [[.8,.1,.02,.02,.03,.03],
                     [.1,.8,.02,.02,.03,.03],
                     [.1,.1,.7,.02,.03,.03],
                     [.1,.1,.02,.7,.04,.02],
                     [.1,.1,.02,.02,.7,.04],
                     [.1,.1,.02,.02,.03,.73]]
    result = summarize_predictions(truth, probabilities)
    assert result["class_labels"] == ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    assert result["accuracy"] == 1
    assert result["balanced_accuracy"] == 1
    assert result["confusion_matrix"] == [[1 if i == j else 0 for j in range(6)] for i in range(6)]


def test_direct_binary_softmax_metrics_and_messages():
    assert classify_binary_probabilities([.15, .85]) == ("recyclable", pytest.approx(.85))
    assert classify_binary_probabilities([.8, .2]) == ("non_recyclable", pytest.approx(.8))
    assert classify_binary_probabilities([.55, .45], threshold=.4)[0] == "recyclable"
    result = summarize_predictions([0, 0, 1, 1], [[.8,.2],[.3,.7],[.1,.9],[.6,.4]])
    assert result["class_labels"] == ["non_recyclable", "recyclable"]
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    message = format_result(Prediction("recyclable", .93, "binary_mobilenetv2", (.07,.93)))
    assert "♻️ Recyclable" in message
    assert "93.0%" in message
    assert "local acceptance varies" in message.lower()

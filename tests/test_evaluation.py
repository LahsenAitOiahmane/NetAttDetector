import numpy as np
import pytest

from netattdetector.evaluation import choose_threshold, metrics


def test_threshold_satisfies_constraints_and_fallback_is_explicit():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    threshold, feasible = choose_threshold(y, scores, 1.0, 0.0)
    assert feasible
    assert threshold == 0.8
    result = metrics(y, scores, threshold)
    assert result["false_positive_rate"] == 0
    assert result["attack_recall"] == 1
    _, feasible = choose_threshold(y, np.zeros(5), 1.0, 0.0)
    assert not feasible


def test_confusion_matrix_and_false_alarms():
    result = metrics([0, 0, 1, 1], [0.1, 0.8, 0.2, 0.9], 0.5)
    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert result["false_alarms_per_10000_benign_flows"] == 5000
    assert result["attack_precision"] == 0.5


def test_single_class_validation_rejected():
    with pytest.raises(ValueError, match="both binary classes"):
        choose_threshold([1, 1], [0.2, 0.7])

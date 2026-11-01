import numpy as np
import pandas as pd
import pytest

from netattdetector.native_errors import compare_predictions, error_slices, load_config


def test_uid_context_uses_highest_segment_and_ties_keep_first():
    frame = pd.DataFrame(
        {
            "group_hash": ["a", "a", "b", "c", "c"],
            "label": [0, 0, 1, 1, 1],
            "detail": ["-", "-", "DDoS", "C&C", "C&C"],
            "protocol": [17, 6, 6, 6, 17],
            "dst_port": [53, 443, 1883, 22, 53],
            "src_port": [100, 55000, 3000, 3000, 3000],
            "src2dst_packets": [1, 1, 2, 1, 1],
            "dst2src_packets": [0, 0, 0, 0, 0],
            "bidirectional_duration_ms": [0, 0, 1, 0, 0],
        }
    )
    result = error_slices(frame, [0.2, 0.9, 0.1, 0.8, 0.8], 0.5)
    assert result["connections"] == 3
    assert result["slices"]["protocol"]["TCP"] == {
        "connections": 3,
        "tn": 0,
        "fp": 1,
        "fn": 1,
        "tp": 1,
    }
    assert result["slices"]["observed_destination_service"]["https_443"]["fp"] == 1
    assert result["slices"]["observed_destination_service"]["ssh_22"]["tp"] == 1
    assert "dns_53" not in result["slices"]["observed_destination_service"]
    assert result["score_distributions"]["tn"]["score_quantiles_min_p25_p50_p75_max"] is None
    for table in result["slices"].values():
        assert sum(cell["connections"] for cell in table.values()) == 3
        assert sum(cell["fp"] for cell in table.values()) == 1
        assert sum(cell["fn"] for cell in table.values()) == 1
    assert "group_hash" not in result and "scores" not in result


def test_score_tolerance_and_decision_stability_are_separate():
    result = compare_predictions([0.1, 0.9], [0.100002, 0.900002], 0.5)
    assert result["max_absolute_score_difference"] > 1e-7
    assert result["frozen_threshold_decision_differences"] == 0
    assert not result["scores_exactly_identical"]
    flipped = compare_predictions([0.49, 0.9], [0.51, 0.9], 0.5)
    assert flipped["frozen_threshold_decision_differences"] == 1
    assert compare_predictions([0.1, 0.9], [0.1, 0.9], 0.5)["scores_exactly_identical"]
    with pytest.raises(ValueError):
        compare_predictions([0.1, 0.2], [np.nan, 0.2], 0.5)


def test_registered_diagnostic_has_three_fits_and_no_deployment():
    config = load_config("configs/native-errors.json")
    assert config["training_threads"] == [2, 1, 1]
    assert config["maximum_fits"] == 3 and config["deployment_authorized"] is False

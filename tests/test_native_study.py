import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from netattdetector.inference import load_bundle
from netattdetector.measurement import CORE_FEATURES
from netattdetector.native_study import (
    NativeFeatures,
    connection_scores,
    evaluate_scores,
    feature_overlap,
    fit_one,
    load_study_config,
    threshold_connections,
    training_weights,
)


def frames():
    base = json.loads(Path("tests/fixtures/pcap/expected.json").read_text())["rows"][0]
    result = {}
    for split in ("train", "calibration", "validation"):
        rows = []
        for index in range(40):
            label = index % 2
            rows.append(
                {
                    **{name: base[name] for name in CORE_FEATURES},
                    "src_port": 10000 + index,
                    "src2dst_bytes": 32 + label * 100,
                    "label": label,
                    "detail": "Attack" if label else "-",
                    "group_hash": f"{split}-{index}",
                }
            )
        result[split] = pd.DataFrame(rows)
    return result


def test_derived_zero_duration_features_are_finite_and_ports_excluded():
    data = frames()["train"][CORE_FEATURES].iloc[:1].copy()
    data["bidirectional_duration_ms"] = 0
    transform = NativeFeatures("behavioral_derived")
    values = transform.fit_transform(data)
    assert values.shape == (1, 16) and np.isfinite(values).all()
    names = transform.get_feature_names_out().tolist()
    assert "src_port" not in names and "dst_port" not in names
    assert values[0, names.index("smoothed_packets_per_second")] == 2000
    assert values[0, names.index("reverse_packet_fraction")] == 0.5
    assert values[0, names.index("mean_ip_bytes_per_packet")] == 34


@pytest.mark.parametrize("case", ["metadata", "reordered", "negative", "fractional", "nan"])
def test_model_inputs_cannot_silently_change_contract(case):
    data = frames()["train"][CORE_FEATURES].copy()
    if case == "metadata":
        data["label"] = 1
    elif case == "reordered":
        data = data[data.columns[::-1]]
    elif case == "negative":
        data.loc[0, "src_port"] = -1
    elif case == "fractional":
        data["src_port"] = data["src_port"].astype(float) + 0.5
    else:
        data["src_port"] = np.nan
    with pytest.raises(ValueError):
        NativeFeatures().transform(data)


def test_connection_class_weights_do_not_favor_long_segments():
    frame = pd.DataFrame(
        {"group_hash": ["a", "a", "b", "c", "c", "c", "d"], "label": [0, 0, 0, 1, 1, 1, 1]}
    )
    weights = training_weights(frame, "connection_class_balanced")
    assert weights.mean() == pytest.approx(1)
    totals = pd.Series(weights).groupby(frame["group_hash"]).sum()
    assert totals["a"] == pytest.approx(totals["b"])
    assert totals["c"] == pytest.approx(totals["d"])
    assert weights[frame["label"] == 0].sum() == pytest.approx(weights[frame["label"] == 1].sum())
    assert training_weights(frame, "flow_natural") is None


def test_connection_max_scores_count_benign_long_connection_once():
    frame = pd.DataFrame(
        {"group_hash": ["a", "a", "b"], "label": [0, 0, 1], "detail": ["-", "-", "Attack"]}
    )
    result = evaluate_scores(frame, np.array([0.1, 0.9, 0.8]), 0.5)
    assert result["connections"]["connections"] == 2
    assert result["connections"]["confusion_matrix"] == {"tn": 0, "fp": 1, "fn": 0, "tp": 1}
    assert result["flows"]["false_positive_rate"] == 0.5
    assert result["connections"]["false_positive_rate"] == 1
    frame.loc[1, "label"] = 1
    with pytest.raises(ValueError, match="Inconsistent"):
        connection_scores(frame, np.array([0.1, 0.9, 0.8]))


def test_threshold_includes_whole_score_ties_and_reports_fallback():
    screen = {
        "minimum_precision": 1,
        "maximum_fpr": 0,
        "minimum_recall": 0.8,
        "minimum_true_positive_connections": 1,
    }
    feasible = threshold_connections([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.8], screen)
    assert feasible == {
        "threshold": 0.8,
        "calibration_precision_fpr_support_feasible": True,
        "used_f1_fallback": False,
    }
    impossible = threshold_connections([0, 0, 1, 1], [0.1, 0.9, 0.8, 0.7], screen)
    assert impossible == {
        "threshold": 0.7,
        "calibration_precision_fpr_support_feasible": False,
        "used_f1_fallback": True,
    }
    screen["minimum_true_positive_connections"] = 25
    assert threshold_connections([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.8], screen)["used_f1_fallback"]


@pytest.mark.parametrize(
    "model,policy",
    [
        (name, policy)
        for name in ("dummy", "logistic", "histogram", "extra_trees")
        for policy in ("native11", "behavioral_derived")
    ],
)
def test_pipeline_fits_training_only_and_reload_matches(tmp_path, model, policy):
    config = load_study_config("configs/native-study.json")
    data = frames()
    fitted, scores, report = fit_one(
        {
            "id": "synthetic",
            "model": model,
            "features": policy,
            "weights": "connection_class_balanced",
        },
        data,
        config,
    )
    assert len(scores["validation"]) == 40
    assert report["metrics"]["validation"]["connections"]["connections"] == 40
    if model == "logistic":
        expected = np.log1p(NativeFeatures(policy).transform(data["train"][CORE_FEATURES])).mean(
            axis=0
        )
        assert np.allclose(fitted.named_steps["scale"].mean_, expected)
    path = tmp_path / "native.joblib"
    joblib.dump(fitted, path)
    with threadpool_limits(limits=1):
        actual = joblib.load(path).predict_proba(data["validation"][CORE_FEATURES])[:, 1]
    assert np.array_equal(scores["validation"], actual)


def test_feature_policy_audit_reveals_new_collisions_without_deleting_rows():
    data = frames()
    original = feature_overlap(data, "native11")
    reduced = feature_overlap(data, "behavioral_derived")
    assert original["unique_vectors"] == 40 and reduced["unique_vectors"] == 2
    assert original["pairwise_overlap"][0]["shared_vectors"] == 40
    assert reduced["pairwise_overlap"][0]["shared_vectors"] == 2
    assert len(data["train"]) == 40


def test_native_bundle_does_not_load_as_uq_model(tmp_path):
    path = tmp_path / "native.joblib"
    joblib.dump({"format": "native-diagnostic-v1", "schema": "nfstream-core-v1"}, path)
    with pytest.raises(ValueError, match="Unsupported model artifact"):
        load_bundle(path)

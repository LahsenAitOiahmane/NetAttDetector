import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from netattdetector.measurement import CORE_FEATURES
from netattdetector.native_study import NativeFeatures, feature_overlap, fit_one
from netattdetector.statistical import FEATURES
from netattdetector.statistical_study import (
    StatisticalFeatures,
    load_config,
    statistical_model,
)


def frames():
    rows = json.loads(Path("tests/fixtures/pcap/statistical-expected.json").read_text())["rows"]
    return {
        split: pd.DataFrame(
            [
                {
                    **{name: rows[index % 5][name] for name in FEATURES},
                    "label": index % 2,
                    "detail": "Attack" if index % 2 else "-",
                    "group_hash": f"{split}-{index}",
                }
                for index in range(50)
            ]
        )
        for split in ("train", "calibration", "validation")
    }


def test_core_control_has_identical_numeric_values_and_no_port_policy_is_exact():
    data = frames()["train"]
    assert np.array_equal(
        StatisticalFeatures("core11_control").transform(data[FEATURES]),
        NativeFeatures("native11").transform(data[CORE_FEATURES]),
    )
    reduced = StatisticalFeatures("statistics25_no_ports")
    assert reduced.transform(data[FEATURES]).shape == (50, 25)
    assert reduced.get_feature_names_out().tolist() == FEATURES[2:]
    with pytest.raises(ValueError, match="27-field"):
        reduced.transform(data[CORE_FEATURES])


@pytest.mark.parametrize("policy", ["core11_control", "statistics27", "statistics25_no_ports"])
@pytest.mark.parametrize("model", ["logistic", "histogram"])
def test_statistical_fit_reuses_calibration_logic_and_reload_is_exact(tmp_path, policy, model):
    data = frames()
    config = load_config("configs/statistical-study.json")
    fitted, scores, report = fit_one(
        {
            "id": "synthetic",
            "model": model,
            "features": policy,
            "weights": "connection_class_balanced",
        },
        data,
        config,
        threads=1,
        input_features=FEATURES,
        model_factory=statistical_model,
    )
    assert len(scores["validation"]) == 50
    assert report["metrics"]["validation"]["connections"]["connections"] == 50
    path = tmp_path / "model.joblib"
    joblib.dump(fitted, path)
    with threadpool_limits(limits=1):
        replay = joblib.load(path).predict_proba(data["validation"][FEATURES])[:, 1]
    assert np.array_equal(replay, scores["validation"])


def test_float32_audit_distinguishes_numeric_roundoff_from_new_information():
    data = frames()
    data["train"].loc[0, "bidirectional_mean_ps"] += 1e-8
    high = feature_overlap(
        data,
        "statistics27",
        input_features=FEATURES,
        transformer=StatisticalFeatures(),
        data_type="<f8",
    )
    low = feature_overlap(
        data,
        "statistics27",
        input_features=FEATURES,
        transformer=StatisticalFeatures(),
        data_type="<f4",
    )
    assert high["unique_vectors"] == low["unique_vectors"] + 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("bidirectional_syn_packets", 999),
        ("bidirectional_mean_ps", -1),
        ("src_port", 1.5),
        ("bidirectional_max_ps", 1),
    ],
)
def test_corrupt_measurements_do_not_enter_models(field, value):
    data = frames()["train"][FEATURES].copy()
    data[field] = data[field].astype(float)
    data.loc[0, field] = value
    with pytest.raises(ValueError):
        StatisticalFeatures().transform(data)


def test_registered_study_keeps_fixed_threads_and_no_deployment():
    config = load_config("configs/statistical-study.json")
    assert config["threads"] == 1 and config["maximum_fits"] == 15
    assert config["deployment_authorized"] is False

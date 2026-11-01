import copy

import joblib
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from netattdetector.context_cohort import FEATURES
from netattdetector.context_study import (
    ContextFeatures,
    context_model,
    context_selection_key,
    family_guard,
    load_config,
    synthetic_probes,
)
from netattdetector.native_study import fit_one
from netattdetector.statistical import FEATURES as STAT_FEATURES
from netattdetector.statistical_study import StatisticalFeatures


def frames():
    probes = synthetic_probes()
    result = {}
    for split in ("train", "calibration", "validation"):
        part = probes.copy()
        part["label"] = np.arange(len(part)) % 2
        part["detail"] = ["DDoS" if label else "-" for label in part["label"]]
        part["group_hash"] = [f"{split}-{index}" for index in range(len(part))]
        result[split] = part
    return result


def test_contract_control_and_no_port_are_exact_subsets():
    probes = synthetic_probes()
    assert probes.shape == (45, 36)
    assert np.array_equal(
        ContextFeatures("statistics27_control").transform(probes),
        StatisticalFeatures().transform(probes[STAT_FEATURES]),
    )
    reduced = ContextFeatures("context34_no_ports")
    assert reduced.get_feature_names_out().tolist() == FEATURES[2:]
    assert reduced.transform(probes).shape == (45, 34)
    with pytest.raises(ValueError, match="36-field"):
        reduced.transform(probes[STAT_FEATURES])


@pytest.mark.parametrize(
    "field,value",
    [
        ("src_flow_starts_1s", 999),
        ("dst_flow_starts_60s", 0),
        ("src_distinct_destinations_60s", 100_001),
        ("dst_distinct_sources_60s", 1.5),
        ("src_flow_starts_10s", float("nan")),
    ],
)
def test_invalid_context_is_rejected(field, value):
    probes = synthetic_probes().astype(float)
    probes.loc[8, field] = value
    with pytest.raises(ValueError):
        ContextFeatures().transform(probes)


@pytest.mark.parametrize("model", ["logistic", "histogram"])
@pytest.mark.parametrize("policy", ["statistics27_control", "context36", "context34_no_ports"])
def test_fixed_thread_fit_and_reload_parity(tmp_path, model, policy):
    data = frames()
    fitted, scores, report = fit_one(
        {
            "id": "synthetic",
            "model": model,
            "features": policy,
            "weights": "connection_class_balanced",
        },
        data,
        load_config(),
        threads=1,
        input_features=FEATURES,
        model_factory=context_model,
    )
    path = tmp_path / "synthetic.joblib"
    joblib.dump(fitted, path)
    with threadpool_limits(limits=1):
        actual = joblib.load(path).predict_proba(data["validation"][FEATURES])[:, 1]
    assert np.array_equal(actual, scores["validation"])
    assert report["metrics"]["validation"]["connections"]["connections"] == 45


def test_family_guard_rejects_missing_small_or_missed_minority_and_ranks_qualified_first():
    config = load_config()
    families = {
        name: {"attack_connections": 25, "attack_recall": 0.8}
        for name in config["family_screen"]["required_validation_families"]
    }
    report = {
        "passes_calibration_and_validation_screen": True,
        "metrics": {
            "validation": {
                "families": families,
                "connections": {
                    "attack_f1": 0.9,
                    "attack_recall": 0.9,
                    "attack_precision": 0.99,
                    "false_positive_rate": 0.001,
                },
            }
        },
        "artifact_bytes": 100,
        "id": "qualified",
    }
    good = family_guard(copy.deepcopy(report), config["family_screen"])
    assert good["passes_binary_and_family_screen"]
    assert good["family_screen"]["checks"]["DDoS"]["detected_connections"] == 20
    for replacement in (
        {},
        {"attack_connections": 24, "attack_recall": 1.0},
        {"attack_connections": 1000, "attack_recall": 0.79},
    ):
        bad = copy.deepcopy(report)
        bad["metrics"]["validation"]["families"]["DDoS"] = replacement
        bad["metrics"]["validation"]["connections"]["attack_f1"] = 0.999
        bad = family_guard(bad, config["family_screen"])
        assert not bad["passes_binary_and_family_screen"]
        assert context_selection_key(good) < context_selection_key(bad)


def test_modified_registration_is_rejected(tmp_path):
    import json

    config = load_config()
    config["threads"] = 2
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="registered"):
        load_config(path)


def test_complete_bounded_runner_retains_only_synthetic_transfer_payload(tmp_path, monkeypatch):
    import json

    from netattdetector import context_study
    from netattdetector.private_artifacts import generate_recipient, recover_model

    recipient = tmp_path / "recipient.json"
    generate_recipient(tmp_path / "keys", recipient)
    config = load_config()
    config["artifact_recipient"] = str(recipient)
    monkeypatch.setattr(context_study, "load_config", lambda _: config)
    monkeypatch.setattr(context_study, "load_frames", lambda *_: frames())
    monkeypatch.setattr(
        context_study,
        "verify_context_inputs",
        lambda *_: (
            {"audit": {}},
            {"sidecar": {"sha256": "synthetic"}},
            {"database": {"sha256": "synthetic"}, "exporter_version": "synthetic"},
        ),
    )
    output = tmp_path / "study"
    result = context_study.run_context_study(tmp_path, tmp_path, tmp_path, output)
    assert result["completed_fits"] == 15
    assert result["repeatability_screen_met"]
    assert not result["deployment_authorized"]
    freeze = json.loads((output / "freeze.json").read_text())
    model_bytes, probes = recover_model(
        output / "selected.nadenc", tmp_path / "keys/recipient-v1.key", freeze
    )
    assert len(model_bytes) > 0
    assert len(probes["rows"]) == 45
    assert set(probes) == {"features", "rows", "scores", "scope"}
    assert all(
        "group_hash" not in name and name not in ("label", "detail") for name in probes["features"]
    )
    with pytest.raises(FileExistsError):
        context_study.run_context_study(tmp_path, tmp_path, tmp_path, output)

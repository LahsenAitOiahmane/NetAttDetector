import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from netattdetector.audit import prepare_sources
from netattdetector.data import FEATURES, fingerprints
from netattdetector.multisource import development_split
from netattdetector.stability import (
    EXPORT_RATES,
    GROUP_FEATURES,
    array_digest,
    canonical_matrix,
    canonical_weights,
    compare_studies,
    feature_policy,
    load_study_config,
    run_stability_study,
    study_groups,
    validation_screen,
)


def test_ablation_groups_keep_rate_only_variants_together(flows):
    frame = flows[FEATURES].astype(np.float32)
    frame["__label"] = flows.Label
    frame["__source"] = "a"
    twin = frame.iloc[[0]].copy()
    twin["SRC_TO_DST_SECOND_BYTES"] += 100
    twin["__label"] = 1 - twin["__label"]
    frame = pd.concat([frame, twin], ignore_index=True)
    frame["__full_hash"] = fingerprints(frame)
    frame = study_groups(frame)
    assert len(frame) == 601
    assert frame["__hash"].iloc[0] == frame["__hash"].iloc[-1]
    assert len(GROUP_FEATURES) == 30
    assert not set(GROUP_FEATURES) & EXPORT_RATES
    for policy in ("full", "no_ttl", "no_export_rates", "previous_reduced"):
        assert set(GROUP_FEATURES) <= set(feature_policy(policy))
    parts = development_split(frame, 42)
    assert any((part["__hash"] == frame["__hash"].iloc[0]).sum() == 2 for part in parts)


def test_canonical_payloads_and_weights(flows):
    frame = flows[FEATURES].astype(np.float32)
    frame["__source"] = ["a"] * 100 + ["b"] * 500
    frame["__label"] = flows.Label
    matrix = canonical_matrix(frame, FEATURES).to_numpy()
    assert matrix.flags.c_contiguous and matrix.dtype == np.float32
    assert array_digest(matrix) == array_digest(np.asfortranarray(matrix))
    assert array_digest(matrix) != array_digest(matrix[::-1])
    weights = canonical_weights(frame)
    assert weights.dtype == np.float32
    np.testing.assert_allclose(weights[:100].sum(), weights[100:].sum(), rtol=1e-6)


def test_development_screen_rejects_low_recall_and_transformed_leakage():
    config = json.loads(Path("configs/stability.json").read_text())
    metric = {
        "attack_precision": 1.0,
        "attack_recall": 0.1,
        "conditional_wilson_95": {
            "attack_precision": [0.97, 1.0],
            "false_positive_rate": [0, 0.001],
        },
        "confusion_matrix": {"tp": 100, "fp": 0},
    }
    record = {
        "calibration_feasible": True,
        "transformed_overlap_rows": {
            "train_calibration": 0,
            "train_validation": 0,
            "calibration_validation": 0,
        },
        "by_source": {"a": metric},
    }
    assert not validation_screen(record, config)
    metric["attack_recall"] = 0.9
    assert validation_screen(record, config)
    record["transformed_overlap_rows"]["calibration_validation"] = 1
    assert not validation_screen(record, config)


def test_study_runs_without_holdout_access(tmp_path, flows):
    # Tiny genuine train/calibration/validation path; no CSE file is provided.
    pytest.importorskip("xgboost")
    paths = {}
    for i, source in enumerate(("nf-unsw-nb15-v2", "nf-ton-iot-v2")):
        frame = flows.copy()
        frame["OUT_BYTES"] += i * 1000
        frame["PROTOCOL"] = 6
        paths[source] = tmp_path / f"{source}.csv"
        frame.to_csv(paths[source], index=False)
    prepared = tmp_path / "prepared"
    prepare_sources(paths, prepared, rows_per_source=1000, buckets=4)
    config = json.loads(Path("configs/stability.json").read_text())
    config.update(
        split_seeds=[42],
        models=["hist"],
        feature_policies=["no_export_rates"],
        iterations=3,
        run_controls=False,
        run_source_transfer=False,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    result = run_stability_study(prepared, config_path, tmp_path / "study")
    assert len(result["records"]) == 1
    assert result["grouping"]["features"] == GROUP_FEATURES
    assert not any(result["records"][0]["transformed_overlap_rows"].values())
    assert result["summary"][0]["complete"]
    assert result["controls"] == []
    with pytest.raises(ValueError, match="fresh"):
        run_stability_study(prepared, config_path, tmp_path / "study")


def test_comparison_rejects_input_or_missing_control_mismatch(tmp_path):
    dirs = [tmp_path / "a", tmp_path / "b"]
    report = {
        "config": {"control_seed": 42},
        "config_sha256": "config",
        "source_sha256": {"source": "raw"},
        "partition_payloads": {"42": {"train": "signature"}},
    }
    for root in dirs:
        root.mkdir()
        (root / "study.json").write_text(json.dumps(report))
        np.savez(
            root / "control-scores.npz",
            scores=np.array([[0.1, 0.9]]),
            names=["full-1t:0"],
            validation_full_hash=[1, 2],
            validation_labels=[0, 1],
        )
    with pytest.raises(ValueError, match="eight registered controls"):
        compare_studies(*dirs)
    report["partition_payloads"]["42"]["train"] = "different"
    (dirs[1] / "study.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="inputs/order/labels/weights"):
        compare_studies(*dirs)


def test_registered_grid_limits(tmp_path):
    config = json.loads(Path("configs/stability.json").read_text())
    config["models"] = []
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="nonempty"):
        load_study_config(path)

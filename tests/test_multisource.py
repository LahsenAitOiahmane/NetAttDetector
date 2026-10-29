import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from netattdetector.audit import (
    STABLE_FEATURES,
    load_prepared,
    overlaps_development,
    prepare_sources,
)
from netattdetector.data import FEATURES, SCHEMA_VERSION, fingerprints, numeric_features
from netattdetector.evaluation import metrics, wilson_interval
from netattdetector.inference import load_bundle, predict_csv, set_inference_threads
from netattdetector.multisource import (
    development_split,
    evaluate_frozen,
    robust_threshold,
    source_weights,
    train_multisource,
)


@pytest.fixture
def development(tmp_path, flows):
    first = tmp_path / "nf-unsw-nb15-v2.csv"
    second = tmp_path / "nf-ton-iot-v2.csv"
    other = flows.copy()
    other["OUT_BYTES"] += 1000
    conflict = flows.iloc[[0]].copy()
    conflict["Label"] = 1 - conflict["Label"]
    conflict["Attack"] = np.where(conflict["Label"], "TestAttack", "Benign")
    pd.concat([flows, flows.iloc[[1]]], ignore_index=True).to_csv(first, index=False)
    pd.concat([other, conflict], ignore_index=True).to_csv(second, index=False)
    prepared = tmp_path / "prepared"
    report = prepare_sources(
        {"unsw": first, "ton": second}, prepared, rows_per_source=1000, chunk_rows=113, buckets=4
    )
    return prepared, report, first, second


def test_global_conflicts_and_duplicates(development):
    prepared, report, _, _ = development
    assert report["global_audit"]["full"]["rows"] == 1202
    assert report["global_audit"]["full"]["rows_in_conflicting_groups"] == 2
    assert report["global_audit"]["full"]["groups_shared_across_sources"] == 1
    assert report["sample_global_conflict_rows_removed"] == 2
    assert report["sample_repeated_full_rows_removed"] == 1
    frame, _ = load_prepared(prepared)
    assert frame["__hash"].is_unique
    assert len(frame) == 1199
    assert not list((prepared / "hash-index").glob("*.bin"))


def test_full_index_detects_rows_outside_sample(tmp_path, flows):
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    flows.to_csv(first, index=False)
    other = flows.copy()
    other["OUT_BYTES"] += 1000
    other.to_csv(second, index=False)
    root = tmp_path / "prepared"
    prepare_sources({"a": first, "b": second}, root, rows_per_source=100, buckets=4)
    sample, _ = load_prepared(root)
    values = numeric_features(flows)
    assert len(set(fingerprints(values, STABLE_FEATURES)) - set(sample["__hash"])) > 400
    assert overlaps_development(values, root / "hash-index", 4).all()
    values["OUT_BYTES"] += 5000
    assert not overlaps_development(values, root / "hash-index", 4).any()


def test_split_and_weighting_preserve_source_isolation(development):
    frame, _ = load_prepared(development[0])
    parts = development_split(frame, 42)
    assert sum(map(len, parts)) == len(frame)
    for i in range(3):
        for j in range(i):
            assert not set(parts[i]["__hash"]) & set(parts[j]["__hash"])
    for policy, columns in (("source", ["__source"]), ("source_class", ["__source", "__label"])):
        weighted = parts[0].assign(weight=source_weights(parts[0], policy))
        sums = weighted.groupby(columns)["weight"].sum()
        np.testing.assert_allclose(sums, sums.iloc[0])


def test_pooled_precision_cannot_hide_source_failure():
    frame = pd.DataFrame(
        {"__source": ["a"] * 100 + ["b"] * 100, "__label": [0] * 99 + [1] + [1] * 99 + [0]}
    )
    scores = np.array([0.1] * 98 + [0.8, 0.8] + [0.9] * 99 + [0.1])
    assert metrics(frame["__label"], scores, 0.5)["attack_precision"] > 0.98
    config = {
        "min_precision": 0.98,
        "precision_lower_bound": 0.5,
        "max_fpr": 0.2,
        "min_alerts_per_source": 1,
    }
    _, feasible = robust_threshold(frame, scores, config)
    assert not feasible


def test_wilson_interval_reports_uncertainty():
    low, high = wilson_interval(0, 100)
    assert low == 0
    assert 0.036 < high < 0.038
    assert wilson_interval(0, 0) is None


def test_reduced_label_ambiguity_is_retained(tmp_path, flows):
    other = flows.copy()
    other["OUT_BYTES"] += 1000
    ambiguous = flows.iloc[[0]].copy()
    ambiguous["L4_SRC_PORT"] += 1
    ambiguous["Label"] = 1 - ambiguous["Label"]
    ambiguous["Attack"] = np.where(ambiguous["Label"], "TestAttack", "Benign")
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    flows.to_csv(first, index=False)
    pd.concat([other, ambiguous], ignore_index=True).to_csv(second, index=False)
    root = tmp_path / "prepared"
    report = prepare_sources({"a": first, "b": second}, root, rows_per_source=1000, buckets=4)
    frame, _ = load_prepared(root)
    assert report["global_audit"]["stable"]["rows_in_conflicting_groups"] == 2
    assert report["sample_global_conflict_rows_removed"] == 0
    assert len(frame) == 1201
    assert (frame.groupby("__hash")["__label"].nunique() > 1).any()
    parts = development_split(frame, 42)
    for i in range(3):
        for j in range(i):
            assert not set(parts[i]["__hash"]) & set(parts[j]["__hash"])


def test_small_registered_experiment_and_single_holdout_access(tmp_path, development, flows):
    prepared, _, first, second = development
    config = json.loads(Path("configs/multisource.json").read_text())
    config.update(
        rows_per_source=1000,
        audit_buckets=4,
        min_precision=0.8,
        precision_lower_bound=0.6,
        max_fpr=0.2,
        min_alerts_per_source=2,
    )
    config["candidates"] = [
        dict(config["candidates"][0], iterations=10),
        dict(config["candidates"][1], iterations=10),
    ]
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(config))
    run = tmp_path / "run"
    freeze = train_multisource(prepared, protocol, run)
    assert freeze["budget"]["completed_fits"] == 2
    assert len(freeze["validation_leaderboard"]) == 4
    assert freeze["partitions"]["calibration"]["rows"] > 0
    bundle = load_bundle(run / "model.joblib")
    assert bundle["format_version"] == 2
    example = tmp_path / "flows.csv"
    flows.to_csv(example, index=False)
    assert predict_csv(run / "model.joblib", example, tmp_path / "pred.csv", threads=1)[
        "rows"
    ] == len(flows)
    # Exercise nested calibration thread control even if the uncalibrated variant won.
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    calibrated = CalibratedClassifierCV(FrozenEstimator(bundle["model"]), n_jobs=4)
    calibrated.fit(flows[bundle["features"]], flows["Label"])
    set_inference_threads(calibrated, 1)
    assert calibrated.n_jobs == 1
    heldout = flows.copy()
    heldout["OUT_BYTES"] += 5000
    heldout = pd.concat([heldout, pd.read_csv(first).iloc[[5]]], ignore_index=True)
    path = tmp_path / "nf-cse-cic-ids2018-v2.csv"
    heldout.to_csv(path, index=False)
    report = evaluate_frozen(run, prepared, path, tmp_path / "holdout.json")
    assert report["all_development_overlap_rows_excluded"] == 1
    assert report["evaluated_rows"] == len(flows)
    with pytest.raises(FileExistsError):
        evaluate_frozen(run, prepared, path, tmp_path / "another-holdout.json")
    # Freeze prevents a changed index from silently changing the evaluated population.
    index = next((prepared / "hash-index").glob("stable-*.npy"))
    index.write_bytes(b"changed")
    with pytest.raises(ValueError, match="index changed"):
        evaluate_frozen(run, prepared, path, tmp_path / "bad.json")


def test_subset_contract_validation(tmp_path):
    path = tmp_path / "bad.joblib"
    joblib.dump(
        {"format_version": 2, "schema": SCHEMA_VERSION, "features": [FEATURES[0], "Label"]}, path
    )
    with pytest.raises(ValueError, match="contract"):
        load_bundle(path)

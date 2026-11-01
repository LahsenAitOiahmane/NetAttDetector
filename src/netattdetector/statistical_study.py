"""Fixed-thread core/statistical diagnostics with unchanged candidate labels and partitions."""

import importlib.metadata
import json
import sqlite3
import subprocess
import time
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from threadpoolctl import threadpool_limits

from netattdetector.io import sha256_file, write_json
from netattdetector.measurement import CORE_FEATURES
from netattdetector.native_cohort import SPLITS
from netattdetector.native_errors import compare_predictions, error_slices
from netattdetector.native_study import feature_overlap, fit_one, make_model, selection_key
from netattdetector.packet_evidence import POLICY
from netattdetector.resources import ResourceMonitor
from netattdetector.statistical import EXTRA_FEATURES, FEATURES, FLOAT_FIELDS, SCHEMA
from netattdetector.statistical_cohort import verify_statistical_inputs

CONFIG = "configs/statistical-study.json"
POLICIES = {
    "core11_control": CORE_FEATURES,
    "statistics27": FEATURES,
    "statistics25_no_ports": FEATURES[2:],
}


class StatisticalFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, policy="statistics27"):
        self.policy = policy

    def fit(self, X, y=None):
        self.transform(X)
        return self

    def transform(self, X):
        if (
            not isinstance(X, pd.DataFrame)
            or list(X.columns) != FEATURES
            or self.policy not in POLICIES
        ):
            raise ValueError(
                "Statistical input requires the exact ordered 27-field contract and policy."
            )
        values = X.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Statistical model inputs must be finite nonnegative measurements.")
        integers = [i for i, name in enumerate(FEATURES) if name not in FLOAT_FIELDS]
        if not np.equal(values[:, integers], np.floor(values[:, integers])).all():
            raise ValueError("Raw core/count/min/max fields must be integers.")
        if (
            (X["src_port"] > 65535).any()
            or (X["dst_port"] > 65535).any()
            or (X["protocol"] > 255).any()
            or not X["ip_version"].isin([4, 6]).all()
        ):
            raise ValueError("Invalid native port/protocol/IP-version range.")
        packets = (X["src2dst_packets"] + X["dst2src_packets"]).to_numpy()
        flags = X[EXTRA_FEATURES[8:]].to_numpy()
        if (
            (packets == 0).any()
            or (flags > packets[:, None]).any()
            or np.any(flags[X["protocol"].to_numpy() != 6] != 0)
        ):
            raise ValueError("Invalid statistical packet or TCP-flag counts.")
        for suffix in ("ps", "piat_ms"):
            if (
                X[f"bidirectional_min_{suffix}"] > X[f"bidirectional_mean_{suffix}"] + 1e-9
            ).any() or (
                X[f"bidirectional_mean_{suffix}"] > X[f"bidirectional_max_{suffix}"] + 1e-9
            ).any():
                raise ValueError("Inconsistent statistical minimum/mean/maximum.")
        intervals = [f"bidirectional_{name}_piat_ms" for name in ("min", "mean", "stddev", "max")]
        if np.any(X[intervals].to_numpy()[packets == 1] != 0):
            raise ValueError("Single-packet inputs cannot have interarrival observations.")
        return X[POLICIES[self.policy]].to_numpy(dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        if self.policy not in POLICIES:
            raise ValueError("Unregistered statistical feature policy.")
        return np.asarray(POLICIES[self.policy], dtype=object)


def statistical_model(name, policy, seed=42, threads=1):
    if name not in ("dummy", "logistic", "histogram"):
        raise ValueError("Unregistered statistical diagnostic model.")
    pipeline = make_model(name, "native11", seed, threads)
    pipeline.set_params(features=StatisticalFeatures(policy))
    return pipeline


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if any(
        config.get(name) != value
        for name, value in {
            "schema_version": 1,
            "study_id": "statistical-capture-diagnostic-v1",
            "expected_core_logical_sha256": "9be2a2d5c39b3cb78a31fec40118971ba03b0e62b398e1151f7480ce8c312d59",
            "expected_statistical_logical_sha256": "a4e6d0ff836814acadd4cc778b0b63320fe5dc38cc080f040caaeb95db189259",
            "statistical_config": "configs/data/statistical-features.json",
            "feature_contract": SCHEMA,
            "label_policy": POLICY,
            "research_diagnostic_fitting_authorized": True,
            "candidate_policy_promoted": False,
            "deployment_authorized": False,
            "public_model_release_authorized": False,
            "seed": 42,
            "threads": 1,
            "maximum_fits": 15,
            "maximum_wall_seconds": 3600,
            "sampled_memory_screen_mib": 1536,
            "models": ["logistic", "histogram"],
            "feature_policies": list(POLICIES),
            "weight_policies": ["flow_natural", "connection_class_balanced"],
            "research_screen": {
                "minimum_precision": 0.98,
                "maximum_fpr": 0.005,
                "minimum_recall": 0.8,
                "minimum_true_positive_connections": 25,
            },
        }.items()
    ):
        raise ValueError("Use the registered fixed-thread statistical diagnostic configuration.")
    return config


def verify_inputs(cohort, sidecar, output, config):
    return verify_statistical_inputs(cohort, sidecar, output, config)


def load_frames(cohort, sidecar):
    db = sqlite3.connect((sidecar / "statistics.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    try:
        db.execute(
            "ATTACH DATABASE ? AS core",
            ((cohort / "cohort.sqlite").resolve().as_uri() + "?mode=ro",),
        )
        fields = [
            "c.label",
            "c.detail",
            "c.group_hash",
            *[f"c.{name}" for name in CORE_FEATURES],
            *[f"f.{name}" for name in EXTRA_FEATURES],
        ]
        return {
            split: pd.read_sql_query(
                "SELECT "
                + ",".join(fields)
                + " FROM features f JOIN core.flows c ON f.ordinal=c.ordinal AND f.flow_hash=c.flow_hash WHERE c.split=? AND c.status='matched' ORDER BY c.ordinal",
                db,
                params=(split,),
            )
            for split in SPLITS
        }
    finally:
        db.close()


def run_statistical_study(cohort, sidecar, output, config_path=CONFIG):
    config = load_config(config_path)
    cohort, sidecar, output = map(Path, (cohort, sidecar, output))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    verified, stat_manifest = verify_inputs(cohort, sidecar, output, config)
    write_json(output / "study-config.json", config)
    records = []
    with ResourceMonitor(include_children=True) as resources:
        frames = load_frames(cohort, sidecar)
        overlaps = {
            kind: [
                feature_overlap(
                    frames,
                    policy,
                    input_features=FEATURES,
                    transformer=StatisticalFeatures(policy),
                    data_type=kind,
                )
                for policy in POLICIES
            ]
            for kind in ("<f8", "<f4")
        }
        specs = [
            {
                "id": "dummy",
                "model": "dummy",
                "features": "core11_control",
                "weights": "flow_natural",
            }
        ]
        specs.extend(
            {
                "id": f"{name}-{features}-{weights}",
                "model": name,
                "features": features,
                "weights": weights,
            }
            for name, features, weights in product(
                config["models"], config["feature_policies"], config["weight_policies"]
            )
        )
        for spec in specs:
            if time.perf_counter() - resources.start >= config["maximum_wall_seconds"]:
                raise ValueError("Statistical fit wall budget reached.")
            model, scores, report = fit_one(
                spec,
                frames,
                config,
                threads=1,
                input_features=FEATURES,
                model_factory=statistical_model,
            )
            bundle = {
                "format": "statistical-diagnostic-v1",
                "schema": SCHEMA,
                "features": FEATURES,
                "model": model,
                "specification": spec,
                "threshold": report["operating_point"]["threshold"],
                "training_threads": 1,
                "prediction_threads": 1,
                "core_logical_sha256": config["expected_core_logical_sha256"],
                "statistical_logical_sha256": config["expected_statistical_logical_sha256"],
                "label_policy": POLICY,
                "candidate_policy_promoted": False,
                "deployment_authorized": False,
                "public_model_release_authorized": False,
            }
            path = output / f"{spec['id']}.joblib"
            joblib.dump(bundle, path, compress=3)
            restored = joblib.load(path)
            probe = frames["validation"][FEATURES].iloc[:5000]
            with threadpool_limits(limits=1):
                expected = model.predict_proba(probe)[:, 1]
                start = time.perf_counter()
                actual = restored["model"].predict_proba(probe)[:, 1]
                latency = time.perf_counter() - start
            difference = compare_predictions(expected, actual, bundle["threshold"])
            if not difference["scores_exactly_identical"]:
                raise ValueError("Statistical model changed predictions after reload.")
            report.update(
                {
                    "artifact_bytes": path.stat().st_size,
                    "artifact_sha256": sha256_file(path),
                    "reload": difference,
                    "inference_probe": {
                        "rows": len(probe),
                        "seconds": latency,
                        "microseconds_per_flow": latency * 1e6 / len(probe),
                        "scope": "One warmed batch including feature validation/transforms, excluding I/O/export/sensing.",
                    },
                }
            )
            records.append(report)
            write_json(
                output / "progress.json",
                {"completed_candidate_fits": len(records), "records": records},
            )
            del model, scores, restored, bundle
        chosen = min((row for row in records if row["id"] != "dummy"), key=selection_key)
        bundle = joblib.load(output / f"{chosen['id']}.joblib")
        with threadpool_limits(limits=1):
            reference = bundle["model"].predict_proba(frames["validation"][FEATURES])[:, 1]
        errors = error_slices(frames["validation"], reference, bundle["threshold"])
        controls = []
        for number in range(2):
            if time.perf_counter() - resources.start >= config["maximum_wall_seconds"]:
                raise ValueError("Statistical repeat wall budget reached.")
            repeated, scores, report = fit_one(
                chosen["specification"],
                frames,
                config,
                threads=1,
                input_features=FEATURES,
                model_factory=statistical_model,
            )
            comparison = compare_predictions(
                reference, scores["validation"], chosen["operating_point"]["threshold"]
            )
            comparison.update(
                {
                    "repeat": number + 1,
                    "training_threads": 1,
                    "calibration_threshold_identical": chosen["operating_point"]["threshold"]
                    == report["operating_point"]["threshold"],
                    "fit_seconds": report["fit_seconds"],
                    "passed": comparison["max_absolute_score_difference"] <= 1e-7
                    and comparison["frozen_threshold_decision_differences"] == 0,
                }
            )
            controls.append(comparison)
            del repeated, scores
    result = {
        "study_id": config["study_id"],
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config_sha256": sha256_file(config_path),
        "uv_lock_sha256": sha256_file("uv.lock"),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "joblib")
        },
        "exporter_version": stat_manifest["exporter_version"],
        "core_logical_sha256": config["expected_core_logical_sha256"],
        "statistical_logical_sha256": config["expected_statistical_logical_sha256"],
        "statistical_sidecar_sha256": stat_manifest["sidecar"]["sha256"],
        "cohort_audit": verified["audit"],
        "feature_policy_diagnostics": overlaps,
        "records": records,
        "completed_fits": len(records) + len(controls),
        "selected_diagnostic_id": chosen["id"],
        "qualifying_non_dummy_candidates": sum(
            row["passes_calibration_and_validation_screen"]
            for row in records
            if row["id"] != "dummy"
        ),
        "repeat_controls": controls,
        "repeatability_screen_met": all(row["passed"] for row in controls),
        "validation_errors": errors,
        "resources": {
            "wall_seconds": resources.elapsed,
            "sampled_peak_process_tree_rss_sum_mib": resources.peak_tree / 2**20,
            "scope": "Frame loading, six pattern audits, fits, scoring, serialization and controls; excludes imports, preparation and input verification. 50 ms RSS samples are not OS-enforced limits.",
        },
        "resource_screen_met": resources.elapsed < config["maximum_wall_seconds"]
        and resources.peak_tree < config["sampled_memory_screen_mib"] * 2**20,
        "deployment_authorized": False,
        "public_model_release_authorized": False,
        "candidate_policy_promoted": False,
        "external_test_accessed": False,
        "device_isolation_verified": False,
        "scope": config["scope"],
    }
    write_json(output / "study.json", result)
    write_json(
        output / "freeze.json",
        {
            "study_id": config["study_id"],
            "selected_diagnostic_id": chosen["id"],
            "specification": chosen["specification"],
            "artifact_sha256": chosen["artifact_sha256"],
            "threshold": chosen["operating_point"]["threshold"],
            "study_sha256": sha256_file(output / "study.json"),
            "core_logical_sha256": config["expected_core_logical_sha256"],
            "statistical_logical_sha256": config["expected_statistical_logical_sha256"],
            "passes_development_research_screen": chosen[
                "passes_calibration_and_validation_screen"
            ],
            "repeatability_screen_met": result["repeatability_screen_met"],
            "deployment_authorized": False,
            "public_model_release_authorized": False,
            "external_test_accessed": False,
        },
    )
    if not result["repeatability_screen_met"] or not result["resource_screen_met"]:
        raise ValueError(
            "Statistical diagnostic resource/repeatability screen failed; evidence retained."
        )
    return result

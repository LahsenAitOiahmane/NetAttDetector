"""Registered one-thread context comparison with per-family validation guards."""

import importlib.metadata
import json
import subprocess
import time
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from threadpoolctl import threadpool_limits

from netattdetector.causal_context import CONTEXT_FEATURES, SCHEMA
from netattdetector.context_cohort import FEATURES, verify_context_inputs
from netattdetector.io import sha256_file, write_json
from netattdetector.measurement import CORE_FEATURES
from netattdetector.native_cohort import SPLITS
from netattdetector.native_errors import compare_predictions, error_slices
from netattdetector.native_study import feature_overlap, fit_one, make_model, selection_key
from netattdetector.private_artifacts import seal_model
from netattdetector.resources import ResourceMonitor
from netattdetector.statistical import EXTRA_FEATURES
from netattdetector.statistical import FEATURES as STAT_FEATURES
from netattdetector.statistical_cohort import connect_readonly
from netattdetector.statistical_study import StatisticalFeatures

CONFIG = "configs/context-study.json"
POLICIES = {
    "statistics27_control": STAT_FEATURES,
    "context36": FEATURES,
    "context34_no_ports": FEATURES[2:],
}


class ContextFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, policy="context36"):
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
                "Context inputs require the exact ordered 36-field contract and policy."
            )
        StatisticalFeatures().transform(X[STAT_FEATURES])
        values = X[CONTEXT_FEATURES].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or (values > 100_000).any()
            or not np.equal(values, np.floor(values)).all()
        ):
            raise ValueError("Context counts must be bounded nonnegative integers.")
        if (
            (np.diff(values[:, :3], axis=1) < 0).any()
            or (np.diff(values[:, 3:6], axis=1) < 0).any()
            or (values[:, 6:8] > values[:, 2:3]).any()
            or (values[:, 8] > values[:, 5]).any()
        ):
            raise ValueError("Context window/distinct counts are inconsistent.")
        return X[POLICIES[self.policy]].to_numpy(dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(POLICIES[self.policy], dtype=object)


def context_model(name, policy, seed=42, threads=1):
    if name not in ("dummy", "logistic", "histogram") or threads != 1:
        raise ValueError("Use registered context models at one thread.")
    model = make_model(name, "native11", seed, threads)
    model.set_params(features=ContextFeatures(policy))
    return model


def load_config(path=CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "study_id": "context-capture-diagnostic-v1",
        "expected_core_logical_sha256": "9be2a2d5c39b3cb78a31fec40118971ba03b0e62b398e1151f7480ce8c312d59",
        "expected_statistical_logical_sha256": "a4e6d0ff836814acadd4cc778b0b63320fe5dc38cc080f040caaeb95db189259",
        "expected_context_logical_sha256": "7a5960b15ce59d56955c3975289bbef6e9cacbbad16c78c8336109589092c897",
        "statistical_config": "configs/data/statistical-features.json",
        "context_config": "configs/data/causal-context.json",
        "feature_contract": SCHEMA,
        "label_policy": "iot23-packet-point-candidate-v1",
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
        "family_screen": {
            "required_validation_families": ["C&C", "DDoS", "PartOfAHorizontalPortScan"],
            "minimum_connections": 25,
            "minimum_recall": 0.8,
        },
        "artifact_recipient": "configs/security/model-recipient.json",
    }
    if any(config.get(name) != value for name, value in expected.items()):
        raise ValueError("Use the registered family-aware context comparison.")
    return config


def family_guard(report, screen):
    families = report["metrics"]["validation"]["families"]
    checks = {}
    for name in screen["required_validation_families"]:
        family = families.get(name, {})
        support = family.get("attack_connections", 0)
        recall = family.get("attack_recall")
        enough = support >= screen["minimum_connections"]
        tp = round(support * recall) if recall is not None else 0
        # Wilson interval conditional on the accepted UIDs, not independent attacks.
        if support:
            z2, p = 1.959963984540054**2, tp / support
            middle = (p + z2 / (2 * support)) / (1 + z2 / support)
            half = (z2 * (p * (1 - p) / support + z2 / (4 * support**2))) ** 0.5 / (
                1 + z2 / support
            )
            interval = [max(0.0, middle - half), min(1.0, middle + half)]
        else:
            interval = None
        checks[name] = {
            "attack_connections": support,
            "detected_connections": tp,
            "recall": recall,
            "conditional_wilson_95": interval,
            "assessed": enough,
            "passed": enough and recall is not None and recall >= screen["minimum_recall"],
        }
    report["family_screen"] = {
        "checks": checks,
        "passed": all(row["passed"] for row in checks.values()),
        "limitation": "Accepted UID conditional intervals; one capture, not independent attacks/devices, capture or model-selection uncertainty. Other families remain unassessed.",
    }
    report["passes_binary_and_family_screen"] = (
        report["passes_calibration_and_validation_screen"] and report["family_screen"]["passed"]
    )
    return report


def context_selection_key(record):
    # Once the stronger screen fails, use the registered diagnostic ranking directly.
    return (-int(record["passes_binary_and_family_screen"]), *selection_key(record)[1:])


def load_frames(cohort, statistics, context):
    db = connect_readonly(Path(context) / "context.sqlite")
    try:
        for alias, path in (
            ("core", Path(cohort) / "cohort.sqlite"),
            ("statistics", Path(statistics) / "statistics.sqlite"),
        ):
            db.execute(f"ATTACH DATABASE ? AS {alias}", (path.resolve().as_uri() + "?mode=ro",))
        fields = [
            "c.label",
            "c.detail",
            "c.group_hash",
            *[f"c.{name}" for name in CORE_FEATURES],
            *[f"s.{name}" for name in EXTRA_FEATURES],
            *[f"f.{name}" for name in CONTEXT_FEATURES],
        ]
        return {
            split: pd.read_sql_query(
                "SELECT "
                + ",".join(fields)
                + " FROM features f JOIN core.flows c ON f.ordinal=c.ordinal AND f.flow_hash=c.flow_hash JOIN statistics.features s ON f.ordinal=s.ordinal AND f.flow_hash=s.flow_hash WHERE c.split=? AND c.status='matched' ORDER BY c.ordinal",
                db,
                params=(split,),
            )
            for split in SPLITS
        }
    finally:
        db.close()


def synthetic_probes():
    """Known public synthetic packet fixture, crossed with invented valid counts."""
    rows = json.loads(Path("tests/fixtures/pcap/statistical-expected.json").read_text())["rows"]
    return pd.DataFrame(
        [
            {
                **{name: row[name] for name in STAT_FEATURES},
                **dict(zip(CONTEXT_FEATURES, [count] * 6 + [min(count, 3)] * 3, strict=True)),
            }
            for row in rows
            for count in (0, 1, 2, 5, 10, 25, 60, 135, 306)
        ],
        columns=FEATURES,
    )


def run_context_study(cohort, statistics, context, output, config_path=CONFIG):
    config = load_config(config_path)
    cohort, statistics, context, output = map(Path, (cohort, statistics, context, output))
    output.mkdir(parents=True, exist_ok=False)
    verified, stat_manifest, manifest = verify_context_inputs(
        cohort, statistics, context, output, config
    )
    write_json(output / "study-config.json", config)
    records = []
    with ResourceMonitor(include_children=True) as resources:
        frames = load_frames(cohort, statistics, context)
        overlaps = {
            kind: [
                feature_overlap(
                    frames,
                    policy,
                    input_features=FEATURES,
                    transformer=ContextFeatures(policy),
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
                "features": "statistics27_control",
                "weights": "flow_natural",
            },
            *(
                {
                    "id": f"{name}-{features}-{weights}",
                    "model": name,
                    "features": features,
                    "weights": weights,
                }
                for name, features, weights in product(
                    config["models"], config["feature_policies"], config["weight_policies"]
                )
            ),
        ]
        for spec in specs:
            if time.perf_counter() - resources.start >= config["maximum_wall_seconds"]:
                raise ValueError("Context fit wall budget reached.")
            model, scores, report = fit_one(
                spec,
                frames,
                config,
                threads=1,
                input_features=FEATURES,
                model_factory=context_model,
            )
            bundle = {
                "format": "context-diagnostic-v1",
                "schema": SCHEMA,
                "features": FEATURES,
                "model": model,
                "specification": spec,
                "threshold": report["operating_point"]["threshold"],
                "training_threads": 1,
                "prediction_threads": 1,
                "label_policy": config["label_policy"],
                "candidate_policy_promoted": False,
                "deployment_authorized": False,
                "public_model_release_authorized": False,
                **{
                    name.removeprefix("expected_"): value
                    for name, value in config.items()
                    if name.startswith("expected_")
                },
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
                raise ValueError("Context model changed predictions after reload.")
            report.update(
                {
                    "artifact_bytes": path.stat().st_size,
                    "artifact_sha256": sha256_file(path),
                    "reload": difference,
                    "inference_probe": {
                        "rows": len(probe),
                        "seconds": latency,
                        "microseconds_per_flow": latency * 1e6 / len(probe),
                        "scope": "One warmed batch with validation/transforms; excludes I/O, export and sensing.",
                    },
                }
            )
            records.append(family_guard(report, config["family_screen"]))
            write_json(
                output / "progress.json",
                {"completed_candidate_fits": len(records), "records": records},
            )
            del model, scores, restored, bundle
        chosen = min((row for row in records if row["id"] != "dummy"), key=context_selection_key)
        bundle = joblib.load(output / f"{chosen['id']}.joblib")
        with threadpool_limits(limits=1):
            reference = bundle["model"].predict_proba(frames["validation"][FEATURES])[:, 1]
        errors = error_slices(frames["validation"], reference, bundle["threshold"])
        controls = []
        for number in range(2):
            if time.perf_counter() - resources.start >= config["maximum_wall_seconds"]:
                raise ValueError("Context repeat wall budget reached.")
            repeated, scores, report = fit_one(
                chosen["specification"],
                frames,
                config,
                threads=1,
                input_features=FEATURES,
                model_factory=context_model,
            )
            comparison = compare_predictions(reference, scores["validation"], bundle["threshold"])
            comparison.update(
                {
                    "repeat": number + 1,
                    "training_threads": 1,
                    "calibration_threshold_identical": chosen["operating_point"]["threshold"]
                    == report["operating_point"]["threshold"],
                    "fit_seconds": report["fit_seconds"],
                }
            )
            comparison["passed"] = (
                comparison["max_absolute_score_difference"] <= 1e-7
                and comparison["frozen_threshold_decision_differences"] == 0
                and comparison["calibration_threshold_identical"]
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
            for name in ("numpy", "pandas", "scikit-learn", "joblib", "cryptography")
        },
        "exporter_version": manifest["exporter_version"],
        "cohort_audit": verified["audit"],
        "statistical_sidecar_sha256": stat_manifest["sidecar"]["sha256"],
        "context_sidecar_sha256": manifest["database"]["sha256"],
        **{
            name.removeprefix("expected_"): value
            for name, value in config.items()
            if name.startswith("expected_")
        },
        "feature_policy_diagnostics": overlaps,
        "records": records,
        "completed_fits": len(records) + len(controls),
        "selected_diagnostic_id": chosen["id"],
        "qualifying_non_dummy_candidates": sum(
            row["passes_binary_and_family_screen"] for row in records if row["id"] != "dummy"
        ),
        "repeat_controls": controls,
        "repeatability_screen_met": all(row["passed"] for row in controls),
        "validation_errors": errors,
        "resources": {
            "wall_seconds": resources.elapsed,
            "sampled_peak_process_tree_rss_sum_mib": resources.peak_tree / 2**20,
            "scope": "Frame loading, six pattern audits, fits/scoring, serialization and repeats; excludes imports, preparation, verification and encryption. RSS sampled every 50 ms, not an OS-enforced bound.",
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
    if result["completed_fits"] != config["maximum_fits"]:
        raise ValueError("Context fit budget differs from registration.")
    write_json(output / "study.json", result)
    freeze = {
        "study_id": config["study_id"],
        "selected_diagnostic_id": chosen["id"],
        "specification": chosen["specification"],
        "artifact_sha256": chosen["artifact_sha256"],
        "threshold": bundle["threshold"],
        "study_sha256": sha256_file(output / "study.json"),
        "passes_development_research_screen": chosen["passes_binary_and_family_screen"],
        "repeatability_screen_met": result["repeatability_screen_met"],
        "deployment_authorized": False,
        "public_model_release_authorized": False,
        "external_test_accessed": False,
        **{
            name.removeprefix("expected_"): value
            for name, value in config.items()
            if name.startswith("expected_")
        },
    }
    write_json(output / "freeze.json", freeze)
    probes = synthetic_probes()
    with threadpool_limits(limits=1):
        expected = bundle["model"].predict_proba(probes)[:, 1].tolist()
    transfer = seal_model(
        output / f"{chosen['id']}.joblib",
        freeze,
        {
            "features": FEATURES,
            "rows": probes.to_numpy().tolist(),
            "scores": expected,
            "scope": "Public synthetic packet fixture and invented bounded context counts; no real rows or labels.",
        },
        config["artifact_recipient"],
        output / "selected.nadenc",
    )
    write_json(output / "transfer.json", transfer)
    if not result["repeatability_screen_met"] or not result["resource_screen_met"]:
        raise ValueError("Context resource/repeatability screen failed; evidence retained.")
    return result

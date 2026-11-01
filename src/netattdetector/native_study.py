"""Registered native-feature diagnostic fits; candidate labels, no production promotion."""

import hashlib
import importlib.metadata
import importlib.util
import json
import sqlite3
import subprocess
import time
import warnings
from itertools import combinations, product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from netattdetector.evaluation import metrics
from netattdetector.io import sha256_file, write_json
from netattdetector.measurement import CORE_FEATURES, SCHEMA
from netattdetector.native_cohort import SPLITS, verify_native_cohort
from netattdetector.packet_evidence import POLICY
from netattdetector.resources import ResourceMonitor

CONFIG = "configs/native-study.json"
STUDY = "native-capture-diagnostic-v1"


class NativeFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, policy="native11"):
        self.policy = policy

    def fit(self, X, y=None):
        self.transform(X)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame) or list(X.columns) != CORE_FEATURES:
            raise ValueError(
                "Native model input requires exactly the ordered eleven measured fields."
            )
        values = X.to_numpy(dtype=np.float64)
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or not np.equal(values, np.floor(values)).all()
        ):
            raise ValueError("Native raw model features must be finite nonnegative integers.")
        if self.policy == "native11":
            return values
        if self.policy != "behavioral_derived":
            raise ValueError("Unknown native feature policy.")
        column = {name: values[:, index] for index, name in enumerate(CORE_FEATURES)}
        forward, reverse = column["src2dst_packets"], column["dst2src_packets"]
        fbytes, rbytes = column["src2dst_bytes"], column["dst2src_bytes"]
        packets, ip_bytes = forward + reverse, fbytes + rbytes
        duration = column["bidirectional_duration_ms"] + 1.0
        extra = [
            reverse / np.maximum(packets, 1),
            rbytes / np.maximum(ip_bytes, 1),
            ip_bytes / np.maximum(packets, 1),
            fbytes / np.maximum(forward, 1),
            rbytes / np.maximum(reverse, 1),
            1000 * packets / duration,
            1000 * ip_bytes / duration,
        ]
        return np.column_stack([values[:, 2:], *extra])

    def get_feature_names_out(self, input_features=None):
        if self.policy == "native11":
            return np.asarray(CORE_FEATURES, dtype=object)
        if self.policy != "behavioral_derived":
            raise ValueError("Unknown native feature policy.")
        return np.asarray(
            [
                *CORE_FEATURES[2:],
                "reverse_packet_fraction",
                "reverse_byte_fraction",
                "mean_ip_bytes_per_packet",
                "forward_mean_ip_bytes_per_packet",
                "reverse_mean_ip_bytes_per_packet",
                "smoothed_packets_per_second",
                "smoothed_ip_bytes_per_second",
            ],
            dtype=object,
        )


def training_weights(frame, policy):
    if policy == "flow_natural":
        return None
    if policy != "connection_class_balanced":
        raise ValueError("Unknown native training weighting policy.")
    unique = frame.drop_duplicates("group_hash")
    if frame.groupby("group_hash")["label"].nunique().max() != 1 or set(unique["label"]) != {0, 1}:
        raise ValueError("Training weights require consistent connections and both classes.")
    class_groups = unique.groupby("label").size()
    segments = frame.groupby("group_hash")["label"].transform("size").to_numpy()
    weights = 1.0 / segments / frame["label"].map(class_groups).to_numpy()
    return weights / weights.mean()


def make_model(name, feature_policy, seed=42, threads=2):
    steps = [("features", NativeFeatures(feature_policy))]
    if name == "dummy":
        estimator = DummyClassifier(strategy="prior", random_state=seed)
    elif name == "logistic":
        steps.extend([("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler())])
        estimator = LogisticRegression(C=1, solver="lbfgs", max_iter=300, random_state=seed)
    elif name == "histogram":
        estimator = HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.08,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1,
            early_stopping=False,
            random_state=seed,
        )
    elif name == "extra_trees":
        estimator = ExtraTreesClassifier(
            n_estimators=64,
            max_depth=14,
            min_samples_leaf=5,
            max_features=1.0,
            bootstrap=False,
            n_jobs=threads,
            random_state=seed,
        )
    else:
        raise ValueError("Unregistered native model.")
    return Pipeline([*steps, ("model", estimator)])


def checked_scores(scores):
    scores = np.asarray(scores, dtype=float)
    if (
        scores.ndim != 1
        or not len(scores)
        or not np.isfinite(scores).all()
        or (scores < 0).any()
        or (scores > 1).any()
    ):
        raise ValueError("Attack scores must be a nonempty finite probability vector.")
    return scores


def connection_scores(frame, scores):
    scores = checked_scores(scores)
    if len(frame) != len(scores):
        raise ValueError("Score/row lengths differ.")
    work = frame[["group_hash", "label", "detail"]].copy()
    work["score"] = scores
    grouped = work.groupby("group_hash", sort=True)
    if grouped["label"].nunique().max() != 1 or grouped["detail"].nunique().max() != 1:
        raise ValueError("Inconsistent native connection labels/details.")
    return grouped.agg(label=("label", "first"), detail=("detail", "first"), score=("score", "max"))


def threshold_connections(y, scores, screen):
    """Exact tie-aware frontier; no validation labels enter threshold selection."""
    scores, y = checked_scores(scores), np.asarray(y)
    if len(y) != len(scores) or set(y.tolist()) != {0, 1}:
        raise ValueError("Calibration requires both binary connection classes.")
    order = np.argsort(-scores, kind="stable")
    ranked, labels = scores[order], y[order]
    ends = np.r_[np.flatnonzero(ranked[:-1] != ranked[1:]), len(ranked) - 1]
    tp = np.cumsum(labels)[ends]
    fp = ends + 1 - tp
    precision, recall, fpr = tp / (ends + 1), tp / y.sum(), fp / (len(y) - y.sum())
    eligible = (
        (precision >= screen["minimum_precision"])
        & (fpr <= screen["maximum_fpr"] + 1e-12)
        & (tp >= screen["minimum_true_positive_connections"])
    )
    if eligible.any():
        candidates = np.flatnonzero(eligible)
        selected = max(candidates, key=lambda i: (recall[i], precision[i], ranked[ends[i]]))
    else:
        f1 = 2 * tp / (2 * tp + fp + y.sum() - tp)
        selected = int(np.argmax(f1))  # Descending scores choose the higher threshold on ties.
    return {
        "threshold": float(ranked[ends[selected]]),
        "calibration_precision_fpr_support_feasible": bool(eligible.any()),
        "used_f1_fallback": not bool(eligible.any()),
    }


def passes_screen(result, screen):
    return (
        result["attack_precision"] >= screen["minimum_precision"]
        and result["attack_recall"] >= screen["minimum_recall"]
        and result["false_positive_rate"] <= screen["maximum_fpr"]
        and result["confusion_matrix"]["tp"] >= screen["minimum_true_positive_connections"]
    )


def evaluate_scores(frame, scores, threshold):
    connections = connection_scores(frame, scores)
    flow = metrics(frame["label"].to_numpy(), scores, threshold)
    group = metrics(connections["label"].to_numpy(), connections["score"].to_numpy(), threshold)
    group["connections"] = group.pop("rows")
    group["false_flags_per_10000_benign_connections"] = group.pop(
        "false_alarms_per_10000_benign_flows"
    )
    group["conditional_wilson_95"]["limitation"] = (
        "Conditional on independent accepted UIDs; not independent attacks/devices, capture-level uncertainty or model-selection uncertainty. One capture per partition."
    )
    families = {}
    for detail, part in connections.groupby("detail"):
        attacks = part[part["label"] == 1]
        families[detail] = {
            "connections": len(part),
            "attack_connections": len(attacks),
            "attack_recall": float((attacks["score"] >= threshold).mean())
            if len(attacks)
            else None,
        }
    return {"flows": flow, "connections": group, "families": families}


def feature_overlap(frames, policy, *, input_features=None, transformer=None, data_type="<f8"):
    input_features = CORE_FEATURES if input_features is None else input_features
    transformer = NativeFeatures(policy) if transformer is None else transformer
    if data_type not in ("<f8", "<f4"):
        raise ValueError("Pattern diagnostics support explicit float64/float32 representations.")
    parts = []
    for split in SPLITS:
        values = np.ascontiguousarray(
            transformer.transform(frames[split][input_features]), dtype=data_type
        )
        hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in values]
        parts.append(
            pd.DataFrame(
                {"split": split, "hash": hashes, "label": frames[split]["label"].to_numpy()}
            )
        )
    all_rows = pd.concat(parts, ignore_index=True)
    patterns = all_rows.groupby("hash").agg(rows=("label", "size"), classes=("label", "nunique"))
    overlaps = []
    for left, right in combinations(SPLITS, 2):
        a = all_rows[all_rows["split"] == left]["hash"].value_counts()
        b = all_rows[all_rows["split"] == right]["hash"].value_counts()
        shared = a.index.intersection(b.index)
        overlaps.append(
            {
                "left": left,
                "right": right,
                "shared_vectors": len(shared),
                "left_rows": int(a.loc[shared].sum()),
                "right_rows": int(b.loc[shared].sum()),
            }
        )
    return {
        "policy": policy,
        "features": transformer.get_feature_names_out().tolist(),
        "unique_vectors": len(patterns),
        "conflicting_vectors": int((patterns["classes"] > 1).sum()),
        "rows_in_conflicting_vectors": int(patterns.loc[patterns["classes"] > 1, "rows"].sum()),
        "pairwise_overlap": overlaps,
        "scope": f"Exact {'float64' if data_type == '<f8' else 'float32'} transformed inputs before model-specific log/scaling or estimator quantization; all repeated/conflicting rows retained.",
    }


def load_study_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "study_id": STUDY,
        "cohort_config": "configs/data/native-cohort.json",
        "expected_logical_rows_sha256": "9be2a2d5c39b3cb78a31fec40118971ba03b0e62b398e1151f7480ce8c312d59",
        "feature_contract": SCHEMA,
        "label_policy": POLICY,
        "research_diagnostic_fitting_authorized": True,
        "candidate_policy_promoted": False,
        "deployment_authorized": False,
        "public_model_release_authorized": False,
        "seed": 42,
        "threads": 2,
        "maximum_fits": 15,
        "maximum_wall_seconds": 3600,
        "sampled_memory_screen_mib": 1536,
        "models": ["logistic", "histogram", "extra_trees"],
        "feature_policies": ["native11", "behavioral_derived"],
        "weight_policies": ["flow_natural", "connection_class_balanced"],
        "research_screen": {
            "minimum_precision": 0.98,
            "maximum_fpr": 0.005,
            "minimum_recall": 0.8,
            "minimum_true_positive_connections": 25,
        },
    }
    if any(config.get(name) != value for name, value in expected.items()):
        raise ValueError(
            "Use the registered native diagnostic study and unchanged research screens."
        )
    return config


def load_frames(cohort):
    db = sqlite3.connect((Path(cohort) / "cohort.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    try:
        frames = {
            split: pd.read_sql_query(
                "SELECT label,detail,group_hash,"
                + ",".join(CORE_FEATURES)
                + " FROM flows WHERE split=? AND status='matched' ORDER BY ordinal",
                db,
                params=(split,),
            )
            for split in SPLITS
        }
    finally:
        db.close()
    return frames


def fit_one(spec, frames, config, *, threads=2, input_features=None, model_factory=None):
    input_features = CORE_FEATURES if input_features is None else input_features
    model_factory = make_model if model_factory is None else model_factory
    model = model_factory(spec["model"], spec["features"], config["seed"], threads)
    weights = training_weights(frames["train"], spec["weights"])
    start = time.perf_counter()
    with threadpool_limits(limits=threads), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(
            frames["train"][input_features],
            frames["train"]["label"].to_numpy(),
            model__sample_weight=weights,
        )
    fit_seconds = time.perf_counter() - start
    if spec["model"] == "extra_trees":
        # Sequential tree accumulation avoids scheduling-dependent floating-point sums.
        model.named_steps["model"].set_params(n_jobs=1)
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        scores = {
            split: checked_scores(model.predict_proba(frames[split][input_features])[:, 1])
            for split in ("calibration", "validation")
        }
    scoring_seconds = time.perf_counter() - start
    calibration = connection_scores(frames["calibration"], scores["calibration"])
    operating = threshold_connections(
        calibration["label"].to_numpy(), calibration["score"].to_numpy(), config["research_screen"]
    )
    reports = {
        split: evaluate_scores(frames[split], scores[split], operating["threshold"])
        for split in ("calibration", "validation")
    }
    result = {
        "id": spec["id"],
        "specification": spec,
        "fit_seconds": fit_seconds,
        "scoring_seconds": scoring_seconds,
        "warnings": [str(item.message) for item in caught],
        "operating_point": operating,
        "metrics": reports,
        "passes_calibration_and_validation_screen": operating[
            "calibration_precision_fpr_support_feasible"
        ]
        and all(
            passes_screen(reports[split]["connections"], config["research_screen"])
            for split in reports
        ),
    }
    return model, scores, result


def selection_key(record):
    result = record["metrics"]["validation"]["connections"]
    return (
        -int(record["passes_calibration_and_validation_screen"]),
        -result["attack_f1"],
        -result["attack_recall"],
        -result["attack_precision"],
        result["false_positive_rate"],
        record["artifact_bytes"],
        record["id"],
    )


def run_native_study(cohort, output, config_path=CONFIG):
    config = load_study_config(config_path)
    cohort, output = Path(cohort), Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    verification = verify_native_cohort(cohort, output / "cohort-verification.json")
    if verification["audit"]["rows_logical_sha256"] != config["expected_logical_rows_sha256"]:
        raise ValueError("Native study cohort differs from the registered exact rows.")
    cohort_manifest = json.loads((cohort / "manifest.json").read_text(encoding="utf-8"))
    write_json(output / "cohort-manifest.json", cohort_manifest)
    write_json(output / "study-config.json", config)
    records = []
    with ResourceMonitor(include_children=True) as resources:
        frames = load_frames(cohort)
        overlaps = [feature_overlap(frames, policy) for policy in config["feature_policies"]]
        specifications = [
            {"id": "dummy", "model": "dummy", "features": "native11", "weights": "flow_natural"}
        ]
        specifications.extend(
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
        for spec in specifications:
            if time.perf_counter() - resources.start >= config["maximum_wall_seconds"]:
                raise ValueError("Study wall budget reached before the next fit.")
            model, scores, record = fit_one(spec, frames, config)
            path = output / f"{spec['id']}.joblib"
            bundle = {
                "format": "native-diagnostic-v1",
                "schema": SCHEMA,
                "features": CORE_FEATURES,
                "model": model,
                "threshold": record["operating_point"]["threshold"],
                "specification": spec,
                "study_id": STUDY,
                "cohort_database_sha256": cohort_manifest["database"]["sha256"],
                "cohort_logical_sha256": verification["audit"]["rows_logical_sha256"],
                "label_policy": POLICY,
                "candidate_policy_promoted": False,
                "deployment_authorized": False,
                "public_model_release_authorized": False,
            }
            joblib.dump(bundle, path, compress=3)
            restored = joblib.load(path)
            probe = frames["validation"][CORE_FEATURES].iloc[:5000]
            with threadpool_limits(limits=1):
                original = model.predict_proba(probe)[:, 1]
                start = time.perf_counter()
                replay = restored["model"].predict_proba(probe)[:, 1]
                latency = time.perf_counter() - start
            record.update(
                {
                    "artifact_bytes": path.stat().st_size,
                    "artifact_sha256": sha256_file(path),
                    "reload_max_absolute_score_difference": float(
                        np.max(np.abs(original - replay))
                    ),
                    "reload_threshold_decision_differences": int(
                        np.count_nonzero(
                            (original >= bundle["threshold"]) != (replay >= bundle["threshold"])
                        )
                    ),
                    "inference_probe": {
                        "rows": len(probe),
                        "seconds": latency,
                        "microseconds_per_flow": latency * 1e6 / len(probe),
                        "scope": "One warmed batch probe including feature transforms; excludes file I/O and live export.",
                    },
                }
            )
            if (
                record["reload_max_absolute_score_difference"] != 0
                or record["reload_threshold_decision_differences"]
            ):
                raise ValueError("Saved native model predictions changed after reload.")
            records.append(record)
            write_json(
                output / "progress.json",
                {"completed_candidate_fits": len(records), "records": records},
            )
            del model, restored, scores, bundle
        selected = min((row for row in records if row["id"] != "dummy"), key=selection_key)
        selected_bundle = joblib.load(output / f"{selected['id']}.joblib")
        with threadpool_limits(limits=1):
            reference = selected_bundle["model"].predict_proba(frames["validation"][CORE_FEATURES])[
                :, 1
            ]
        controls = []
        for threads in (2, 1):
            if time.perf_counter() - resources.start >= config["maximum_wall_seconds"]:
                raise ValueError("Study wall budget reached before repeat controls.")
            repeated, repeated_scores, repeat_report = fit_one(
                selected["specification"], frames, config, threads=threads
            )
            probe = repeated_scores["validation"]
            maximum = float(np.max(np.abs(reference - probe)))
            threshold = selected["operating_point"]["threshold"]
            differences = int(np.count_nonzero((reference >= threshold) != (probe >= threshold)))
            controls.append(
                {
                    "threads": threads,
                    "max_absolute_validation_score_difference": maximum,
                    "validation_decision_differences_at_frozen_threshold": differences,
                    "same_selected_calibration_threshold": repeat_report["operating_point"][
                        "threshold"
                    ]
                    == threshold,
                    "fit_seconds": repeat_report["fit_seconds"],
                    "passed": maximum <= 1e-7 and differences == 0,
                }
            )
            del repeated, repeated_scores
    result = {
        "study_id": STUDY,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config_sha256": sha256_file(config_path),
        "uv_lock_sha256": sha256_file("uv.lock"),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "joblib", "nfstream")
            if name != "nfstream" or importlib.util.find_spec("nfstream")
        },
        "cohort_logical_sha256": verification["audit"]["rows_logical_sha256"],
        "cohort_audit": verification["audit"],
        "feature_policy_diagnostics": overlaps,
        "records": records,
        "selected_diagnostic_id": selected["id"],
        "repeat_controls": controls,
        "completed_fits": len(records) + len(controls),
        "qualifying_non_dummy_candidates": sum(
            record["passes_calibration_and_validation_screen"]
            for record in records
            if record["id"] != "dummy"
        ),
        "repeatability_screen_met": all(control["passed"] for control in controls),
        "resources": {
            "wall_seconds": resources.elapsed,
            "sampled_peak_process_rss_mib": resources.peak / 2**20,
            "sampled_peak_process_tree_rss_sum_mib": resources.peak_tree / 2**20,
            "scope": "50 ms samples covering frame loading, transformations, fits, scoring, serialization and controls; excludes imports, cohort construction and initial verification. Shared pages may be double-counted; no OS-enforced memory/time bound.",
        },
        "resource_screen_met": resources.elapsed < config["maximum_wall_seconds"]
        and resources.peak_tree < config["sampled_memory_screen_mib"] * 2**20,
        "research_diagnostic_fitting_authorized": True,
        "candidate_policy_promoted": False,
        "deployment_authorized": False,
        "public_model_release_authorized": False,
        "external_test_accessed": False,
        "device_isolation_verified": False,
        "scope": config["scope"],
    }
    write_json(output / "study.json", result)
    freeze = {
        "study_id": STUDY,
        "selected_diagnostic_id": selected["id"],
        "artifact_sha256": selected["artifact_sha256"],
        "threshold": selected["operating_point"]["threshold"],
        "specification": selected["specification"],
        "passes_development_research_screen": selected["passes_calibration_and_validation_screen"],
        "repeatability_screen_met": result["repeatability_screen_met"],
        "study_sha256": sha256_file(output / "study.json"),
        "cohort_logical_sha256": result["cohort_logical_sha256"],
        "deployment_authorized": False,
        "public_model_release_authorized": False,
        "external_test_accessed": False,
    }
    write_json(output / "freeze.json", freeze)
    if not result["resource_screen_met"] or not result["repeatability_screen_met"]:
        raise ValueError(
            "Study resource/repeatability screen failed; diagnostic evidence retained."
        )
    return result

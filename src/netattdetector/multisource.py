"""Registered multi-source selection, immutable freeze, and one-time new-source test."""

import hashlib
import importlib.metadata
import io
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from netattdetector.audit import STABLE_FEATURES, load_prepared, overlaps_development
from netattdetector.data import FEATURES, SCHEMA_VERSION, fingerprints, load_sample
from netattdetector.evaluation import attack_slices, metrics, wilson_bounds
from netattdetector.hardware import inventory
from netattdetector.inference import load_bundle
from netattdetector.io import sha256_file, write_json
from netattdetector.training import ResourceMonitor, TrainConfig, make_model, split_summary


def load_protocol(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("protocol_version") != 2 or not 1 <= config["threads"] <= 8:
        raise ValueError("Unsupported protocol version or thread count.")
    if config["holdout_source"] in config["development_sources"]:
        raise ValueError("Holdout source cannot be a development source.")
    if not config["candidates"] or len({x["id"] for x in config["candidates"]}) != len(
        config["candidates"]
    ):
        raise ValueError("Candidate IDs must be nonempty and unique.")
    if (
        config["search_budget_seconds"] <= 0
        or not 0 < config["min_precision"] <= 1
        or not 0 < config["max_fpr"] < 1
    ):
        raise ValueError("Invalid search/precision/FPR budget.")
    if not 0 < config["precision_lower_bound"] <= 1 or config["min_alerts_per_source"] < 1:
        raise ValueError("Invalid confidence/support constraints.")
    for candidate in config["candidates"]:
        if candidate["features"] not in {"full", "stable"} or candidate["weighting"] not in {
            "source",
            "source_class",
        }:
            raise ValueError("Unsupported feature/weight policy.")
        if (
            candidate["model"] not in {"hist", "extra_trees", "xgboost"}
            or min(candidate["iterations"], candidate["depth"]) < 1
        ):
            raise ValueError("Unsupported model or estimator size.")
    return config


def development_split(frame, seed):
    if not frame["__full_hash"].is_unique:
        raise ValueError("Development cohort must contain unique full-feature groups.")
    strata = frame["__source"] + ":" + frame["__label"].astype(str)
    folds = np.empty(len(frame), dtype=np.int8)
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    for fold, (_, indices) in enumerate(splitter.split(frame, strata, groups=frame["__hash"])):
        folds[indices] = fold
    train, cal, val = (
        np.flatnonzero(folds < 12),
        np.flatnonzero((folds >= 12) & (folds < 15)),
        np.flatnonzero(folds >= 15),
    )
    result = [frame.iloc[x].reset_index(drop=True) for x in (train, cal, val)]
    for i in range(3):
        for j in range(i):
            if set(result[i]["__hash"]) & set(result[j]["__hash"]):
                raise ValueError("Reduced feature group overlap between development partitions.")
    for part in result:
        for source in frame["__source"].unique():
            counts = part.loc[part["__source"] == source, "__label"].value_counts()
            if len(counts) != 2 or counts.min() < 2:
                raise ValueError("Each source/split must contain both classes with support >=2.")
    return result


def source_weights(frame, policy):
    columns = ["__source", "__label"] if policy == "source_class" else ["__source"]
    groups = frame.groupby(columns, observed=True)["__label"].transform("size")
    weights = 1 / groups.to_numpy(dtype=float)
    return weights / weights.mean()


def source_metrics(frame, scores, threshold):
    return {
        str(source): metrics(group["__label"], np.asarray(scores)[group.index], threshold)
        for source, group in frame.groupby("__source")
    }


def robust_threshold(frame, scores, config):
    """Choose one operating point that must satisfy constraints in every source."""
    scores = np.asarray(scores)
    order = np.argsort(-scores, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(scores[order])), len(scores) - 1]
    thresholds = scores[order][ends]
    eligible = np.ones(len(ends), dtype=bool)
    recalls, f1s, precisions = [], [], []
    y = frame["__label"].to_numpy()[order]
    names = frame["__source"].to_numpy()[order]
    for source in np.unique(names):
        source_mask = names == source
        positives = int(((y == 1) & source_mask).sum())
        negatives = int(((y == 0) & source_mask).sum())
        if not positives or not negatives:
            raise ValueError("Robust threshold selection requires both classes in every source.")
        tp = np.cumsum((y == 1) & source_mask)[ends]
        fp = np.cumsum((y == 0) & source_mask)[ends]
        alerts = tp + fp
        precision = np.divide(tp, alerts, out=np.zeros(len(ends)), where=alerts > 0)
        recall = tp / positives
        low, _ = wilson_bounds(tp, alerts)
        _, fpr_high = wilson_bounds(fp, negatives)
        eligible &= (
            (precision >= config["min_precision"])
            & (low >= config["precision_lower_bound"])
            & (fpr_high <= config["max_fpr"])
            & (alerts >= config["min_alerts_per_source"])
        )
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(
            np.divide(
                2 * precision * recall,
                precision + recall,
                out=np.zeros(len(ends)),
                where=(precision + recall) > 0,
            )
        )
    worst_recall = np.min(recalls, axis=0)
    worst_f1 = np.min(f1s, axis=0)
    objective = worst_recall if eligible.any() else worst_f1
    indices = np.flatnonzero(eligible) if eligible.any() else np.arange(len(ends))
    # Higher threshold is the deterministic final tie break for an identical objective.
    selected = indices[
        np.lexsort((thresholds[indices], np.mean(precisions, axis=0)[indices], objective[indices]))[
            -1
        ]
    ]
    return float(thresholds[selected]), bool(eligible.any())


def candidate_model(spec, config):
    base = TrainConfig(
        threads=config["threads"],
        seed=config["seed"],
        trees=spec["iterations"],
        boosting_iterations=spec["iterations"],
        max_depth=spec["depth"],
    )
    model = make_model(spec["model"], base)
    estimator = model.named_steps["classifier"]
    params = estimator.get_params(deep=False)
    if "class_weight" in params:
        estimator.set_params(class_weight=None)
    estimator.set_params(max_depth=spec["depth"])
    return model


def selection_rank(record):
    values = list(record["by_source"].values())
    feasible = record["constraints_met_each_source"]
    objective = min(x["attack_recall"] if feasible else x["attack_f1"] for x in values)
    return (
        feasible,
        objective,
        float(np.mean([x["average_precision"] for x in values])),
        -record["model_bytes"],
    )


def train_multisource(prepared, protocol, output, baseline=None):
    config = load_protocol(protocol)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output must be fresh.")
    output.mkdir(parents=True, exist_ok=True)
    if baseline and not Path(baseline).is_file():
        raise ValueError("Baseline artifact is missing.")
    records, best_model, best_record, best_features = [], None, None, None
    hardware = inventory()
    with ResourceMonitor() as monitor, threadpool_limits(limits=config["threads"]):
        frame, audit = load_prepared(prepared)
        if audit["seed"] != config["seed"] or audit["rows_per_source"] != config["rows_per_source"]:
            raise ValueError("Prepared cohort does not match registered sampling policy.")
        if (
            audit.get("cleaning_policy")
            != "full-feature conflicts/duplicates removed; reduced ambiguity retained"
        ):
            raise ValueError("Prepare a version-2 cohort that retains reduced-feature ambiguity.")
        actual_inputs = {Path(v["path"]).stem for v in audit["sources"].values()}
        if actual_inputs != set(config["development_sources"]):
            raise ValueError("Prepared sources do not match the registered development inputs.")
        index_checksums = {
            p.name: sha256_file(p) for p in sorted((Path(prepared) / "hash-index").glob("*.npy"))
        }
        if len(index_checksums) != 2 * audit["buckets"]:
            raise ValueError("Prepared global overlap index is incomplete.")
        train, calibration, validation = development_split(frame, config["seed"])
        deadline = time.perf_counter() + config["search_budget_seconds"]
        for spec in config["candidates"]:
            if time.perf_counter() >= deadline:
                break
            print(f"Comparing {spec['id']}...", flush=True)
            features = FEATURES if spec["features"] == "full" else STABLE_FEATURES
            model = candidate_model(spec, config)
            start = time.perf_counter()
            model.fit(
                train[features],
                train["__label"],
                classifier__sample_weight=source_weights(train, spec["weighting"]),
            )
            fit_seconds = time.perf_counter() - start
            # FrozenEstimator guarantees calibration does not refit classifier/preprocessing.
            calibrated = CalibratedClassifierCV(
                FrozenEstimator(model), method="sigmoid", n_jobs=config["threads"]
            )
            start = time.perf_counter()
            calibrated.fit(calibration[features], calibration["__label"])
            calibration_seconds = time.perf_counter() - start
            for calibration_name, variant in (("none", model), ("sigmoid", calibrated)):
                scores = variant.predict_proba(validation[features])[:, 1]
                threshold, feasible = robust_threshold(validation, scores, config)
                buffer = io.BytesIO()
                joblib.dump(variant, buffer, compress=3)
                record = {
                    "id": spec["id"] + ":" + calibration_name,
                    "spec": spec,
                    "calibration": calibration_name,
                    "fit_seconds": fit_seconds,
                    "calibration_seconds": calibration_seconds
                    if calibration_name == "sigmoid"
                    else 0.0,
                    "model_bytes": buffer.tell(),
                    "threshold": threshold,
                    "constraints_met_each_source": feasible,
                    "pooled_validation": metrics(validation["__label"], scores, threshold),
                    "by_source": source_metrics(validation, scores, threshold),
                }
                records.append(record)
                if best_record is None or selection_rank(record) > selection_rank(best_record):
                    best_model, best_record, best_features = variant, record, features
            write_json(output / "validation-progress.json", records)
        if best_model is None:
            raise ValueError("No candidate finished within the search budget.")
        bundle = {
            "format_version": 2,
            "schema": SCHEMA_VERSION,
            "features": best_features,
            "label_mapping": {0: "benign", 1: "attack"},
            "model": best_model,
            "threshold": best_record["threshold"],
            "config": config,
            "selection_id": best_record["id"],
            "score_kind": "sigmoid calibration on development data"
            if best_record["calibration"] == "sigmoid"
            else "uncalibrated model estimate",
        }
        joblib.dump(bundle, output / "model.joblib", compress=3)
        batch = validation.iloc[:2048][best_features]
        best_model.predict_proba(batch)
        timings = []
        for _ in range(7):
            start = time.perf_counter()
            best_model.predict_proba(batch)
            timings.append(time.perf_counter() - start)
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            dirty = bool(
                subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
            )
        except (OSError, subprocess.CalledProcessError):
            revision, dirty = None, None
        freeze = {
            "frozen_at_utc": datetime.now(UTC).isoformat(),
            "protocol_version": config["protocol_version"],
            "git_revision": revision,
            "git_worktree_dirty": dirty,
            "python": platform.python_version(),
            "config": config,
            "protocol_sha256": sha256_file(protocol),
            "prepared_audit_sha256": sha256_file(Path(prepared) / "audit.json"),
            "prepared_cache_sha256": audit["cache_sha256"],
            "development_sources": {k: v["sha256"] for k, v in audit["sources"].items()},
            "partitions": {
                k: {
                    **split_summary(v),
                    "source_label_counts": {
                        f"{s}:{y}": len(g) for (s, y), g in v.groupby(["__source", "__label"])
                    },
                    "full_fingerprint_sha256": hashlib.sha256(
                        np.sort(v["__full_hash"].to_numpy()).tobytes()
                    ).hexdigest(),
                }
                for k, v in zip(
                    ("train", "calibration", "validation"),
                    (train, calibration, validation),
                    strict=True,
                )
            },
            "validation_leaderboard": records,
            "selected": best_record,
            "validation_attack_slices_by_source": {
                str(source): attack_slices(
                    group.reset_index(drop=True),
                    best_model.predict_proba(group[best_features])[:, 1],
                    best_record["threshold"],
                )
                for source, group in validation.groupby("__source")
            },
            "model_sha256": sha256_file(output / "model.joblib"),
            "baseline_model_sha256": sha256_file(baseline) if baseline else None,
            "budget": {
                "max_new_fits": len(config["candidates"]),
                "completed_fits": len(records) // 2,
                "deadline_exceeded": time.perf_counter() > deadline,
            },
            "hardware": hardware,
            "inference": {
                "batch_rows": len(batch),
                "median_batch_ms": 1000 * float(np.median(timings)),
                "p95_batch_ms": 1000 * float(np.percentile(timings, 95)),
                "scope": "warm model inference only; excludes parsing/export/queueing",
            },
        }
    freeze["hash_index_checksums"] = index_checksums
    freeze["versions"] = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "xgboost")
        if name != "xgboost" or any(c["model"] == "xgboost" for c in config["candidates"])
    }
    freeze["resources"] = {
        "wall_seconds": monitor.elapsed,
        "sampled_peak_process_rss_mib": monitor.peak / 2**20,
    }
    write_json(output / "freeze.json", freeze)
    return freeze


def evaluate_frozen(run, prepared, holdout, output, baseline=None):
    run, output = Path(run), Path(output)
    freeze = json.loads((run / "freeze.json").read_text(encoding="utf-8"))
    config = freeze["config"]
    if sha256_file(run / "model.joblib") != freeze["model_sha256"]:
        raise ValueError("Frozen model checksum changed.")
    if sha256_file(Path(prepared) / "audit.json") != freeze["prepared_audit_sha256"]:
        raise ValueError("Prepared audit changed after model freeze.")
    for name, digest in freeze["hash_index_checksums"].items():
        if sha256_file(Path(prepared) / "hash-index" / name) != digest:
            raise ValueError("Global overlap index changed after model freeze.")
    if Path(holdout).stem != config["holdout_source"]:
        raise ValueError("Holdout input does not match the registered source.")
    if freeze["baseline_model_sha256"] and (
        not baseline or sha256_file(baseline) != freeze["baseline_model_sha256"]
    ):
        raise ValueError("Baseline must match the artifact frozen before holdout access.")
    if output.exists():
        raise ValueError("Holdout report already exists.")
    receipt = run / "holdout-access.json"
    # Exclusive creation records the unsealing before parsing labels or computing predictions.
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "opened_at_utc": datetime.now(UTC).isoformat(),
                "freeze_sha256": sha256_file(run / "freeze.json"),
                "holdout_sha256": sha256_file(holdout),
            },
            stream,
            indent=2,
        )
    with ResourceMonitor() as monitor, threadpool_limits(limits=config["threads"]):
        audit = json.loads((Path(prepared) / "audit.json").read_text(encoding="utf-8"))
        frame, profile = load_sample(
            holdout, config["holdout_rows"], config["chunk_rows"], config["holdout_seed"]
        )
        overlap = overlaps_development(frame, Path(prepared) / "hash-index", audit["buckets"])
        stable_overlap = overlaps_development(
            frame, Path(prepared) / "hash-index", audit["buckets"], kind="stable"
        )
        frame = frame.loc[~overlap].reset_index(drop=True)
        frame["__hash"] = fingerprints(frame, STABLE_FEATURES)
        conflicts = frame.groupby("__hash")["__label"].nunique()
        ambiguous = set(conflicts[conflicts > 1].index)
        if frame["__label"].nunique() != 2:
            raise ValueError("Holdout cleaning removed a binary class.")
        result = {
            "status": "independent-source research test; no deployment validation",
            "freeze_sha256": sha256_file(run / "freeze.json"),
            "profile": profile,
            "all_development_overlap_rows_excluded": int(overlap.sum()),
            "reduced_representation_overlap_rows_retained": int((stable_overlap & ~overlap).sum()),
            "reduced_ambiguous_groups_retained": len(ambiguous),
            "reduced_ambiguous_rows_retained": int(frame["__hash"].isin(ambiguous).sum()),
            "evaluated_rows": len(frame),
            "models": {},
        }
        artifacts = {"multisource": run / "model.joblib"}
        if baseline:
            artifacts["original_unsw_baseline"] = Path(baseline)
        for name, path in artifacts.items():
            bundle = load_bundle(path)
            scores = bundle["model"].predict_proba(frame[bundle["features"]])[:, 1]
            result["models"][name] = {
                "model_sha256": sha256_file(path),
                "metrics": metrics(frame["__label"], scores, bundle["threshold"]),
                "attack_slices": attack_slices(frame, scores, bundle["threshold"]),
            }
        new = result["models"]["multisource"]["metrics"]
        result["point_targets_met"] = (
            new["attack_precision"] >= config["min_precision"]
            and new["false_positive_rate"] <= config["max_fpr"]
        )
    result["resources"] = {
        "wall_seconds": monitor.elapsed,
        "sampled_peak_process_rss_mib": monitor.peak / 2**20,
    }
    write_json(output, result)
    return result

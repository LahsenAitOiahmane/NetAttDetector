"""Development-only ablations and controlled training-stability diagnostics."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from threadpoolctl import threadpool_limits

from netattdetector.audit import STABLE_FEATURES, load_prepared
from netattdetector.data import FEATURES, fingerprints
from netattdetector.evaluation import attack_slices, metrics
from netattdetector.hardware import inventory
from netattdetector.io import sha256_file, write_json
from netattdetector.multisource import (
    candidate_model,
    development_split,
    robust_threshold,
    source_metrics,
)
from netattdetector.training import ResourceMonitor

TTL = {"MIN_TTL", "MAX_TTL"}
PORTS = {"L4_SRC_PORT", "L4_DST_PORT"}
PROTOCOL_IDS = {"PROTOCOL", "L7_PROTO"}
EXPORT_RATES = {
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
}
REMOVALS = {
    "full": set(),
    "no_ttl": TTL,
    "no_ports": PORTS,
    "no_protocol_ids": PROTOCOL_IDS,
    "no_export_rates": EXPORT_RATES,
    "no_ttl_ports": TTL | PORTS,
    "no_ttl_export_rates": TTL | EXPORT_RATES,
    "previous_reduced": TTL | PORTS | PROTOCOL_IDS,
}
GROUP_FEATURES = [f for f in STABLE_FEATURES if f not in EXPORT_RATES]


def study_groups(frame):
    """Retain all observations, grouping on the intersection of ablation inputs."""
    result = frame.copy()
    result["__hash"] = fingerprints(result, GROUP_FEATURES)
    return result


def feature_policy(name):
    if name not in REMOVALS:
        raise ValueError(f"Unknown feature policy: {name}")
    return [x for x in FEATURES if x not in REMOVALS[name]]


def array_digest(values):
    array = np.asarray(values)
    if array.dtype.kind not in "fiu":
        raise ValueError("Numerical payload signatures require numeric arrays.")
    array = np.array(array, dtype=array.dtype.newbyteorder("<"), order="C", copy=True)
    if array.dtype.kind == "f":
        array[np.isnan(array)] = np.nan
    header = json.dumps(
        {"shape": list(array.shape), "dtype": array.dtype.str}, sort_keys=True
    ).encode()
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def canonical_matrix(frame, features):
    return pd.DataFrame(
        np.ascontiguousarray(frame[features].to_numpy(dtype=np.float32)),
        columns=features,
        copy=False,
    )


def canonical_weights(frame):
    counts = frame.groupby("__source")["__label"].transform("size").to_numpy(dtype=np.float64)
    return np.asarray(len(frame) / (frame["__source"].nunique() * counts), dtype=np.float32)


def partition_signature(frame):
    weights = canonical_weights(frame)
    return {
        "rows": len(frame),
        "features_order": FEATURES,
        "ordered_values_sha256": array_digest(canonical_matrix(frame, FEATURES).to_numpy()),
        "ordered_labels_sha256": array_digest(frame["__label"].to_numpy(dtype=np.int8)),
        "ordered_full_hash_sha256": array_digest(frame["__full_hash"].to_numpy(dtype=np.uint64)),
        "ordered_weights_sha256": array_digest(weights),
        "source_label_counts": {
            f"{s}:{y}": len(g) for (s, y), g in frame.groupby(["__source", "__label"])
        },
    }


def measurement_audit(frame, source_audit):
    result = {
        "scope": "checksummed bounded development cohort; not a full re-audit of every raw row",
        "raw_source_audit": {
            s: {
                "rows_scanned": p["rows_scanned"],
                "nonfinite_or_coerced_values": p["coerced_or_nonfinite_nonmissing"],
                "out_of_range": p["out_of_range"],
            }
            for s, p in source_audit["sources"].items()
        },
        "groups": [],
    }
    for (source, label, proto), group in frame.groupby(["__source", "__label", "PROTOCOL"]):
        result["groups"].append(
            {
                "source": str(source),
                "label": int(label),
                "ip_protocol": float(proto),
                "rows": len(group),
                "min_ttl_zero_fraction": float(group.MIN_TTL.eq(0).mean()),
                "max_ttl_zero_fraction": float(group.MAX_TTL.eq(0).mean()),
                "ttl_min_exceeds_max_rows": int((group.MIN_TTL > group.MAX_TTL).sum()),
                "zero_duration_with_positive_bytes_rows": int(
                    (
                        group.FLOW_DURATION_MILLISECONDS.eq(0)
                        & ((group.IN_BYTES + group.OUT_BYTES) > 0)
                    ).sum()
                ),
                "tcp_window_nonzero_rows": int(
                    ((group.TCP_WIN_MAX_IN > 0) | (group.TCP_WIN_MAX_OUT > 0)).sum()
                ),
                "export_rate_quantiles": {
                    name: [float(x) for x in group[name].dropna().quantile([0.5, 0.99, 1.0])]
                    if group[name].notna().any()
                    else None
                    for name in sorted(EXPORT_RATES)
                },
            }
        )
    result["interpretation"] = (
        "Zero TTL/duration and unusual protocol/counter combinations are observations requiring PCAP/exporter validation; no values are relabeled or silently repaired. Zero duration can reflect quantization."
    )
    return result


def make_study_model(model_name, config, features, threads=None, stochastic=False, iterations=None):
    threads = threads or config["threads"]
    settings = {"threads": threads, "seed": config["model_seed"]}
    spec = {
        "model": model_name,
        "iterations": iterations or config["iterations"],
        "depth": 6 if model_name == "xgboost" else 8,
    }
    model = candidate_model(spec, settings)
    if model_name == "xgboost":
        model.named_steps["classifier"].set_params(
            subsample=0.8 if stochastic else 1.0, colsample_bytree=0.8 if stochastic else 1.0
        )
    else:
        model.named_steps["classifier"].set_params(early_stopping=False)
    model.steps.insert(
        -1, ("contiguous", FunctionTransformer(np.ascontiguousarray, validate=False))
    )
    return model


def transformed_overlaps(model, train, cal, validation, features):
    def hashes(frame):
        values = model[:-1].transform(canonical_matrix(frame, features))
        return pd.util.hash_pandas_object(pd.DataFrame(values), index=False).to_numpy()

    a, b, c = hashes(train), hashes(cal), hashes(validation)
    return {
        "train_calibration": int(np.isin(b, a).sum()),
        "train_validation": int(np.isin(c, a).sum()),
        "calibration_validation": int(np.isin(c, b).sum()),
    }


def validation_screen(record, config):
    if not record["calibration_feasible"] or any(record["transformed_overlap_rows"].values()):
        return False
    for metric in record["by_source"].values():
        intervals = metric["conditional_wilson_95"]
        c = metric["confusion_matrix"]
        if (
            metric["attack_precision"] < config["min_precision"]
            or metric["attack_recall"] < config["min_validation_recall"]
            or intervals["attack_precision"] is None
            or intervals["attack_precision"][0] < config["precision_lower_bound"]
            or intervals["false_positive_rate"][1] > config["max_fpr"]
            or c["tp"] + c["fp"] < config["min_alerts_per_source"]
        ):
            return False
    return True


def summarize_candidates(records, config):
    output = []
    for model_name in config["models"]:
        for policy in config["feature_policies"]:
            rows = [r for r in records if r["model"] == model_name and r["policy"] == policy]
            complete = {r["split_seed"] for r in rows} == set(config["split_seeds"])
            metrics_list = [m for r in rows for m in r["by_source"].values()]
            output.append(
                {
                    "model": model_name,
                    "policy": policy,
                    "complete": complete,
                    "completed_seeds": len(rows),
                    "passes_all_seeds": complete
                    and all(r["passes_validation_screen"] for r in rows),
                    "worst_source_seed_recall": min(
                        (m["attack_recall"] for m in metrics_list), default=None
                    ),
                    "worst_source_seed_precision": min(
                        (m["attack_precision"] for m in metrics_list), default=None
                    ),
                    "max_source_seed_fpr": max(
                        (m["false_positive_rate"] for m in metrics_list), default=None
                    ),
                    "mean_average_precision": float(
                        np.mean([m["average_precision"] for m in metrics_list])
                    )
                    if metrics_list
                    else None,
                    "feature_count": len(feature_policy(policy)),
                }
            )
    return output


def load_study_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("protocol_version") != 1 or not 1 <= config["threads"] <= 2:
        raise ValueError("Use study protocol 1 with one/two threads.")
    if not config["split_seeds"] or len(set(config["split_seeds"])) != len(config["split_seeds"]):
        raise ValueError("Partition seeds must be nonempty and unique.")
    if (
        len(config["split_seeds"]) > 3
        or len(config["feature_policies"]) > 8
        or len(config["models"]) > 2
    ):
        raise ValueError("Study exceeds the registered laptop grid limits.")
    if (
        set(config["models"]) - {"hist", "xgboost"}
        or not 0 < config["search_budget_seconds"] <= 3600
    ):
        raise ValueError("Invalid models or search budget.")
    if not 0 < config["min_validation_recall"] <= 1 or config["control_repeats"] != 2:
        raise ValueError("Invalid recall screen or control repeat count.")
    for key in ("models", "feature_policies"):
        if not config[key] or len(set(config[key])) != len(config[key]):
            raise ValueError(f"{key} must be nonempty and unique.")
    if not 1 <= config["iterations"] <= 150:
        raise ValueError("Iterations must be between 1 and 150.")
    if (
        not 0 < config["min_precision"] <= 1
        or not 0 < config["max_fpr"] < 1
        or not 0 < config["precision_lower_bound"] <= 1
        or config["min_alerts_per_source"] < 1
    ):
        raise ValueError("Invalid precision/FPR/confidence/support constraints.")
    for policy in config["feature_policies"]:
        feature_policy(policy)
    return config


def run_stability_study(prepared, config_path, output):
    config = load_study_config(config_path)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Study output must be fresh.")
    output.mkdir(parents=True, exist_ok=True)
    frame, audit = load_prepared(prepared)
    if (
        audit.get("cleaning_policy")
        != "full-feature conflicts/duplicates removed; reduced ambiguity retained"
    ):
        raise ValueError(
            "Use the corrected version-2 development cache, retaining reduced ambiguity."
        )
    source_names = {alias: Path(p["path"]).stem for alias, p in audit["sources"].items()}
    if set(source_names.values()) != {"nf-unsw-nb15-v2", "nf-ton-iot-v2"}:
        raise ValueError("This study accepts the registered UNSW/ToN development cache only.")
    frame["__source"] = frame["__source"].map(source_names)
    frame = study_groups(frame)
    grouping = {
        "features": GROUP_FEATURES,
        "groups": int(frame["__hash"].nunique()),
        "largest_group_rows": int(frame.groupby("__hash").size().max()),
        "rows_in_label_ambiguous_groups": int(
            frame.groupby("__hash")["__label"].transform("nunique").gt(1).sum()
        ),
        "policy": "preserve ambiguity; isolate common ablation inputs across partitions",
    }
    records, controls, transfers, signatures = [], [], [], {}
    control_scores, control_names = [], []
    hardware = inventory()
    deadline = time.perf_counter() + config["search_budget_seconds"]
    budget_exhausted = False
    with ResourceMonitor() as resources, threadpool_limits(limits=config["threads"]):
        write_json(output / "measurements.json", measurement_audit(frame, audit))
        seeds = list(dict.fromkeys([config["control_seed"], *config["split_seeds"]]))
        for seed in seeds:
            print(f"Preparing canonical group partitions at seed {seed}...", flush=True)
            train, cal, val = [
                p.sort_values("__full_hash", kind="stable").reset_index(drop=True)
                for p in development_split(frame, seed)
            ]
            signatures[str(seed)] = {
                name: partition_signature(p)
                for name, p in zip(
                    ("train", "calibration", "validation"), (train, cal, val), strict=True
                )
            }
            if seed == config["control_seed"] and config["run_controls"]:
                for stochastic in (True, False):
                    for threads in (1, 2):
                        name = f"{'stochastic' if stochastic else 'full'}-{threads}t"
                        previous = None
                        for repeat in range(config["control_repeats"]):
                            if time.perf_counter() >= deadline:
                                budget_exhausted = True
                                break
                            print(f"Stability control {name}, repeat {repeat + 1}...", flush=True)
                            with threadpool_limits(limits=threads):
                                model = make_study_model(
                                    "xgboost", config, FEATURES, threads, stochastic, 100
                                )
                                start = time.perf_counter()
                                model.fit(
                                    canonical_matrix(train, FEATURES),
                                    train["__label"],
                                    classifier__sample_weight=canonical_weights(train),
                                )
                                seconds = time.perf_counter() - start
                                scores = model.predict_proba(canonical_matrix(val, FEATURES))[:, 1]
                                transformed = model[:-1].transform(
                                    canonical_matrix(train, FEATURES)
                                )
                            raw = (
                                model.named_steps["classifier"]
                                .get_booster()
                                .save_raw(raw_format="ubj")
                            )
                            controls.append(
                                {
                                    "id": name,
                                    "repeat": repeat,
                                    "fit_seconds": seconds,
                                    "score_sha256": array_digest(scores),
                                    "stable_tree_sha256": hashlib.sha256(raw).hexdigest(),
                                    "transformed_train_sha256": array_digest(transformed),
                                    "transformed_validation_sha256": array_digest(
                                        model[:-1].transform(canonical_matrix(val, FEATURES))
                                    ),
                                    "transformed_features": transformed.shape[1],
                                    "max_repeat_probability_difference": float(
                                        np.max(np.abs(scores - previous))
                                    )
                                    if previous is not None
                                    else None,
                                }
                            )
                            control_scores.append(scores)
                            control_names.append(f"{name}:{repeat}")
                            previous = scores
                if control_scores:
                    np.savez_compressed(
                        output / "control-scores.npz",
                        scores=np.stack(control_scores),
                        names=np.asarray(control_names),
                        validation_full_hash=val["__full_hash"].to_numpy(),
                        validation_labels=val["__label"].to_numpy(),
                    )
            if seed == config["control_seed"] and config["run_source_transfer"]:
                for policy in ("full", "no_ttl"):
                    features = feature_policy(policy)
                    for source in sorted(train["__source"].unique()):
                        if time.perf_counter() >= deadline:
                            budget_exhausted = True
                            break
                        origin_train = train.loc[train["__source"] == source].reset_index(drop=True)
                        origin_cal = cal.loc[cal["__source"] == source].reset_index(drop=True)
                        target = val.loc[val["__source"] != source].reset_index(drop=True)
                        model = make_study_model("xgboost", config, features)
                        print(f"Known-source transfer {policy}, training {source}...", flush=True)
                        model.fit(
                            canonical_matrix(origin_train, features),
                            origin_train["__label"],
                            classifier__sample_weight=canonical_weights(origin_train),
                        )
                        threshold, feasible = robust_threshold(
                            origin_cal,
                            model.predict_proba(canonical_matrix(origin_cal, features))[:, 1],
                            config,
                        )
                        scores = model.predict_proba(canonical_matrix(target, features))[:, 1]
                        transfers.append(
                            {
                                "policy": policy,
                                "training_source": source,
                                "target_source": str(target["__source"].iloc[0]),
                                "calibration_feasible": feasible,
                                "metrics": metrics(target["__label"], scores, threshold),
                                "scope": "known development-source transfer; no target fitting/threshold tuning",
                            }
                        )
            if seed not in config["split_seeds"]:
                continue
            for model_name in config["models"]:
                for policy in config["feature_policies"]:
                    if time.perf_counter() >= deadline:
                        budget_exhausted = True
                        break
                    print(
                        f"Ablation seed={seed}, model={model_name}, policy={policy}...", flush=True
                    )
                    features = feature_policy(policy)
                    model = make_study_model(model_name, config, features)
                    start = time.perf_counter()
                    model.fit(
                        canonical_matrix(train, features),
                        train["__label"],
                        classifier__sample_weight=canonical_weights(train),
                    )
                    seconds = time.perf_counter() - start
                    threshold, feasible = robust_threshold(
                        cal, model.predict_proba(canonical_matrix(cal, features))[:, 1], config
                    )
                    scores = model.predict_proba(canonical_matrix(val, features))[:, 1]
                    row = {
                        "model": model_name,
                        "policy": policy,
                        "split_seed": seed,
                        "feature_count": len(features),
                        "fit_seconds": seconds,
                        "threshold": threshold,
                        "calibration_feasible": feasible,
                        "by_source": source_metrics(val, scores, threshold),
                        "pooled_validation": metrics(val["__label"], scores, threshold),
                        "transformed_overlap_rows": transformed_overlaps(
                            model, train, cal, val, features
                        ),
                        "attack_slices_by_source": {
                            str(source): attack_slices(
                                part.reset_index(drop=True), scores[part.index], threshold
                            )
                            for source, part in val.groupby("__source")
                        },
                    }
                    row["passes_validation_screen"] = validation_screen(row, config)
                    records.append(row)
                    write_json(
                        output / "progress.json",
                        {"controls": controls, "transfer": transfers, "records": records},
                    )
            if budget_exhausted:
                break
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            dirty = bool(
                subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
            )
        except (OSError, subprocess.CalledProcessError):
            revision, dirty = None, None
    result = {
        "scope": "development diagnostics only; no CSE or new independent-source scoring",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_revision": revision,
        "git_worktree_dirty": dirty,
        "config": config,
        "config_sha256": sha256_file(config_path),
        "source_sha256": {source_names[s]: p["sha256"] for s, p in audit["sources"].items()},
        "cache_sha256": audit["cache_sha256"],
        "hardware": hardware,
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost")
        },
        "partition_payloads": signatures,
        "controls": controls,
        "known_source_transfer": transfers,
        "records": records,
        "summary": summarize_candidates(records, config),
        "budget_exhausted": budget_exhausted,
        "resources": {
            "wall_seconds": resources.elapsed,
            "sampled_peak_process_rss_mib": resources.peak / 2**20,
        },
    }
    result["grouping"] = grouping
    write_json(output / "study.json", result)
    return result


def compare_studies(first, second):
    first, second = Path(first), Path(second)
    a = json.loads((first / "study.json").read_text())
    b = json.loads((second / "study.json").read_text())
    seed = str(a["config"]["control_seed"])
    if a["config_sha256"] != b["config_sha256"] or a["source_sha256"] != b["source_sha256"]:
        raise ValueError("Study configurations or raw sources differ.")
    same_input = a["partition_payloads"] == b["partition_payloads"]
    comparisons = []
    with (
        np.load(first / "control-scores.npz", allow_pickle=False) as x,
        np.load(second / "control-scores.npz", allow_pickle=False) as y,
    ):
        same_rows = np.array_equal(
            x["validation_full_hash"], y["validation_full_hash"]
        ) and np.array_equal(x["validation_labels"], y["validation_labels"])
        if not same_input or not same_rows:
            raise ValueError(
                "Control inputs/order/labels/weights differ; scores are not comparable."
            )
        left = {name: scores for name, scores in zip(x["names"], x["scores"], strict=True)}
        right = {name: scores for name, scores in zip(y["names"], y["scores"], strict=True)}
        expected = {
            f"{mode}-{t}t:{r}" for mode in ("stochastic", "full") for t in (1, 2) for r in range(2)
        }
        if left.keys() != expected or right.keys() != expected:
            raise ValueError("Both studies must contain all eight registered controls.")
        left_meta = {f"{r['id']}:{r['repeat']}": r for r in a["controls"]}
        right_meta = {f"{r['id']}:{r['repeat']}": r for r in b["controls"]}
        for saved, report, metadata, scores in ((x, a, left_meta, left), (y, b, right_meta, right)):
            payload = report["partition_payloads"][seed]["validation"]
            if (
                array_digest(saved["validation_full_hash"]) != payload["ordered_full_hash_sha256"]
                or array_digest(saved["validation_labels"]) != payload["ordered_labels_sha256"]
            ):
                raise ValueError("Control row identities/labels do not match the recorded payload.")
            if any(
                array_digest(value) != metadata[name]["score_sha256"]
                for name, value in scores.items()
            ):
                raise ValueError("Control scores do not match their recorded hashes.")
        for name in sorted(expected):
            for key in ("transformed_train_sha256", "transformed_validation_sha256"):
                if left_meta[name][key] != right_meta[name][key]:
                    raise ValueError("Post-preprocessing training/probe matrices differ.")
            comparisons.append(
                {
                    "control": str(name),
                    "max_probability_difference": float(np.max(np.abs(left[name] - right[name]))),
                    "mean_probability_difference": float(np.mean(np.abs(left[name] - right[name]))),
                    "exact_score_match": bool(np.array_equal(left[name], right[name])),
                    "stable_tree_bytes_match": left_meta[name]["stable_tree_sha256"]
                    == right_meta[name]["stable_tree_sha256"],
                    "decision_disagreements_at_diagnostic_0_5": int(
                        np.count_nonzero((left[name] >= 0.5) != (right[name] >= 0.5))
                    ),
                }
            )
    within_run = {}
    for label, scores in (("first", left), ("second", right)):
        within_run[label] = {
            "max_repeat_probability_difference": {
                f"{mode}-{t}t": float(
                    np.max(np.abs(scores[f"{mode}-{t}t:0"] - scores[f"{mode}-{t}t:1"]))
                )
                for mode in ("stochastic", "full")
                for t in (1, 2)
            },
            "max_one_vs_two_thread_probability_difference": {
                mode: float(np.max(np.abs(scores[f"{mode}-1t:0"] - scores[f"{mode}-2t:0"])))
                for mode in ("stochastic", "full")
            },
        }
    key_fields = ("model", "policy", "split_seed")
    a_rows = {tuple(r[k] for k in key_fields): r for r in a.get("records", [])}
    b_rows = {tuple(r[k] for k in key_fields): r for r in b.get("records", [])}
    ablation_differences = []
    for key in sorted(a_rows.keys() & b_rows.keys()):
        first_row, second_row = a_rows[key], b_rows[key]
        if first_row["by_source"] != second_row["by_source"]:
            ablation_differences.append(
                {
                    "model": key[0],
                    "policy": key[1],
                    "split_seed": key[2],
                    "thresholds": [first_row["threshold"], second_row["threshold"]],
                    "same_source_confusion_matrices": all(
                        first_row["by_source"][s]["confusion_matrix"]
                        == second_row["by_source"][s]["confusion_matrix"]
                        for s in first_row["by_source"]
                    ),
                    "max_absolute_source_metric_difference": {
                        m: max(
                            abs(first_row["by_source"][s][m] - second_row["by_source"][s][m])
                            for s in first_row["by_source"]
                        )
                        for m in (
                            "attack_precision",
                            "attack_recall",
                            "false_positive_rate",
                            "average_precision",
                        )
                    },
                }
            )
    return {
        "same_canonical_inputs_order_labels_weights": same_input,
        "same_validation_rows": same_rows,
        "controls": comparisons,
        "within_run": within_run,
        "same_all_partition_payloads": a["partition_payloads"] == b["partition_payloads"],
        "same_ablation_grid": a_rows.keys() == b_rows.keys(),
        "compared_ablation_fits": len(a_rows.keys() & b_rows.keys()),
        "ablation_metric_differences": ablation_differences,
        "scope": "controlled fitting comparison, not independent attack-detection evidence",
    }

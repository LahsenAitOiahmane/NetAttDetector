"""Aggregate native error slices and fixed-thread controls, without altering the baseline."""

import json
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np

from netattdetector.io import sha256_file, write_json
from netattdetector.measurement import CORE_FEATURES
from netattdetector.native_cohort import verify_native_cohort
from netattdetector.native_study import (
    connection_scores,
    evaluate_scores,
    fit_one,
    load_frames,
    load_study_config,
)
from netattdetector.resources import ResourceMonitor

CONFIG = "configs/native-errors.json"
SERVICES = {
    22: "ssh_22",
    23: "telnet_23",
    53: "dns_53",
    80: "http_80",
    123: "ntp_123",
    443: "https_443",
    1883: "mqtt_1883",
    5353: "mdns_5353",
}


def compare_predictions(left, right, threshold):
    a, b = np.asarray(left), np.asarray(right)
    if (
        a.shape != b.shape
        or a.ndim != 1
        or not len(a)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Repeat score vectors must have matching finite rows.")
    return {
        "max_absolute_score_difference": float(np.abs(a - b).max()),
        "frozen_threshold_decision_differences": int(
            np.count_nonzero((a >= threshold) != (b >= threshold))
        ),
        "scores_exactly_identical": bool(np.array_equal(a, b)),
    }


def error_slices(frame, scores, threshold):
    # Validate consistent group labels/details before picking one segment per accepted UID.
    connection_scores(frame, scores)
    context = frame.reset_index(drop=True).copy()
    context["score"] = np.asarray(scores)
    context = context.loc[context.groupby("group_hash", sort=True)["score"].idxmax()].copy()
    label, flagged = context["label"].to_numpy(), context["score"].to_numpy() >= threshold
    context["outcome"] = np.select(
        [(label == 0) & ~flagged, (label == 0) & flagged, (label == 1) & ~flagged],
        ["tn", "fp", "fn"],
        default="tp",
    )
    packets = context["src2dst_packets"] + context["dst2src_packets"]
    slices = {
        "protocol": context["protocol"].map(
            lambda value: {6: "TCP", 17: "UDP"}.get(value, f"other_{value}")
        ),
        "observed_destination_service": context["dst_port"].map(
            lambda value: SERVICES.get(value, "other")
        ),
        "source_port_range": context["src_port"].map(
            lambda value: (
                "0-1023" if value < 1024 else "1024-49151" if value < 49152 else "49152-65535"
            )
        ),
        "packet_count_range": packets.map(
            lambda value: (
                "1" if value == 1 else "2-4" if value <= 4 else "5-20" if value <= 20 else "21+"
            )
        ),
        "duration": context["bidirectional_duration_ms"].map(
            lambda value: "zero_ms" if value == 0 else "positive_ms"
        ),
        "publisher_detail": context["detail"],
    }
    output = {}
    for name, categories in slices.items():
        table = {}
        for category in sorted(categories.unique()):
            part = context[categories == category]
            table[str(category)] = {
                "connections": len(part),
                **{key: int((part["outcome"] == key).sum()) for key in ("tn", "fp", "fn", "tp")},
            }
        output[name] = table
    quantiles = {}
    for name in ("tn", "fp", "fn", "tp"):
        values = context.loc[context["outcome"] == name, "score"].to_numpy()
        quantiles[name] = {
            "connections": len(values),
            "score_quantiles_min_p25_p50_p75_max": np.quantile(
                values, [0, 0.25, 0.5, 0.75, 1]
            ).tolist()
            if len(values)
            else None,
        }
    return {
        "connections": len(context),
        "threshold": threshold,
        "slices": output,
        "score_distributions": quantiles,
        "scope": "One highest-scoring accepted segment per UID, first stored on ties; observed direction, not verified client/service role. Aggregate bins only, no endpoint/UID/feature rows.",
    }


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if any(
        config.get(name) != value
        for name, value in {
            "schema_version": 1,
            "study_id": "native-error-diagnostic-v1",
            "baseline_config": "configs/native-study.json",
            "baseline_freeze": "reports/evidence/native-study/freeze.json",
            "baseline_results": "reports/evidence/native-study/study.json",
            "maximum_fits": 3,
            "training_threads": [2, 1, 1],
            "score_tolerance": 1e-7,
            "deployment_authorized": False,
            "public_model_release_authorized": False,
        }.items()
    ):
        raise ValueError("Use the registered three-fit native error diagnostic.")
    return config


def run_native_errors(cohort, output, config_path=CONFIG):
    config = load_config(config_path)
    baseline = load_study_config(config["baseline_config"])
    freeze = json.loads(Path(config["baseline_freeze"]).read_text(encoding="utf-8"))
    old = json.loads(Path(config["baseline_results"]).read_text(encoding="utf-8"))
    if sha256_file(config["baseline_results"]) != freeze["study_sha256"]:
        raise ValueError("Saved native baseline result differs from its diagnostic freeze.")
    if (
        freeze["specification"]["model"] != "logistic"
        or freeze["deployment_authorized"] is not False
    ):
        raise ValueError("Only the frozen research logistic specification is registered here.")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    verification = verify_native_cohort(cohort, output / "cohort-verification.json")
    if verification["audit"]["rows_logical_sha256"] != baseline["expected_logical_rows_sha256"]:
        raise ValueError("Error diagnostic cohort identity changed.")
    threshold = freeze["threshold"]
    records, predictions, coefficients = [], [], []
    with ResourceMonitor(include_children=True) as resources:
        frames = load_frames(cohort)
        for number, threads in enumerate(config["training_threads"]):
            if time.perf_counter() - resources.start >= baseline["maximum_wall_seconds"]:
                raise ValueError("Diagnostic wall budget exhausted before another fit.")
            model, scores, report = fit_one(
                freeze["specification"], frames, baseline, threads=threads
            )
            predictions.append(scores["validation"])
            coefficients.append(model.named_steps["model"].coef_.copy())
            records.append(
                {
                    "run": number,
                    "training_threads": threads,
                    "fit_seconds": report["fit_seconds"],
                    "calibration_selected_threshold": report["operating_point"]["threshold"],
                    "validation_at_original_frozen_threshold": evaluate_scores(
                        frames["validation"], scores["validation"], threshold
                    ),
                    "calibration_at_original_frozen_threshold": evaluate_scores(
                        frames["calibration"], scores["calibration"], threshold
                    ),
                }
            )
            if number == 0:
                errors = error_slices(frames["validation"], scores["validation"], threshold)
                original = next(
                    row for row in old["records"] if row["id"] == freeze["selected_diagnostic_id"]
                )
                replay_same_metrics = (
                    records[0]["validation_at_original_frozen_threshold"]
                    == original["metrics"]["validation"]
                )
            if number == 1:
                # Private fixed-thread artifact for offline research only; never in evidence allowlists.
                joblib.dump(
                    {
                        "format": "native-fixed-thread-diagnostic-v1",
                        "schema": "nfstream-core-v1",
                        "features": CORE_FEATURES,
                        "model": model,
                        "threshold": report["operating_point"]["threshold"],
                        "specification": freeze["specification"],
                        "training_threads": 1,
                        "deployment_authorized": False,
                        "public_model_release_authorized": False,
                    },
                    output / "fixed-thread.joblib",
                    compress=3,
                )
            del model
        repeated = compare_predictions(predictions[1], predictions[2], threshold)
        repeated["coefficient_max_absolute_difference"] = float(
            np.abs(coefficients[1] - coefficients[2]).max()
        )
        repeated["calibration_threshold_identical"] = (
            records[1]["calibration_selected_threshold"]
            == records[2]["calibration_selected_threshold"]
        )
        repeated["passed"] = (
            repeated["max_absolute_score_difference"] <= config["score_tolerance"]
            and repeated["frozen_threshold_decision_differences"] == 0
        )
        cross_thread = compare_predictions(predictions[0], predictions[1], threshold)
        cross_thread["coefficient_max_absolute_difference"] = float(
            np.abs(coefficients[0] - coefficients[1]).max()
        )
    result = {
        "study_id": config["study_id"],
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config_sha256": sha256_file(config_path),
        "baseline_freeze_sha256": sha256_file(config["baseline_freeze"]),
        "cohort_logical_sha256": baseline["expected_logical_rows_sha256"],
        "original_frozen_threshold": threshold,
        "completed_fits": len(records),
        "records": records,
        "two_thread_validation_metrics_replay_identical": replay_same_metrics,
        "fixed_one_thread_repeat": repeated,
        "two_vs_one_thread": cross_thread,
        "validation_errors_at_original_threshold": errors,
        "private_fixed_thread_artifact": {
            "sha256": sha256_file(output / "fixed-thread.joblib"),
            "bytes": (output / "fixed-thread.joblib").stat().st_size,
        },
        "resources": {
            "wall_seconds": resources.elapsed,
            "sampled_peak_process_tree_rss_sum_mib": resources.peak_tree / 2**20,
            "scope": "Frame loading, three fits and aggregate diagnostics; excludes imports, preparation, verification and downloads. 50 ms RSS samples are not hard process limits.",
        },
        "resource_screen_met": resources.elapsed < baseline["maximum_wall_seconds"]
        and resources.peak_tree < baseline["sampled_memory_screen_mib"] * 2**20,
        "baseline_acceptance_unchanged": True,
        "deployment_authorized": False,
        "public_model_release_authorized": False,
        "external_test_accessed": False,
        "candidate_policy_promoted": False,
    }
    write_json(output / "diagnostics-config.json", config)
    write_json(output / "diagnostics.json", result)
    if not repeated["passed"] or not result["resource_screen_met"]:
        raise ValueError(
            "Fixed-thread/resource diagnostic screen failed; aggregate evidence retained."
        )
    return result

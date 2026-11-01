"""Exact observed connection information ceilings; no model or threshold is produced."""

import json
import math
import subprocess
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from netattdetector.io import sha256_file, write_json
from netattdetector.measurement import CORE_FEATURES
from netattdetector.resources import ResourceMonitor
from netattdetector.statistical import EXTRA_FEATURES, FEATURES
from netattdetector.statistical_cohort import (
    configure_database,
    connect_readonly,
    verify_statistical_inputs,
)

CONFIG = "configs/data/information-bound.json"
POLICIES = {
    "core11_control": CORE_FEATURES,
    "statistics27": FEATURES,
    "statistics25_no_ports": FEATURES[2:],
}


def connection_information_bound(
    groups, labels, values, *, maximum_fpr="0.005", minimum_recall="0.8", max_dp_updates=10_000_000
):
    """Solve a relaxation over exact vector-set groups, retaining no oracle decisions."""
    values, labels = np.asarray(values), np.asarray(labels)
    if (
        values.ndim != 2
        or not 0 < len(values) <= 250_000
        or values.shape[1] == 0
        or len(groups) != len(values)
        or labels.shape != (len(values),)
        or not np.isfinite(values).all()
        or not set(labels.tolist()) <= {0, 1}
    ):
        raise ValueError("Information bounds require matching finite rows and binary labels.")
    if values.dtype not in (np.dtype("<f8"), np.dtype("<f4")):
        raise ValueError("Specify an exact float64/float32 input representation.")
    fpr, recall = Fraction(str(maximum_fpr)), Fraction(str(minimum_recall))
    if not 0 <= fpr <= 1 or not 0 < recall <= 1:
        raise ValueError("Invalid registered operating-point fractions.")
    # Python byte/tuple equality resolves hash collisions; no digest merges feature sets.
    interned, connections = {}, {}
    for uid, label, row in zip(groups, labels, values, strict=True):
        if not isinstance(uid, str) or not uid:
            raise ValueError("Connection group identities must be nonempty strings.")
        vector = row.tobytes()
        vector = interned.setdefault(vector, vector)
        if uid not in connections:
            connections[uid] = (int(label), set())
        elif connections[uid][0] != label:
            raise ValueError("A connection group has inconsistent binary labels.")
        connections[uid][1].add(vector)
    clusters = defaultdict(lambda: [0, 0])
    for label, vectors in connections.values():
        clusters[tuple(sorted(vectors))][label] += 1
    benign = sum(counts[0] for counts in clusters.values())
    attacks = sum(counts[1] for counts in clusters.values())
    if not benign or not attacks:
        raise ValueError("Information bounds need both accepted connection classes.")
    free = sum(counts[1] for counts in clusters.values() if counts[0] == 0)
    mixed = [(counts[0], counts[1]) for counts in clusters.values() if counts[0] and counts[1]]
    updates = sum(benign - cost + 1 for cost, _ in mixed)
    if updates > max_dp_updates:
        raise ValueError(
            "Exact information-bound update budget exceeded; no approximation returned."
        )
    best = np.full(benign + 1, free, dtype=np.int64)
    for cost, benefit in mixed:
        best[cost:] = np.maximum(best[cost:], best[:-cost] + benefit)
    if int(best[-1]) != attacks:
        raise ValueError("Information-bound accounting invariant failed.")
    budget = math.floor(fpr * benign)
    required = math.ceil(recall * attacks)
    minimum_fp = int(np.flatnonzero(best >= required)[0])
    points = np.r_[0, np.flatnonzero(best[1:] != best[:-1]) + 1]
    return {
        "accepted_connections": benign + attacks,
        "benign_connections": benign,
        "attack_connections": attacks,
        "exact_vector_set_clusters": len(clusters),
        "mixed_label_clusters": len(mixed),
        "benign_connections_in_mixed_clusters": sum(cost for cost, _ in mixed),
        "attack_connections_in_mixed_clusters": sum(value for _, value in mixed),
        "optimistic_free_attack_connections": free,
        "maximum_fpr": float(fpr),
        "exact_budget_fraction": str(fpr),
        "benign_flag_budget": budget,
        "optimistic_max_true_positives_at_budget": int(best[budget]),
        "optimistic_max_recall_at_budget": int(best[budget]) / attacks,
        "minimum_recall": float(recall),
        "exact_recall_fraction": str(recall),
        "required_true_positives": required,
        "minimum_benign_flags_for_target_in_relaxation": minimum_fp,
        "minimum_fpr_for_target_in_relaxation": minimum_fp / benign,
        "target_not_ruled_out_by_relaxation": int(best[budget]) >= required,
        "learnable_operating_point_proven": False,
        "precision_constraints_ignored": True,
        "dp_updates": updates,
        "aggregate_upper_frontier": [
            {
                "benign_flag_budget": int(index),
                "optimistic_max_true_positives": int(best[index]),
                "optimistic_max_recall": int(best[index]) / attacks,
            }
            for index in points
        ],
        "scope": "Optimistic upper bound for this exact observed representation and UID-max family only; distinct feature-set groups choose independently even when they share vectors. Not a trained classifier, threshold, attack truth or generalization estimate.",
    }


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if any(
        config.get(name) != value
        for name, value in {
            "schema_version": 1,
            "study_id": "connection-information-bound-v1",
            "statistical_config": "configs/data/statistical-features.json",
            "expected_core_logical_sha256": "9be2a2d5c39b3cb78a31fec40118971ba03b0e62b398e1151f7480ce8c312d59",
            "expected_statistical_logical_sha256": "a4e6d0ff836814acadd4cc778b0b63320fe5dc38cc080f040caaeb95db189259",
            "partitions": ["calibration", "validation"],
            "feature_policies": list(POLICIES),
            "representations": ["float64", "float32"],
            "maximum_false_positive_rate": "0.005",
            "minimum_recall": "0.8",
            "maximum_partition_rows": 250_000,
            "maximum_dp_updates": 10_000_000,
            "maximum_wall_seconds": 600,
            "sampled_memory_screen_mib": 512,
            "training_enabled": False,
            "deployment_authorized": False,
        }.items()
    ):
        raise ValueError("Use the registered no-fit information-bound configuration.")
    return config


def run_information_bound(cohort, sidecar, output, config_path=CONFIG):
    config = load_config(config_path)
    cohort, sidecar, output = map(Path, (cohort, sidecar, output))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    verify_statistical_inputs(cohort, sidecar, output, config)
    records = []
    with ResourceMonitor(include_children=True) as resources:
        db = connect_readonly(sidecar / "statistics.sqlite")
        try:
            configure_database(db)
            db.execute(
                "ATTACH DATABASE ? AS core",
                ((cohort / "cohort.sqlite").resolve().as_uri() + "?mode=ro",),
            )
            fields = [
                "c.group_hash",
                "c.label",
                *[f"c.{name}" for name in CORE_FEATURES],
                *[f"f.{name}" for name in EXTRA_FEATURES],
            ]
            for split in config["partitions"]:
                rows = db.execute(
                    "SELECT "
                    + ",".join(fields)
                    + " FROM features f JOIN core.flows c ON f.ordinal=c.ordinal AND f.flow_hash=c.flow_hash WHERE c.split=? AND c.status='matched' ORDER BY c.ordinal LIMIT ?",
                    (split, config["maximum_partition_rows"] + 1),
                ).fetchall()
                if len(rows) > config["maximum_partition_rows"]:
                    raise ValueError("Information-bound partition completeness limit reached.")
                groups, labels = [row[0] for row in rows], np.asarray([row[1] for row in rows])
                full = np.asarray([row[2:] for row in rows], dtype="<f8")
                for policy, fields_selected in POLICIES.items():
                    indices = [FEATURES.index(name) for name in fields_selected]
                    for representation in config["representations"]:
                        if time.perf_counter() - resources.start >= config["maximum_wall_seconds"]:
                            raise ValueError("Information-bound wall budget exhausted.")
                        matrix = np.ascontiguousarray(
                            full[:, indices], dtype="<f8" if representation == "float64" else "<f4"
                        )
                        bound = connection_information_bound(
                            groups,
                            labels,
                            matrix,
                            maximum_fpr=config["maximum_false_positive_rate"],
                            minimum_recall=config["minimum_recall"],
                            max_dp_updates=config["maximum_dp_updates"],
                        )
                        records.append(
                            {
                                "split": split,
                                "feature_policy": policy,
                                "representation": representation,
                                **bound,
                            }
                        )
        finally:
            db.close()
    result = {
        "study_id": config["study_id"],
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config_sha256": sha256_file(config_path),
        "core_logical_sha256": config["expected_core_logical_sha256"],
        "statistical_logical_sha256": config["expected_statistical_logical_sha256"],
        "records": records,
        "model_fits": 0,
        "model_or_threshold_produced": False,
        "resources": {
            "wall_seconds": resources.elapsed,
            "sampled_peak_process_tree_rss_sum_mib": resources.peak_tree / 2**20,
            "scope": "Two small partition reads, exact byte-set grouping and knapsack bounds; excludes imports, acquisition/preparation and input verification. 50 ms RSS samples are not OS-enforced limits.",
        },
        "resource_screen_met": resources.elapsed < config["maximum_wall_seconds"]
        and resources.peak_tree < config["sampled_memory_screen_mib"] * 2**20,
        "training_authorized": False,
        "deployment_authorized": False,
        "external_test_accessed": False,
        "candidate_policy_promoted": False,
    }
    write_json(output / "configuration.json", config)
    write_json(output / "information-bound.json", result)
    if not result["resource_screen_met"]:
        raise ValueError(
            "Information-bound resource screen failed; complete aggregate evidence retained."
        )
    return result

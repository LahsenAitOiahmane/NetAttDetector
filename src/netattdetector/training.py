"""CPU-bounded experiments with a validation leaderboard and untouched test set."""

import hashlib
import importlib.metadata
import platform
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from netattdetector.data import FEATURES, SCHEMA_VERSION, describe_sample, load_sample
from netattdetector.evaluation import attack_slices, choose_threshold, metrics
from netattdetector.hardware import inventory
from netattdetector.io import write_json


@dataclass
class TrainConfig:
    max_rows: int = 200_000
    chunk_rows: int = 50_000
    seed: int = 42
    threads: int = 2
    models: tuple = ("sgd", "hist", "extra_trees")
    split: str = "random"
    group_column: str | None = None
    min_precision: float = 0.98
    max_fpr: float = 0.005
    trees: int = 100
    max_depth: int = 14
    min_samples_leaf: int = 5
    boosting_iterations: int = 150
    member: str | None = None

    def validate(self):
        if not 1 <= self.threads <= 8:
            raise ValueError("threads must be between 1 and 8.")
        if not 100 <= self.max_rows <= 2_000_000 or self.chunk_rows < 1:
            raise ValueError("max_rows must be 100..2000000 and chunk_rows must be positive.")
        if not 0 < self.min_precision <= 1 or not 0 <= self.max_fpr < 1:
            raise ValueError("Invalid precision/FPR constraints.")
        if self.split not in {"random", "group", "temporal"}:
            raise ValueError("split must be random, group or temporal.")
        if self.split != "random" and not self.group_column:
            raise ValueError("group/temporal split requires group_column.")
        if not self.models or set(self.models) - {"sgd", "hist", "extra_trees", "xgboost"}:
            raise ValueError("Choose models from sgd, hist, extra_trees, xgboost.")
        if min(self.trees, self.max_depth, self.min_samples_leaf, self.boosting_iterations) < 1:
            raise ValueError("Estimator sizes must be positive.")


class ResourceMonitor:
    """Sample this process's RSS; not an OS-wide or exact peak-memory measurement."""

    def __enter__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.start = time.perf_counter()
        self.done = threading.Event()

        def sample():
            while not self.done.wait(0.05):
                self.peak = max(self.peak, self.process.memory_info().rss)

        self.worker = threading.Thread(target=sample, daemon=True)
        self.worker.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.done.set()
        self.worker.join()
        self.elapsed = time.perf_counter() - self.start


def split_data(frame, config):
    indices = np.arange(len(frame))
    if config.split == "random":
        development, test = train_test_split(
            indices, test_size=0.2, stratify=frame["__label"], random_state=config.seed
        )
        train, validation = train_test_split(
            development,
            test_size=0.25,
            stratify=frame.iloc[development]["__label"],
            random_state=config.seed,
        )
    elif config.split == "group":
        groups = frame["__group"]
        development, test = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.seed).split(
                frame, groups=groups
            )
        )
        tr, va = next(
            GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=config.seed).split(
                development, groups=groups.iloc[development]
            )
        )
        train, validation = development[tr], development[va]
    else:
        times = pd.to_datetime(frame["__group"], utc=True, errors="raise")
        unique = np.sort(times.unique())
        if len(unique) < 5:
            raise ValueError("Temporal split needs at least five distinct timestamps.")
        first, second = unique[int(0.6 * len(unique))], unique[int(0.8 * len(unique))]
        train = indices[times < first]
        validation = indices[(times >= first) & (times < second)]
        test = indices[times >= second]
    parts = [frame.iloc[x].reset_index(drop=True) for x in (train, validation, test)]
    for part in parts:
        if part["__label"].nunique() != 2 or part["__label"].value_counts().min() < 2:
            raise ValueError(
                "Each split needs at least two examples of each class; change sample/split."
            )
    for i in range(3):
        for j in range(i):
            if set(parts[i]["__hash"]) & set(parts[j]["__hash"]):
                raise ValueError("Feature duplicate overlap between splits.")
            if config.split in {"group", "temporal"} and set(parts[i]["__group"]) & set(
                parts[j]["__group"]
            ):
                raise ValueError("Group overlap between splits.")
    return parts


def make_model(name, config):
    if name == "dummy":
        estimator = DummyClassifier(strategy="prior")
    elif name == "sgd":
        estimator = SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            max_iter=1000,
            tol=1e-3,
            random_state=config.seed,
        )
    elif name == "hist":
        estimator = HistGradientBoostingClassifier(
            max_iter=config.boosting_iterations,
            max_leaf_nodes=31,
            max_depth=8,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=config.seed,
        )
    elif name == "extra_trees":
        estimator = ExtraTreesClassifier(
            n_estimators=config.trees,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=config.threads,
            random_state=config.seed,
        )
    elif name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ValueError(
                "XGBoost requested but not installed. Run uv sync --frozen --extra boosting."
            ) from error

        estimator = XGBClassifier(
            n_estimators=config.boosting_iterations,
            max_depth=6,
            learning_rate=0.08,
            tree_method="hist",
            n_jobs=config.threads,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=config.seed,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    steps = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("variance", VarianceThreshold()),
    ]
    if name == "sgd":
        steps.append(("scale", StandardScaler()))
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def score(model, frame):
    return model.predict_proba(frame[FEATURES])[:, 1]


def split_summary(frame):
    return {
        "rows": len(frame),
        "label_counts": {str(k): int(v) for k, v in frame["__label"].value_counts().items()},
        "fingerprint_sha256": hashlib.sha256(
            np.sort(frame["__hash"].to_numpy()).tobytes()
        ).hexdigest(),
    }


def run_experiment(path, output, config, external=None, external_member=None):
    config.validate()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty to preserve previous experiment evidence.")
    output.mkdir(parents=True, exist_ok=True)
    hardware = inventory()
    with ResourceMonitor() as resources, threadpool_limits(limits=config.threads):
        frame, profile = load_sample(
            path,
            config.max_rows,
            config.chunk_rows,
            config.seed,
            config.member,
            config.group_column,
        )
        profile.update(describe_sample(frame))
        write_json(output / "data-profile.json", profile)
        train, validation, test = split_data(frame, config)
        leaderboard = []
        champion = None
        champion_record = None
        for name in ("dummy", *dict.fromkeys(config.models)):
            print(f"Training {name} on {len(train):,} flows...", flush=True)
            model = make_model(name, config)
            start = time.perf_counter()
            model.fit(train[FEATURES], train["__label"])
            fit_seconds = time.perf_counter() - start
            scores = score(model, validation)
            threshold, feasible = choose_threshold(
                validation["__label"], scores, config.min_precision, config.max_fpr
            )
            record = {
                "model": name,
                "fit_seconds": fit_seconds,
                "validation_constraints_met": feasible,
                "validation": metrics(validation["__label"], scores, threshold),
            }
            leaderboard.append(record)

            def rank(item):
                m = item["validation"]
                return (
                    item["validation_constraints_met"],
                    m["attack_recall"] if item["validation_constraints_met"] else m["attack_f1"],
                    m["average_precision"],
                    -item["fit_seconds"],
                )

            if name != "dummy" and (champion is None or rank(record) > rank(champion_record)):
                champion, champion_record = model, record
        threshold = champion_record["validation"]["threshold"]
        test_scores = score(champion, test)
        test_metrics = metrics(test["__label"], test_scores, threshold)
        # Repeated batched inference after warmup, excluding model load and CSV parsing.
        batch = test.iloc[: min(2048, len(test))]
        score(champion, batch)
        timings = []
        for _ in range(7):
            start = time.perf_counter()
            score(champion, batch)
            timings.append(time.perf_counter() - start)
        bundle = {
            "format_version": 1,
            "schema": SCHEMA_VERSION,
            "features": FEATURES,
            "label_mapping": {0: "benign", 1: "attack"},
            "model": champion,
            "threshold": threshold,
            "config": asdict(config),
        }
        joblib.dump(bundle, output / "model.joblib", compress=3)
        external_report = None
        if external:
            ext, ext_profile = load_sample(
                external, config.max_rows, config.chunk_rows, config.seed, external_member
            )
            overlap = ext["__hash"].isin(frame["__hash"])
            overlap_count = int(overlap.sum())
            ext = ext.loc[~overlap].reset_index(drop=True)
            if ext["__label"].nunique() != 2:
                raise ValueError("External set needs both classes after overlap exclusion.")
            ext_scores = score(champion, ext)
            external_report = {
                "profile": ext_profile,
                "overlapping_feature_rows_excluded": overlap_count,
                "metrics": metrics(ext["__label"], ext_scores, threshold),
                "attack_slices": attack_slices(ext, ext_scores, threshold),
                "threshold_policy": "frozen from in-domain validation; no external fitting or tuning",
            }
        try:
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            dirty = bool(
                subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
            )
        except (OSError, subprocess.CalledProcessError):
            revision, dirty = None, None
        versions = {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "joblib", "psutil")
        }
        if "xgboost" in config.models:
            versions["xgboost"] = importlib.metadata.version("xgboost")
        report = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": "research baseline; not production validated",
            "git_revision": revision,
            "git_worktree_dirty": dirty,
            "python": platform.python_version(),
            "versions": versions,
            "hardware": hardware,
            "config": asdict(config),
            "input_profile": {k: profile[k] for k in ("sha256", "rows_scanned", "sample_rows")},
            "split": {
                "strategy": config.split,
                "limitation": "Random splitting is exploratory and does not establish host/time generalization."
                if config.split == "random"
                else "Group/time isolation alone does not establish production readiness.",
                "train": split_summary(train),
                "validation": split_summary(validation),
                "test": split_summary(test),
                "feature_hash_overlap": 0,
            },
            "validation_leaderboard": leaderboard,
            "champion": champion_record["model"],
            "test": test_metrics,
            "test_constraints_met": test_metrics["attack_precision"] >= config.min_precision
            and test_metrics["false_positive_rate"] <= config.max_fpr,
            "test_attack_slices": attack_slices(test, test_scores, threshold),
            "inference": {
                "batch_rows": len(batch),
                "repetitions": len(timings),
                "median_batch_ms": 1000 * float(np.median(timings)),
                "p95_batch_ms": 1000 * float(np.percentile(timings, 95)),
                "flows_per_second": float(len(batch) / np.median(timings)),
                "scope": "warm batched model inference only; not packet capture throughput",
            },
            "model_bytes": (output / "model.joblib").stat().st_size,
            "external": external_report,
        }
    report["resources"] = {
        "wall_seconds": resources.elapsed,
        "sampled_peak_process_rss_mib": resources.peak / 2**20,
        "sampling_interval_seconds": 0.05,
    }
    write_json(output / "metrics.json", report)
    write_json(output / "config.json", asdict(config))
    (output / "summary.md").write_text(render_summary(report), encoding="utf-8")
    return report


def render_summary(report):
    m = report["test"]
    text = f"# Research benchmark\n\nStatus: {report['status']}.\n\nChampion selected using validation only: **{report['champion']}**.\n\n"
    text += "| Holdout metric | Value |\n|---|---:|\n"
    for key in (
        "accuracy",
        "attack_precision",
        "attack_recall",
        "macro_f1",
        "average_precision",
        "false_positive_rate",
    ):
        text += f"| {key} | {m[key]:.6f} |\n"
    text += f"\nTest precision/FPR constraints met: **{report['test_constraints_met']}**.\n\n"
    text += report["split"]["limitation"] + "\n\n"
    if report["external"]:
        ext = report["external"]["metrics"]
        text += f"External frozen-threshold test: precision {ext['attack_precision']:.4f}, recall {ext['attack_recall']:.4f}, FPR {ext['false_positive_rate']:.4f}.\n\n"
    text += f"Model size: {report['model_bytes']:,} bytes. Sampled peak RSS: {report['resources']['sampled_peak_process_rss_mib']:.1f} MiB.\n"
    return text

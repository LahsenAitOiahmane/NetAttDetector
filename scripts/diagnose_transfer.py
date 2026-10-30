"""Development-only baseline dependence and operating-point diagnostics; never tune a holdout."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from netattdetector.audit import load_prepared
from netattdetector.inference import load_bundle, set_inference_threads
from netattdetector.io import sha256_file, write_json
from netattdetector.multisource import development_split, robust_threshold, source_metrics


def diagnose(prepared, run, baseline, output):
    if Path(output).exists():
        raise ValueError("Diagnostic report already exists.")
    cohort, audit = load_prepared(prepared)
    freeze = json.loads((Path(run) / "freeze.json").read_text())
    old, new = load_bundle(baseline), load_bundle(Path(run) / "model.joblib")
    if sha256_file(Path(run) / "model.joblib") != freeze["model_sha256"]:
        raise ValueError("Frozen model changed.")
    result = {
        "scope": "development-only diagnostic; sample may overlap original baseline fitting; not independent performance evidence or causal proof",
        "source_cache_sha256": audit["cache_sha256"],
        "baseline_sha256": sha256_file(baseline),
        "frozen_model_sha256": freeze["model_sha256"],
        "permutation_importance": {},
        "illustrative_validation_operating_points": [],
    }
    with threadpool_limits(limits=2):
        set_inference_threads(old["model"], 2)
        for source, group in cohort.groupby("__source"):
            sample = group.sample(n=min(5000, len(group)), random_state=42)
            x, y = sample[old["features"]], sample["__label"]
            scores = old["model"].predict_proba(x)[:, 1]
            importance = permutation_importance(
                old["model"],
                x,
                y,
                n_repeats=3,
                scoring="average_precision",
                random_state=42,
                n_jobs=1,
            )
            entries = [
                {"feature": feature, "mean_ap_drop": float(mean), "std_ap_drop": float(std)}
                for feature, mean, std in zip(
                    old["features"],
                    importance.importances_mean,
                    importance.importances_std,
                    strict=True,
                )
            ]
            entries.sort(key=lambda x: x["mean_ap_drop"], reverse=True)
            result["permutation_importance"][str(source)] = {
                "rows": len(sample),
                "attack_rows": int(y.sum()),
                "average_precision": float(average_precision_score(y, scores)),
                "features": entries,
            }
        _, _, validation = development_split(cohort, freeze["config"]["seed"])
        scores = new["model"].predict_proba(validation[new["features"]])[:, 1]
        for target in (0.80, 0.90, 0.95, 0.98):
            policy = {
                **freeze["config"],
                "min_precision": target,
                "precision_lower_bound": min(0.95, target - 0.03),
            }
            threshold, feasible = robust_threshold(validation, scores, policy)
            result["illustrative_validation_operating_points"].append(
                {
                    "precision_target": target,
                    "feasible": feasible,
                    "threshold": threshold,
                    "by_source": source_metrics(validation, scores, threshold),
                }
            )
    grouped = cohort.groupby("__hash")["__label"].agg(["sum", "count"])
    unavoidable = int(np.minimum(grouped["sum"], grouped["count"] - grouped["sum"]).sum())
    result["reduced_representation_ambiguity"] = {
        "cohort_rows": len(cohort),
        "ambiguous_groups": int(((grouped["sum"] > 0) & (grouped["sum"] < grouped["count"])).sum()),
        "minimum_empirical_errors_for_a_deterministic_reduced_classifier": unavoidable,
        "scope": "finite development cohort only; not a bound on future distributions",
    }
    result["cautions"] = [
        "Correlated predictors can mask permutation importance; associations are not causes.",
        "Illustrative thresholds use validation only, are not the frozen model, and are not evaluated on the new source.",
    ]
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    diagnose(args.prepared, args.run, args.baseline, args.output)

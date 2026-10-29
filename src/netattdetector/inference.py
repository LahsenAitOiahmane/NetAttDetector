"""Strict-schema batch inference on exported flow CSV files."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from netattdetector.data import FEATURES, SCHEMA_VERSION, numeric_features


def load_bundle(path):
    # joblib is pickle-based. Callers must only use trusted, locally trained artifacts.
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("format_version") not in {1, 2}:
        raise ValueError("Unsupported model artifact; retrain legacy models with netatt train.")
    selected = bundle.get("features")
    valid = (
        selected == FEATURES
        if bundle["format_version"] == 1
        else isinstance(selected, list)
        and bool(selected)
        and selected == [x for x in FEATURES if x in selected]
    )
    if bundle.get("schema") != SCHEMA_VERSION or not valid:
        raise ValueError("Model feature contract does not match this application.")
    return bundle


def set_inference_threads(model, threads, seen=None):
    """Control both pipeline estimators and fitted calibration wrappers."""
    seen = set() if seen is None else seen
    if id(model) in seen:
        return
    seen.add(id(model))
    if hasattr(model, "get_params") and "n_jobs" in model.get_params(deep=False):
        model.set_params(n_jobs=threads)
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            set_inference_threads(step, threads, seen)
    if hasattr(model, "estimator"):
        set_inference_threads(model.estimator, threads, seen)
    for fitted in getattr(model, "calibrated_classifiers_", []):
        set_inference_threads(fitted.estimator, threads, seen)


def predict_csv(model_path, input_path, output_path, chunk_rows=10_000, threads=2):
    if chunk_rows < 1 or not 1 <= threads <= 8:
        raise ValueError("Use positive chunk_rows and 1..8 threads.")
    output = Path(output_path)
    if output.resolve() in {Path(input_path).resolve(), Path(model_path).resolve()}:
        raise ValueError("Prediction output cannot overwrite the input or model.")
    if output.exists():
        raise ValueError("Prediction output already exists; choose a fresh path.")
    bundle = load_bundle(model_path)
    set_inference_threads(bundle["model"], threads)
    selected = bundle.get("features", FEATURES)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")
    rows = 0
    try:
        with threadpool_limits(limits=threads):
            for chunk in pd.read_csv(input_path, chunksize=chunk_rows):
                chunk.columns = chunk.columns.str.strip()
                if chunk.columns.duplicated().any():
                    raise ValueError("Duplicate input column names.")
                features = numeric_features(chunk, selected)
                if features.isna().all(axis=1).any():
                    raise ValueError("A flow row contains no usable numeric features.")
                scores = bundle["model"].predict_proba(features)[:, 1]
                result = pd.DataFrame(
                    {
                        "row_number": np.arange(rows, rows + len(chunk)),
                        "attack_score": scores,
                        "prediction": np.where(scores >= bundle["threshold"], "attack", "benign"),
                        "threshold": bundle["threshold"],
                    }
                )
                result.to_csv(partial, mode="a" if rows else "w", header=rows == 0, index=False)
                rows += len(chunk)
        if rows == 0:
            raise ValueError("Input has no flow records.")
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
    return {"rows": rows, "output": str(output), "schema": SCHEMA_VERSION}

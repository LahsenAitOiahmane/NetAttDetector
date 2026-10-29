"""Export fitted XGBoost trees via stable UBJSON, never a platform memory snapshot."""

import copy
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from netattdetector.inference import load_bundle
from netattdetector.io import sha256_file, write_json


class PortableFrozenXGB(ClassifierMixin, BaseEstimator):
    """Inference-only adapter; its pickle stores UBJSON bytes, not a cached Booster."""

    def __init__(self, model_bytes, n_jobs=2):
        self.model_bytes = model_bytes
        self.n_jobs = n_jobs

    def _load(self):
        if not hasattr(self, "_runtime"):
            from xgboost import XGBClassifier

            self._runtime = XGBClassifier(n_jobs=self.n_jobs)
            self._runtime.load_model(bytearray(self.model_bytes))
        self._runtime.set_params(n_jobs=self.n_jobs)
        return self._runtime

    def __getstate__(self):
        return {"model_bytes": self.model_bytes, "n_jobs": self.n_jobs}

    def __setstate__(self, state):
        self.model_bytes = state["model_bytes"]
        self.n_jobs = state["n_jobs"]

    def __sklearn_is_fitted__(self):
        return True

    @property
    def classes_(self):
        return self._load().classes_

    @property
    def n_features_in_(self):
        return self._load().n_features_in_

    def fit(self, X, y=None):
        raise ValueError("Portable exports are frozen inference artifacts; train a new experiment.")

    def predict_proba(self, X):
        return self._load().predict_proba(X)

    def predict(self, X):
        return self._load().predict(X)


def export_portable(model_path, output):
    """Run this in the original pickle's environment; no estimator is fitted."""
    from xgboost import XGBClassifier

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Portable export output must be fresh.")
    original = load_bundle(model_path)
    output.mkdir(parents=True, exist_ok=True)
    memo, boosters = {}, []

    def convert(model):
        if id(model) in memo:
            return memo[id(model)]
        if isinstance(model, XGBClassifier):
            path = output / f"booster-{len(boosters)}.ubj"
            model.save_model(path)
            raw = path.read_bytes()
            clone = PortableFrozenXGB(raw, n_jobs=model.get_params().get("n_jobs") or 2)
            boosters.append({"path": path.name, "sha256": hashlib.sha256(raw).hexdigest()})
        else:
            clone = copy.copy(model)
        memo[id(model)] = clone
        if isinstance(model, Pipeline):
            clone.steps = [(name, convert(step)) for name, step in model.steps]
        elif not isinstance(model, XGBClassifier):
            if hasattr(model, "estimator"):
                clone.estimator = convert(model.estimator)
            if hasattr(model, "calibrated_classifiers_"):
                clone.calibrated_classifiers_ = []
                for fitted in model.calibrated_classifiers_:
                    copied = copy.copy(fitted)
                    copied.estimator = convert(fitted.estimator)
                    clone.calibrated_classifiers_.append(copied)
        return clone

    bundle = {
        **original,
        "model": convert(original["model"]),
        "export_origin_sha256": sha256_file(model_path),
    }
    rng = np.random.default_rng(42)
    values = rng.lognormal(5, 5, (5000, len(bundle["features"]))).astype(np.float32)
    values[rng.random(values.shape) < 0.2] = 0
    values[rng.random(values.shape) < 0.01] = np.nan
    probes = pd.DataFrame(values, columns=bundle["features"])
    expected = original["model"].predict_proba(probes)[:, 1]
    joblib.dump(bundle, output / "model.joblib", compress=3)
    reloaded = load_bundle(output / "model.joblib")
    actual = reloaded["model"].predict_proba(probes)[:, 1]
    np.testing.assert_allclose(expected, actual, rtol=0, atol=1e-7)
    if not np.array_equal(expected >= bundle["threshold"], actual >= bundle["threshold"]):
        raise ValueError("Portable export changed a probe threshold decision.")
    np.savez_compressed(output / "prediction-probes.npz", values=values, expected=expected)
    manifest = {
        "original_model_sha256": sha256_file(model_path),
        "portable_model_sha256": sha256_file(output / "model.joblib"),
        "boosters": boosters,
        "representation": "stable XGBoost UBJSON plus fitted preprocessing"
        if boosters
        else "existing fitted sklearn representation; no XGBoost component",
        "probe_rows": len(probes),
        "max_absolute_score_difference": float(np.max(np.abs(expected - actual))),
        "threshold_decisions_identical": True,
        "scope": "serialization-only conversion and synthetic parity probes; no fitting or threshold change",
        "limitation": "Preprocessing/calibration still use trusted pickle with pinned Python/ML dependencies; only XGBoost's Booster uses the stable model representation.",
    }
    write_json(output / "conversion.json", manifest)
    return manifest


def verify_portable(path):
    root = Path(path)
    bundle = load_bundle(root / "model.joblib")
    with np.load(root / "prediction-probes.npz", allow_pickle=False) as data:
        scores = bundle["model"].predict_proba(
            pd.DataFrame(data["values"], columns=bundle["features"])
        )[:, 1]
        difference = float(np.max(np.abs(scores - data["expected"])))
        np.testing.assert_allclose(scores, data["expected"], rtol=0, atol=1e-7)
        if not np.array_equal(
            scores >= bundle["threshold"], data["expected"] >= bundle["threshold"]
        ):
            raise ValueError("Cross-environment threshold decisions changed.")
    return {
        "probe_rows": len(scores),
        "max_absolute_score_difference": difference,
        "threshold_decisions_identical": True,
    }

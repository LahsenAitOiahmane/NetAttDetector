import joblib
import numpy as np
import pytest

from netattdetector.data import FEATURES, SCHEMA_VERSION
from netattdetector.inference import load_bundle, set_inference_threads
from netattdetector.portable import export_portable, verify_portable


def test_xgboost_portable_roundtrip(tmp_path, flows):
    xgb = pytest.importorskip("xgboost")
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(SimpleImputer(), xgb.XGBClassifier(n_estimators=5, n_jobs=2))
    model.fit(flows[FEATURES], flows["Label"])
    bundle = {
        "format_version": 2,
        "schema": SCHEMA_VERSION,
        "features": FEATURES,
        "model": model,
        "threshold": 0.5,
    }
    original = tmp_path / "original.joblib"
    joblib.dump(bundle, original)
    manifest = export_portable(original, tmp_path / "portable")
    assert manifest["max_absolute_score_difference"] == 0
    assert verify_portable(tmp_path / "portable")["threshold_decisions_identical"]
    loaded = load_bundle(tmp_path / "portable/model.joblib")
    set_inference_threads(loaded["model"], 1)
    np.testing.assert_array_equal(
        loaded["model"].predict_proba(flows[FEATURES]), model.predict_proba(flows[FEATURES])
    )
    with pytest.raises(ValueError, match="frozen"):
        loaded["model"].steps[-1][1].fit(flows[FEATURES], flows["Label"])


def test_export_also_handles_a_non_xgboost_winner(tmp_path, flows):
    pytest.importorskip("xgboost")
    from netattdetector.training import TrainConfig, make_model

    model = make_model("hist", TrainConfig(boosting_iterations=5))
    model.fit(flows[FEATURES], flows["Label"])
    source = tmp_path / "model.joblib"
    joblib.dump(
        {
            "format_version": 2,
            "schema": SCHEMA_VERSION,
            "features": FEATURES,
            "model": model,
            "threshold": 0.5,
        },
        source,
    )
    result = export_portable(source, tmp_path / "portable")
    assert result["boosters"] == []
    assert verify_portable(tmp_path / "portable")["threshold_decisions_identical"]

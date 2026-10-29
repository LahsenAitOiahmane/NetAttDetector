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

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from netattdetector.data import FEATURES
from netattdetector.inference import load_bundle, predict_csv
from netattdetector.training import TrainConfig, make_model, run_experiment


def test_preprocessing_fits_training_only(flows):
    model = make_model("sgd", TrainConfig())
    train = flows.iloc[:400][FEATURES].copy()
    train.loc[0, "IN_BYTES"] = np.nan
    model.fit(train, flows.iloc[:400]["Label"])
    median = np.nanmedian(train["IN_BYTES"])
    assert model.named_steps["imputer"].statistics_[FEATURES.index("IN_BYTES")] == pytest.approx(
        median
    )
    before = model.named_steps["imputer"].statistics_.copy()
    model.predict_proba(flows.iloc[400:][FEATURES] * 1000)
    np.testing.assert_array_equal(before, model.named_steps["imputer"].statistics_)


def test_end_to_end_train_save_reload_predict(tmp_path, flows):
    data = tmp_path / "flows.csv"
    flows.to_csv(data, index=False)
    output = tmp_path / "run"
    report = run_experiment(
        data, output, TrainConfig(models=("hist", "extra_trees"), trees=10, boosting_iterations=10)
    )
    assert report["split"]["feature_hash_overlap"] == 0
    assert report["test"]["rows"] == 120
    assert report["resources"]["sampled_peak_process_rss_mib"] > 0
    json.loads((output / "metrics.json").read_text())
    bundle = load_bundle(output / "model.joblib")
    original = bundle["model"].predict_proba(flows[FEATURES])[:, 1]
    # Input order and extra label/metadata columns do not change model inputs.
    flows[flows.columns[::-1]].to_csv(data, index=False)
    predictions = tmp_path / "predictions.csv"
    assert predict_csv(output / "model.joblib", data, predictions, chunk_rows=73)["rows"] == 600
    reloaded = pd.read_csv(predictions)
    np.testing.assert_allclose(original, reloaded["attack_score"])
    with pytest.raises(ValueError, match="empty"):
        run_experiment(data, output, TrainConfig())


def test_legacy_artifact_is_rejected(tmp_path):
    path = tmp_path / "legacy.joblib"
    joblib.dump({"old": True}, path)
    with pytest.raises(ValueError, match="Unsupported model artifact"):
        load_bundle(path)


def test_same_dataset_cannot_be_external_holdout(tmp_path, flows):
    path = tmp_path / "flows.csv"
    flows.to_csv(path, index=False)
    with pytest.raises(ValueError, match="External set needs both classes"):
        run_experiment(
            path, tmp_path / "run", TrainConfig(models=("extra_trees",), trees=5), external=path
        )


@pytest.mark.parametrize(
    "settings",
    [
        {"threads": 0},
        {"min_precision": 2},
        {"split": "group"},
        {"models": ["unknown"]},
        {"max_rows": 10},
    ],
)
def test_invalid_config_rejected(settings):
    with pytest.raises(ValueError):
        TrainConfig(**settings).validate()

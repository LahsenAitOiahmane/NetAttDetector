import zipfile

import numpy as np
import pandas as pd
import pytest

from netattdetector.data import FEATURES, load_sample, numeric_features
from netattdetector.training import TrainConfig, split_data


def test_sample_scans_entire_file_and_is_deterministic(tmp_path, flows):
    path = tmp_path / "flows.csv"
    flows.to_csv(path, index=False)
    first, profile = load_sample(path, max_rows=100, chunk_rows=37)
    second, _ = load_sample(path, max_rows=100, chunk_rows=83)
    pd.testing.assert_frame_equal(first, second)
    assert profile["rows_scanned"] == 600
    assert profile["full_label_counts"] == {"0": 400, "1": 200}
    assert first["IN_BYTES"].max() > 100
    # Sampling is not a prefix: most selected rows occur beyond the first 100.
    assert len(set(first["IN_BYTES"]) & set(flows.iloc[100:]["IN_BYTES"].astype(np.float32))) > 50


def test_duplicates_and_conflicting_labels_removed(tmp_path, flows):
    conflict = flows.iloc[[0]].copy()
    conflict["Label"] = 1 - conflict["Label"]
    frame = pd.concat([flows, flows.iloc[[1]], conflict], ignore_index=True)
    path = tmp_path / "flows.csv"
    frame.to_csv(path, index=False)
    sample, profile = load_sample(path, max_rows=1000)
    assert sample["__hash"].is_unique
    assert profile["conflicting_feature_rows_removed"] == 2
    assert profile["duplicate_feature_rows_removed"] == 1
    assert len(sample) == 599


def test_archive_streaming_and_member_selection(tmp_path, flows):
    path = tmp_path / "flows.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("nested/flows.csv", flows.to_csv(index=False))
    frame, profile = load_sample(path, max_rows=200)
    assert len(frame) == 200
    assert profile["rows_scanned"] == len(flows)
    assert not (tmp_path / "nested").exists()


def test_missing_schema_fails(flows):
    with pytest.raises(ValueError, match="schema mismatch"):
        numeric_features(flows.drop(columns=[FEATURES[0]]))


def test_nonfinite_features_become_missing(flows):
    flows.loc[0, "IN_BYTES"] = np.inf
    flows.loc[1, "IN_BYTES"] = -np.inf
    features = numeric_features(flows)
    assert features["IN_BYTES"].iloc[:2].isna().all()
    assert all(dtype == np.float32 for dtype in features.dtypes)
    assert "Attack" not in features and "Label" not in features


def test_invalid_and_single_class_targets(tmp_path, flows):
    flows.loc[0, "Label"] = 7
    path = tmp_path / "flows.csv"
    flows.to_csv(path, index=False)
    _, profile = load_sample(path)
    assert profile["invalid_label_rows"] == 1
    flows["Label"] = 1
    flows.to_csv(path, index=False)
    with pytest.raises(ValueError, match="both benign and attack"):
        load_sample(path)


@pytest.mark.parametrize(
    "strategy,column", [("random", None), ("group", "IPV4_SRC_ADDR"), ("temporal", "Timestamp")]
)
def test_splits_have_no_feature_or_group_overlap(tmp_path, flows, strategy, column):
    # Each host sees both labels.
    flows["IPV4_SRC_ADDR"] = [f"host-{i // 20}" for i in range(len(flows))]
    path = tmp_path / "flows.csv"
    flows.to_csv(path, index=False)
    frame, _ = load_sample(path, group_column=column)
    train, validation, test = split_data(frame, TrainConfig(split=strategy, group_column=column))
    assert len(train) + len(validation) + len(test) == len(frame)
    assert not set(train["__hash"]) & set(test["__hash"])
    if column:
        assert not set(train["__group"]) & set(validation["__group"])
    if strategy == "temporal":
        assert train["__group"].max() < validation["__group"].min()
        assert validation["__group"].max() < test["__group"].min()

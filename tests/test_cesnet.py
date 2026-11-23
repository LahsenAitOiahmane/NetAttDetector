import copy
import hashlib
import io
import sqlite3
import tarfile

import joblib
import numpy as np
import pytest

from netattdetector import cesnet
from netattdetector.context_study import context_model, synthetic_probes
from netattdetector.io import sha256_file, write_json
from netattdetector.statistical import FLOAT_FIELDS, STAT_SETTINGS


def make_archive(path, entries):
    with tarfile.open(path, "w:gz") as archive:
        for name, kind, data in entries:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = len(data) if kind == tarfile.REGTYPE else 0
            if kind == tarfile.SYMTYPE:
                info.linkname = "outside"
            archive.addfile(info, io.BytesIO(data) if info.isfile() else None)


def small_policy():
    policy = cesnet.config()
    policy["captures"] = policy["captures"][:2]
    return policy


def test_registered_policy_rejects_threshold_endpoint_or_budget_changes(tmp_path):
    policy = cesnet.config()
    for mutate in (
        lambda p: p.update(threshold=0.5),
        lambda p: p.update(maximum_archive_bytes=2**30),
        lambda p: p["captures"][0].update(cc_ip="192.0.2.1"),
    ):
        changed = copy.deepcopy(policy)
        mutate(changed)
        path = tmp_path / "config.json"
        write_json(path, changed)
        with pytest.raises(ValueError, match="registered"):
            cesnet.config(path)


def test_inventory_excludes_metadata_and_both_split_namings(tmp_path):
    path = tmp_path / "archive.tar.gz"
    policy = small_policy()
    entries = []
    for row in policy["captures"]:
        prefix = f"botnet/{row['capture_id']}/"
        entries.extend(
            (prefix + name, tarfile.REGTYPE, b"synthetic")
            for name in (
                "sample-filtered.pcapng",
                "._sample-filtered.pcapng",
                "sample-train.pcapng",
                "sample-test.pcapng",
                "sample_train.pcapng",
                "sample_test.pcapng",
            )
        )
    make_archive(path, entries)
    result = cesnet.inventory_archive(path, policy)
    assert len(result["selected"]) == 2
    assert all(
        row["member"].endswith("/sample-filtered.pcapng") for row in result["selected"].values()
    )
    assert result["members"] == 12


@pytest.mark.parametrize(
    "bad", ["traversal", "absolute", "symlink", "duplicate", "ambiguous", "oversize", "missing"]
)
def test_inventory_rejects_unsafe_or_incomplete_archives(tmp_path, bad):
    policy = small_policy()
    entries = [
        (f"botnet/{row['capture_id']}/sample.pcapng", tarfile.REGTYPE, b"synthetic")
        for row in policy["captures"]
    ]
    if bad == "traversal":
        entries.append(("../outside", tarfile.REGTYPE, b"bad"))
    elif bad == "absolute":
        entries.append(("/outside", tarfile.REGTYPE, b"bad"))
    elif bad == "symlink":
        entries.append(("linked", tarfile.SYMTYPE, b""))
    elif bad == "duplicate":
        entries.append(entries[0])
    elif bad == "ambiguous":
        entries.append(
            (
                f"botnet/{policy['captures'][0]['capture_id']}/other.pcapng",
                tarfile.REGTYPE,
                b"other",
            )
        )
    elif bad == "oversize":
        policy["maximum_capture_bytes"] = 1
    else:
        entries.pop()
    path = tmp_path / "bad.tar.gz"
    make_archive(path, entries)
    with pytest.raises(ValueError):
        cesnet.inventory_archive(path, policy)


def test_endpoint_label_is_direction_independent_and_other_flows_stay_unknown():
    sample = {"cc_ip": "192.0.2.10", "cc_port": 80}
    forward = {"src_ip": "198.51.100.1", "src_port": 12345, "dst_ip": "192.0.2.10", "dst_port": 80}
    reverse = {
        "src_ip": forward["dst_ip"],
        "src_port": forward["dst_port"],
        "dst_ip": forward["src_ip"],
        "dst_port": forward["src_port"],
    }
    assert cesnet.endpoint_eligible(forward, sample)
    assert cesnet.endpoint_eligible(reverse, sample)
    assert not cesnet.endpoint_eligible({**forward, "dst_port": 443}, sample)
    assert not cesnet.endpoint_eligible({**forward, "dst_ip": "192.0.2.11"}, sample)


def test_summary_preserves_capture_weighting_unknowns_and_absent_benign_metrics():
    rows = [
        {"eligible_positive_flows": 1000, "detected_positive_flows": 1000},
        {"eligible_positive_flows": 10, "detected_positive_flows": 0},
        {"eligible_positive_flows": 0, "detected_positive_flows": 0},
    ]
    result = cesnet.recall_summary(rows)
    assert result["capture_macro_recall"] == 0.5
    assert result["pooled_positive_flow_recall"] == 1000 / 1010
    assert result["unassessed_captures"] == 1
    assert all(
        result[name] is None
        for name in ("precision", "accuracy", "false_positive_rate", "balanced_accuracy")
    )
    assert result == cesnet.recall_summary(rows)
    empty = cesnet.recall_summary([rows[-1]])
    assert empty["capture_macro_recall"] is None and empty["capture_bootstrap_95"] is None


def test_verified_archive_only_extracts_exact_frozen_members_and_refuses_overwrite(tmp_path):
    policy = small_policy()
    source = tmp_path / "source"
    source.mkdir()
    archive = source / "botnet.tar.gz"
    entries = [
        (f"botnet/{row['capture_id']}/sample.pcapng", tarfile.REGTYPE, b"synthetic")
        for row in policy["captures"]
    ]
    entries.append(("botnet/readme.txt", tarfile.REGTYPE, b"not a capture"))
    make_archive(archive, entries)
    policy["archive"]["bytes"] = archive.stat().st_size
    policy["archive"]["publisher_md5"] = hashlib.md5(
        archive.read_bytes(), usedforsecurity=False
    ).hexdigest()
    acquisition = {
        "archive": cesnet.archive_identity(archive, policy),
        "inventory": cesnet.inventory_archive(archive, policy),
    }
    output = tmp_path / "captures"
    result = cesnet.extract_originals(source, output, acquisition, policy)
    assert set(result) == {row["capture_id"] for row in policy["captures"]}
    assert len(list(output.iterdir())) == 2
    with pytest.raises(FileExistsError):
        cesnet.extract_originals(source, output, acquisition, policy)
    archive.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="byte count"):
        cesnet.extract_originals(source, tmp_path / "other", acquisition, policy)


@pytest.fixture
def synthetic_measurements(tmp_path, monkeypatch):
    policy = small_policy()
    policy_path = tmp_path / "config.json"
    write_json(policy_path, policy)
    directory = tmp_path / "measurements"
    directory.mkdir()
    path = directory / "flows.sqlite"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE flows("
        + ",".join(
            "ordinal INTEGER PRIMARY KEY"
            if name == "ordinal"
            else f"{name} TEXT"
            if name == "capture_id"
            else f"{name} REAL"
            if name in FLOAT_FIELDS
            else f"{name} INTEGER"
            for name in cesnet.COLUMNS
        )
        + ")"
    )
    logical = hashlib.sha256()
    records = []
    ordinal = 0
    probes = synthetic_probes()
    for sample in policy["captures"]:
        positives = 0
        for index, row in enumerate(probes.to_dict("records")):
            ordinal += 1
            positive = int(index % 2 == 0)
            positives += positive
            values = [
                ordinal,
                sample["capture_id"],
                positive,
                *[
                    float(row[name]) if name in FLOAT_FIELDS else int(row[name])
                    for name in cesnet.FEATURES
                ],
            ]
            db.execute("INSERT INTO flows VALUES(" + ",".join("?" for _ in values) + ")", values)
            cesnet.row_digest(logical, values)
        records.append(
            {
                "capture_id": sample["capture_id"],
                "family": sample["family"],
                "exported_flows": len(probes),
                "eligible_positive_flows": positives,
            }
        )
    db.commit()
    db.close()
    manifest = {
        "config_sha256": sha256_file(policy_path),
        "schema": cesnet.SCHEMA,
        "features": cesnet.FEATURES,
        "exporter_version": "6.6.0",
        "settings": {**STAT_SETTINGS, "max_nflows": 250000},
        "database": {"bytes": path.stat().st_size, "sha256": sha256_file(path)},
        "rows": ordinal,
        "logical_sha256": logical.hexdigest(),
        "records": records,
    }
    write_json(directory / "manifest.json", manifest)
    monkeypatch.setattr(cesnet, "config", lambda *_: policy)
    return directory, policy, policy_path, probes


def test_database_verification_rejects_tampering_and_reconciles_capture_counts(
    synthetic_measurements,
):
    directory, policy, policy_path, _ = synthetic_measurements
    assert cesnet.verify_database(directory, policy, policy_path)["rows"] == 90
    db = sqlite3.connect(directory / "flows.sqlite")
    db.execute("UPDATE flows SET eligible=0 WHERE ordinal=1")
    db.commit()
    db.close()
    with pytest.raises(ValueError, match="bytes changed"):
        cesnet.verify_database(directory, policy, policy_path)


def test_evaluation_uses_frozen_hash_before_loading_and_never_calls_fit(
    synthetic_measurements, tmp_path, monkeypatch
):
    directory, policy, policy_path, probes = synthetic_measurements
    model = context_model("dummy", "statistics27_control")
    model.fit(probes, np.arange(len(probes)) % 2)
    specification = {
        "id": "synthetic",
        "model": "dummy",
        "features": "statistics27_control",
        "weights": "flow_natural",
    }
    path = tmp_path / "model.joblib"
    bundle = {
        "format": "context-diagnostic-v1",
        "features": cesnet.FEATURES,
        "schema": cesnet.SCHEMA,
        "threshold": cesnet.THRESHOLD,
        "specification": specification,
        "model": model,
        "deployment_authorized": False,
    }
    joblib.dump(bundle, path)
    monkeypatch.setattr(cesnet, "MODEL_SHA256", sha256_file(path))
    freeze_path = tmp_path / "freeze.json"
    write_json(
        freeze_path,
        {
            "artifact_sha256": cesnet.MODEL_SHA256,
            "threshold": cesnet.THRESHOLD,
            "specification": specification,
            "passes_development_research_screen": False,
        },
    )
    policy["freeze"] = str(freeze_path)
    from sklearn.pipeline import Pipeline

    monkeypatch.setattr(
        Pipeline, "fit", lambda *_args, **_kwargs: pytest.fail("External test must not fit")
    )
    result = cesnet.evaluate(directory, path, tmp_path / "evaluation", policy_path)
    assert result["model_fits"] == 0 and not result["threshold_changed"]
    assert result["summary"]["eligible_positive_flows"] == 46
    assert result["summary"]["false_positive_rate"] is None
    assert len(result["records"]) == 2
    assert not result["deployment_authorized"]
    path.write_bytes(b"malicious altered pickle")
    monkeypatch.setattr(joblib, "load", lambda *_: pytest.fail("Must check hash before load"))
    with pytest.raises(ValueError, match="freeze"):
        cesnet.evaluate(directory, path, tmp_path / "other-evaluation", policy_path)

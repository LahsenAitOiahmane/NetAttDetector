import json
import sqlite3
from copy import deepcopy

import pytest

from netattdetector.io import sha256_file, write_json
from netattdetector.measurement import CORE_FEATURES
from netattdetector.native_cohort import (
    COHORT_SCHEMA,
    COLUMNS,
    SPLITS,
    CohortWriter,
    audit_database,
    load_config,
    verify_native_cohort,
)


def fixture_flow(time, port=53000):
    row = json.loads(open("tests/fixtures/pcap/expected.json", encoding="utf-8").read())["rows"][0]
    row["bidirectional_first_seen_ms"] = time
    row["bidirectional_last_seen_ms"] = time + row["bidirectional_duration_ms"]
    row["src_port"] = port
    return row


def miniature(path, *, contradictory=False, overlap_time=False):
    plan = {
        "partitions": dict(zip(SPLITS, ("a", "b", "c"), strict=True)),
        "purge_seconds": 300,
        "minimum_distinct_connections_per_class": 1,
    }
    writer = CohortWriter(path)
    try:
        for index, split in enumerate(SPLITS):
            for label in (0, 1):
                # Same feature pattern has different labels: retain and disclose it.
                row = {
                    "uid": f"{split}-{label}",
                    "label": label,
                    "detail": "Attack" if label else "Benign",
                }
                writer.add(
                    split,
                    plan["partitions"][split],
                    fixture_flow((index + 1) * (100 if overlap_time else 1_000_000) + label * 1000),
                    "matched",
                    row,
                )
            writer.add(
                split,
                plan["partitions"][split],
                fixture_flow((index + 1) * 1_000_000 + 2000),
                "ambiguous",
                None,
            )
            # Multiple timeout segments of one connection share a group, but not a flow identity.
            row = {"uid": f"{split}-1", "label": 0 if contradictory else 1, "detail": "Attack"}
            writer.add(
                split,
                plan["partitions"][split],
                fixture_flow((index + 1) * 1_000_000 + 3000),
                "matched",
                row,
            )
        writer.finish()
    finally:
        writer.close()
    return plan


def test_realistic_grouping_exclusions_and_repeated_patterns_roundtrip(tmp_path):
    path = tmp_path / "cohort.sqlite"
    audit = audit_database(path, miniature(path))
    assert audit["preparation_checks_passed"]
    assert audit["rows"] == 12
    assert audit["matched_feature_patterns"] == {
        "unique_vectors": 1,
        "rows": 9,
        "repeated_vectors": 1,
        "conflicting_label_vectors": 1,
        "rows_in_conflicting_vectors": 9,
    }
    assert all(
        pair["shared_feature_vectors"] == 1 and pair["shared_vectors_with_conflicting_labels"] == 1
        for pair in audit["exact_feature_overlap"]
    )
    assert all(p["matched_class_connections"] == {"0": 1, "1": 1} for p in audit["partitions"])
    assert all(p["status_counts"] == {"ambiguous": 1, "matched": 3} for p in audit["partitions"])
    with sqlite3.connect(path) as db:
        assert [row[1] for row in db.execute("PRAGMA table_info(flows)")] == COLUMNS
        assert (
            db.execute(
                "SELECT COUNT(*) FROM flows WHERE label IS NULL AND group_hash IS NULL"
            ).fetchone()[0]
            == 3
        )
    assert "src_ip" not in COLUMNS and "uid" not in COLUMNS


@pytest.mark.parametrize(
    "changes",
    [
        {"src2dst_bytes": -1},
        {"src_port": 65536},
        {"ip_version": 5},
        {"src2dst_packets": 1.0},
        {"src2dst_packets": True},
        {"bidirectional_duration_ms": 999},
        {"bidirectional_bytes": 10},
    ],
)
def test_invalid_export_never_becomes_a_training_row(tmp_path, changes):
    writer = CohortWriter(tmp_path / "bad.sqlite")
    try:
        with pytest.raises(ValueError):
            writer.add(
                "train",
                "a",
                {**fixture_flow(1000), **changes},
                "matched",
                {"uid": "x", "label": 1, "detail": "Attack"},
            )
    finally:
        writer.close()


def test_duplicate_flow_rejected_even_with_different_capture(tmp_path):
    writer = CohortWriter(tmp_path / "duplicate.sqlite")
    try:
        writer.add("train", "a", fixture_flow(1000), "unmatched", None)
        with pytest.raises(sqlite3.IntegrityError):
            writer.add("validation", "c", fixture_flow(1000), "unmatched", None)
        with pytest.raises(FileExistsError):
            CohortWriter(writer.path)
    finally:
        writer.close()


def test_unknown_labels_cannot_be_promoted(tmp_path):
    writer = CohortWriter(tmp_path / "unknown.sqlite")
    try:
        for status, row in [
            ("matched", {"uid": "x", "label": None, "detail": "Attack"}),
            ("unmatched", {"uid": "x", "label": 0, "detail": "Benign"}),
            ("made_up", None),
        ]:
            with pytest.raises(ValueError):
                writer.add("train", "a", fixture_flow(1000), status, row)
    finally:
        writer.close()


def test_bad_chronology_and_inconsistent_connection_block_acceptance(tmp_path):
    path = tmp_path / "bad.sqlite"
    audit = audit_database(path, miniature(path, contradictory=True, overlap_time=True))
    assert not audit["preparation_checks_passed"]
    assert audit["invalid_or_cross_partition_groups"] == 3
    assert not audit["checks"]["capture_time_order_and_purge_valid"]


def test_feature_and_group_tampering_detected_on_reopen(tmp_path):
    path = tmp_path / "bad.sqlite"
    plan = miniature(path)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE flows SET src_port=53 WHERE ordinal=1")
    with pytest.raises(ValueError, match="feature hash"):
        audit_database(path, plan)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE flows SET src_port=53000,group_hash='fake' WHERE ordinal=1")
    with pytest.raises(ValueError, match="connection group"):
        audit_database(path, plan)


def test_support_and_partition_membership_recomputed_from_rows(tmp_path):
    path = tmp_path / "bad.sqlite"
    plan = miniature(path)
    plan["minimum_distinct_connections_per_class"] = 2
    assert not audit_database(path, plan)["checks"]["distinct_connection_class_support_met"]
    with sqlite3.connect(path) as db:
        db.execute("UPDATE flows SET split='validation' WHERE ordinal=1")
    with pytest.raises(ValueError, match="membership"):
        audit_database(path, plan)


def test_registered_scope_is_training_disabled(tmp_path):
    config, policy, plan = load_config("configs/data/native-cohort.json")
    assert (
        not config["training_enabled"]
        and not policy["training_enabled"]
        and not plan["training_enabled"]
    )
    modified = deepcopy(config)
    modified["training_enabled"] = True
    write_json(tmp_path / "enabled.json", modified)
    with pytest.raises(ValueError, match="training-disabled"):
        load_config(tmp_path / "enabled.json")


def test_manifest_verifier_refuses_changed_database_and_incomplete_run(tmp_path):
    path = tmp_path / "cohort.sqlite"
    miniature(path)
    with pytest.raises(FileNotFoundError):
        verify_native_cohort(tmp_path, tmp_path / "verification.json")
    # A saved byte hash prevents even an otherwise valid SQL mutation from passing.
    manifest = {
        "schema": COHORT_SCHEMA,
        "status": "complete_experimental_candidate",
        "training_authorized": False,
        "candidate_policy_promoted": False,
        "features": CORE_FEATURES,
        "feature_contract": "nfstream-core-v1",
        "label_policy": "iot23-packet-point-candidate-v1",
        "database": {
            "name": "cohort.sqlite",
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
        "config_hashes": {
            "configs/data/native-cohort.json": sha256_file("configs/data/native-cohort.json")
        },
    }
    write_json(tmp_path / "manifest.json", manifest)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE flows SET detail='different' WHERE ordinal=1")
    with pytest.raises(ValueError, match="database bytes"):
        verify_native_cohort(tmp_path, tmp_path / "verification.json")


def test_zero_rows_never_complete(tmp_path):
    path = tmp_path / "empty.sqlite"
    writer = CohortWriter(path)
    writer.finish()
    writer.close()
    with pytest.raises(ValueError, match="Empty"):
        audit_database(path, {"partitions": {}})

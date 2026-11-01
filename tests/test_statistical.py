import json
import math
import platform
import sqlite3
from pathlib import Path

import pytest

from netattdetector.measurement import flow_key
from netattdetector.native_cohort import CohortWriter, digest
from netattdetector.statistical import (
    EXTRA_FEATURES,
    FEATURES,
    SCHEMA,
    STAT_SETTINGS,
    export_statistical_pcap,
    validate_statistics,
    verify_statistical_measurements,
)
from netattdetector.statistical_cohort import COLUMNS, create_sidecar, joined_audit, pattern_hash

EXPECTED = "tests/fixtures/pcap/statistical-expected.json"


def export_document():
    known = json.loads(Path(EXPECTED).read_text())
    return {
        "schema": SCHEMA,
        "features": FEATURES,
        "settings": STAT_SETTINGS,
        "exporter_version": "6.6.0",
        "pcap_sha256": known["pcap_sha256"],
        "rows": known["rows"],
    }


def test_independent_packet_lists_and_full_verifier(tmp_path):
    data = export_document()
    tcp = data["rows"][2]
    assert tcp["bidirectional_mean_ps"] == 42
    assert tcp["bidirectional_stddev_ps"] == math.sqrt(10)
    assert tcp["bidirectional_stddev_piat_ms"] == math.sqrt(950)
    assert tcp["bidirectional_syn_packets"] == 2 and tcp["bidirectional_ack_packets"] == 5
    path = tmp_path / "export.json"
    path.write_text(json.dumps(data))
    result = verify_statistical_measurements(path, EXPECTED, tmp_path / "verified.json")
    assert result["passed"] and result["checked_fields"] == 165
    assert len(FEATURES) == 27 and len(EXTRA_FEATURES) == 16


@pytest.mark.parametrize(
    "field,value",
    [
        ("bidirectional_mean_ps", 35.5),
        ("bidirectional_syn_packets", 1),
        ("bidirectional_stddev_ps", float("nan")),
        ("bidirectional_min_piat_ms", -1),
        ("bidirectional_max_ps", 36.0),
        ("src2dst_bytes", 32.5),
    ],
)
def test_wrong_values_types_and_float_semantics_fail(tmp_path, field, value):
    data = export_document()
    data["rows"][0][field] = value
    path = tmp_path / "export.json"
    path.write_text(json.dumps(data))
    assert not verify_statistical_measurements(path, EXPECTED, tmp_path / "failed.json")["passed"]


def test_small_float_roundoff_is_tolerated_but_wrong_settings_are_not(tmp_path):
    data = export_document()
    data["rows"][2]["bidirectional_stddev_ps"] += 1e-12
    path = tmp_path / "export.json"
    path.write_text(json.dumps(data))
    assert verify_statistical_measurements(path, EXPECTED, tmp_path / "roundoff.json")["passed"]
    data["settings"] = {**STAT_SETTINGS, "statistical_analysis": False}
    path.write_text(json.dumps(data))
    assert not verify_statistical_measurements(path, EXPECTED, tmp_path / "settings.json")["passed"]


def test_single_packet_time_and_flag_limits():
    row = export_document()["rows"][1]
    row["bidirectional_max_piat_ms"] = 1
    with pytest.raises(ValueError, match="Single-packet"):
        validate_statistics(row)
    row = export_document()["rows"][2]
    row["bidirectional_ack_packets"] = 7
    with pytest.raises(ValueError, match="flag counts"):
        validate_statistics(row)


def synthetic_sidecar(tmp_path):
    core_path, path = tmp_path / "core.sqlite", tmp_path / "statistics.sqlite"
    core = CohortWriter(core_path)
    db = create_sidecar(path)
    try:
        for number, row in enumerate(export_document()["rows"], 1):
            core.add(
                "train",
                "fixture",
                row,
                "matched",
                {
                    "uid": f"u{number}",
                    "label": number % 2,
                    "detail": "Attack" if number % 2 else "-",
                },
            )
            values = [
                number,
                digest(flow_key(row)),
                *[row[name] for name in EXTRA_FEATURES],
                pattern_hash([row[name] for name in FEATURES]),
                pattern_hash([row[name] for name in FEATURES[2:]]),
            ]
            db.execute(
                "INSERT INTO features VALUES (" + ",".join("?" for _ in COLUMNS) + ")", values
            )
        core.finish()
        db.commit()
    finally:
        db.close()
        core.close()
    return path, core_path


def test_sidecar_reopens_and_binds_stats_to_exact_core_rows(tmp_path):
    path, core = synthetic_sidecar(tmp_path)
    audit = joined_audit(path, core)
    assert audit["rows"] == 5 and audit["complete_identity_core_join"]
    assert audit["details"]["Attack"]["flags"]["bidirectional_syn_packets"]["flows_with_flag"] == 1
    with sqlite3.connect(path) as db:
        db.execute("UPDATE features SET bidirectional_mean_ps=43 WHERE ordinal=3")
    with pytest.raises(ValueError, match="pattern hashes"):
        joined_audit(path, core)


def test_sidecar_cannot_omit_or_relink_a_flow(tmp_path):
    path, core = synthetic_sidecar(tmp_path)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE features SET flow_hash='different' WHERE ordinal=1")
    with pytest.raises(ValueError, match="identities"):
        joined_audit(path, core)
    with pytest.raises(FileExistsError):
        create_sidecar(path)


def test_real_linux_statistical_export_matches_independent_expectations(tmp_path):
    if platform.system() != "Linux":
        pytest.skip("Pinned exporter is Linux-only")
    pytest.importorskip("nfstream")
    export = tmp_path / "actual.json"
    export_statistical_pcap("tests/fixtures/pcap/core.pcap", export)
    result = verify_statistical_measurements(export, EXPECTED, tmp_path / "checked.json")
    assert result["passed"], result["errors"]
    assert result["checked_fields"] == 165

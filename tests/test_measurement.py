import json
from pathlib import Path

import pandas as pd
import pytest

from netattdetector.data import numeric_features
from netattdetector.measurement import (
    CORE_FEATURES,
    SCHEMA,
    SETTINGS,
    contract_report,
    export_pcap,
    verify_measurements,
)


def saved_export(tmp_path):
    expected = json.loads(Path("tests/fixtures/pcap/expected.json").read_text())
    report = {
        "schema": SCHEMA,
        "features": CORE_FEATURES,
        "settings": dict(SETTINGS),
        "exporter_version": "6.6.0",
        "pcap_sha256": expected["pcap_sha256"],
        "rows": expected["rows"],
    }
    path = tmp_path / "export.json"
    path.write_text(json.dumps(report))
    return path, report


def test_complete_expected_measurement_fields(tmp_path):
    path, report = saved_export(tmp_path)
    result = verify_measurements(path, "tests/fixtures/pcap/expected.json", tmp_path / "check.json")
    assert result["passed"] and result["flows"] == 5
    assert result["checked_fields"] == 85
    # A fixture passing its native measurement contract cannot enter the old UQ model.
    with pytest.raises(ValueError, match="Feature schema mismatch"):
        numeric_features(pd.DataFrame(report["rows"]))
    assert not contract_report()["uq_model_compatible"]


@pytest.mark.parametrize("case", ["bytes", "units", "pcap", "missing_flow", "duplicate"])
def test_measurement_comparison_rejects_wrong_semantics(tmp_path, case):
    path, report = saved_export(tmp_path)
    if case == "bytes":
        report["rows"][0]["src2dst_bytes"] += 14
    elif case == "units":
        report["settings"]["accounting_mode"] = 0
    elif case == "pcap":
        report["pcap_sha256"] = "0" * 64
    elif case == "missing_flow":
        report["rows"].pop()
    elif case == "duplicate":
        report["rows"].append(report["rows"][0])
    path.write_text(json.dumps(report))
    assert not verify_measurements(
        path, "tests/fixtures/pcap/expected.json", tmp_path / "check.json"
    )["passed"]


def test_offline_adapter_refuses_interface_sources_and_overwrite(tmp_path):
    with pytest.raises(ValueError, match="offline"):
        export_pcap("eth0", tmp_path / "out.json")
    path = tmp_path / "existing.json"
    path.write_text("existing evidence")
    with pytest.raises(FileExistsError):
        export_pcap("tests/fixtures/pcap/core.pcap", path)
    assert path.read_text() == "existing evidence"

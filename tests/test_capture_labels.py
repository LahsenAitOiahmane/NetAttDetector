import io
import json
from decimal import Decimal
from pathlib import Path

import pytest

from netattdetector.capture_labels import (
    acquire_capture,
    audit_capture_labels,
    bounded_fetch,
    connection_key,
    match_flow,
    read_zeek_labels,
)


def zeek_log(tmp_path, rows):
    path = tmp_path / "conn.log.labeled"
    path.write_text(
        "#separator \\x09\n"
        "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tduration\ttunnel_parents   label   detailed-label\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    return path


def flow(start=1000, end=1100):
    return {
        "src_ip": "192.0.2.1",
        "src_port": 50000,
        "dst_ip": "198.51.100.1",
        "dst_port": 443,
        "protocol": 6,
        "bidirectional_first_seen_ms": start,
        "bidirectional_last_seen_ms": end,
    }


def test_zeek_mixed_separator_and_reverse_direction(tmp_path):
    path = zeek_log(
        tmp_path,
        ["1.000293\tUID1\t198.51.100.1\t443\t192.0.2.1\t50000\ttcp\t0.100000\t-   Malicious   C&C"],
    )
    index, counts = read_zeek_labels(path)
    status, row = match_flow(flow(), index)
    assert status == "matched" and row["label"] == 1 and counts["rows"] == 1
    assert row["start"] == Decimal("1000.293")
    # A timeout subflow may link to the same connection, but is not a new independent label.
    assert match_flow(flow(1050, 1090), index)[1]["uid"] == "UID1"


@pytest.mark.parametrize(
    "case,expected",
    [
        ("unmatched", "unmatched"),
        ("partial", "partial_interval"),
        ("conflict", "ambiguous"),
        ("same_class_duplicate", "ambiguous"),
        ("unknown_duration", "unknown_label_or_duration"),
        ("unknown_label", "unknown_label_or_duration"),
        ("icmp", "unsupported_protocol"),
    ],
)
def test_fail_closed_label_matching(case, expected):
    candidate = {
        "uid": "u1",
        "start": Decimal(1000),
        "end": Decimal(1100),
        "label": 0,
        "detail": "-",
    }
    key = connection_key("192.0.2.1", 50000, "198.51.100.1", 443, 6)
    index = {key: [candidate]}
    actual = flow()
    if case == "unmatched":
        actual = flow(1200, 1201)
    elif case == "partial":
        actual = flow(1050, 1200)
    elif case in {"conflict", "same_class_duplicate"}:
        index[key].append({**candidate, "uid": "u2", "label": int(case == "conflict")})
    elif case == "unknown_duration":
        candidate["end"] = None
    elif case == "unknown_label":
        candidate["label"] = None
    elif case == "icmp":
        actual["protocol"] = 1
    assert match_flow(actual, index) == (expected, None)


def test_unknown_end_cannot_silently_expose_later_same_tuple_as_benign():
    key = connection_key("192.0.2.1", 50000, "198.51.100.1", 443, 6)
    index = {
        key: [
            {"uid": "old", "start": Decimal(0), "end": None, "label": 1},
            {"uid": "later", "start": Decimal(1000), "end": Decimal(1100), "label": 0},
        ]
    }
    assert match_flow(flow(), index) == ("ambiguous", None)


@pytest.mark.parametrize(
    "row",
    [
        "nan\tu1\t192.0.2.1\t1\t198.51.100.1\t2\ttcp\t1\t- Benign -",
        "1\tu1\t192.0.2.1\t1\t198.51.100.1\t2\ttcp\t-1\t- Benign -",
        "1\tu1\t192.0.2.1\t70000\t198.51.100.1\t2\ttcp\t1\t- Benign -",
        "1\tu1\t192.0.2.1\t1\t198.51.100.1\t2\ttcp\t1\t- Benign",
    ],
)
def test_reject_malformed_labels(tmp_path, row):
    with pytest.raises(ValueError):
        read_zeek_labels(zeek_log(tmp_path, [row]))


def test_duplicate_uid_rejected(tmp_path):
    row = "1\tu1\t192.0.2.1\t1\t198.51.100.1\t2\ttcp\t1\t- Benign -"
    with pytest.raises(ValueError, match="Duplicate"):
        read_zeek_labels(zeek_log(tmp_path, [row, row]))


def test_unpinned_download_and_unknown_capture_refused_before_network(tmp_path):
    config = json.loads(Path("configs/data/iot23-development.json").read_text())
    config["captures"]["CTU-IoT-Malware-Capture-20-1"]["sha256"] = {}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="record-first"):
        acquire_capture("CTU-IoT-Malware-Capture-20-1", tmp_path / "out", path)
    with pytest.raises(ValueError, match="Unknown"):
        acquire_capture("../../cicids2017", tmp_path / "out", path)


@pytest.mark.parametrize(
    "case", ["oversize_header", "oversize_stream", "truncated", "changed_hash"]
)
def test_download_budget_and_identity_fail_before_write(tmp_path, monkeypatch, case):
    class Response(io.BytesIO):
        status = 200
        headers = (
            {"Content-Length": "9"}
            if case == "oversize_header"
            else {"Content-Length": "4"}
            if case == "truncated"
            else {}
        )

    class Opener:
        def open(self, request, timeout):
            return Response(b"abc")

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    target = tmp_path / "file.pcap"
    with pytest.raises(ValueError):
        bounded_fetch(
            "https://mcfp.felk.cvut.cz/example",
            target,
            2 if case == "oversize_stream" else 8,
            "0" * 64 if case == "changed_hash" else None,
        )
    assert not target.exists()


def test_audit_refuses_overwriting_evidence(tmp_path):
    path = tmp_path / "audit.json"
    path.write_text("existing")
    with pytest.raises(FileExistsError):
        audit_capture_labels("missing", "missing", path)
    assert path.read_text() == "existing"

import copy
import json
import platform
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from netattdetector import kitsune_native as native
from netattdetector.causal_context import CONTEXT_FEATURES
from netattdetector.io import write_json

RANGES = {
    "0": {"minimum_ns": 0, "maximum_ns": 99_999_999},
    "1": {"minimum_ns": 200_000_000, "maximum_ns": 300_000_000},
}
FIXTURE_RANGES = {
    "0": {"minimum_ns": 1600000001000000000, "maximum_ns": 1600000003200000000},
    "1": {"minimum_ns": 1600000003320000000, "maximum_ns": 1600000010000000000},
}


def test_quantized_label_windows_are_inclusive_and_gaps_stay_unknown():
    phases = native.PacketPhases(RANGES)
    assert phases.label(0) == phases.label(99) == 0
    assert phases.label(200) == phases.label(300) == 1
    assert phases.label(100) is phases.label(199) is phases.label(301) is None
    with pytest.raises(ValueError, match="integer"):
        phases.label(0.0)
    overlap = copy.deepcopy(RANGES)
    overlap["1"]["minimum_ns"] = 99_999_999
    with pytest.raises(ValueError, match="overlap"):
        native.PacketPhases(overlap)


def test_callback_tracks_every_packet_but_context_only_tracks_flow_initialization():
    plugin = native.PhaseContextPlugin(RANGES)
    flow = SimpleNamespace(
        src_ip="192.0.2.1",
        dst_ip="198.51.100.1",
        dst_port=80,
        bidirectional_first_seen_ms=50,
        udps=SimpleNamespace(),
    )
    plugin.on_init(SimpleNamespace(time=50), flow)
    plugin.on_update(SimpleNamespace(time=250), flow)
    plugin.on_update(SimpleNamespace(time=150), flow)
    assert flow.udps.phase_mask == 7
    assert flow.udps.phase_packets == 3
    assert (
        flow.udps.callback_benign_packets
        == flow.udps.callback_attack_packets
        == flow.udps.callback_unknown_packets
        == 1
    )
    assert flow.udps.callback_min_ms == 50 and flow.udps.callback_max_ms == 250
    assert plugin.state.sequence == 1
    assert flow.udps.ctx_src_flow_starts_60s == 0
    row = vars(flow.udps) | {"bidirectional_packets": 3}
    assert native.label_row(row) == (None, "unknown_clock")
    plugin.cleanup()
    assert plugin.state is None


@pytest.mark.parametrize(
    "benign,attack,unknown,expected",
    [
        (2, 0, 0, (0, "uniform_benign")),
        (0, 3, 0, (1, "uniform_attack")),
        (2, 2, 0, (None, "mixed_phase")),
        (2, 0, 1, (None, "unknown_clock")),
    ],
)
def test_mixed_and_unknown_packets_are_not_silently_given_a_binary_label(
    benign, attack, unknown, expected
):
    count = benign + attack + unknown
    row = dict(
        zip(
            native.LABEL_FIELDS,
            [
                (1 if benign else 0) | (2 if attack else 0) | (4 if unknown else 0),
                count,
                benign,
                attack,
                unknown,
                0,
                1,
            ],
            strict=True,
        )
    )
    row["bidirectional_packets"] = count
    assert native.label_row(row) == expected
    row["bidirectional_packets"] += 1
    with pytest.raises(ValueError, match="reconcile"):
        native.label_row(row)


def fixture_rows():
    rows = json.loads(Path("tests/fixtures/pcap/statistical-expected.json").read_text())["rows"]
    for sequence, (row, counts) in enumerate(
        zip(rows, [(2, 0), (1, 0), (3, 3), (0, 2), (0, 1)], strict=True), 1
    ):
        row.update(dict.fromkeys(CONTEXT_FEATURES, 0))
        if sequence == 5:
            for field in CONTEXT_FEATURES[1:3] + CONTEXT_FEATURES[4:]:
                row[field] = 1
        row.update(
            {
                "context_init_sequence": sequence,
                "context_watermark_ms": row["bidirectional_first_seen_ms"],
                "context_warmup_complete": 0,
                "context_late_start": 0,
                "context_peak_events": sequence,
            }
        )
        benign, attack = counts
        row.update(
            {
                "phase_mask": (1 if benign else 0) | (2 if attack else 0),
                "phase_packets": benign + attack,
                "callback_benign_packets": benign,
                "callback_attack_packets": attack,
                "callback_unknown_packets": 0,
                "callback_min_ms": row["bidirectional_first_seen_ms"],
                "callback_max_ms": row["bidirectional_last_seen_ms"],
            }
        )
        row["label"], row["status"] = native.label_row(row)
    return rows


def test_synthetic_pipeline_reopens_all_rows_and_preserves_mixed_labels_and_tuple_groups(
    tmp_path, monkeypatch
):
    config = native.configuration()
    config["expected_packets"] = 12
    config["expected_class_counts"] = {"0": 6, "1": 6}
    config["label_time_ranges_ns"] = FIXTURE_RANGES
    source_audit = {
        "packets": 12,
        "binary_class_counts": config["expected_class_counts"],
        "time_ranges_by_label": FIXTURE_RANGES,
        "single_benign_prefix_then_attack_suffix": True,
    }
    rows = fixture_rows()
    monkeypatch.setattr(native, "configuration", lambda _: config)
    monkeypatch.setattr(
        native,
        "verified_files",
        lambda *_: ({}, {"files": []}, {"pcap": Path("tests/fixtures/pcap/core.pcap")}),
    )
    monkeypatch.setattr(native, "audit", lambda *_: source_audit)
    monkeypatch.setattr(
        native, "verify_fixture", lambda output: write_json(output, {"synthetic_unit_test": True})
    )
    monkeypatch.setattr(native, "iter_phase_rows", lambda *_: iter(copy.deepcopy(rows)))
    control = [
        {
            name: value
            for name, value in row.items()
            if name not in [*native.LABEL_FIELDS, "label", "status"]
        }
        for row in rows
    ]
    monkeypatch.setattr(
        native, "iter_context_rows", lambda *_a, **_kw: iter(copy.deepcopy(control))
    )
    output = tmp_path / "native"
    result = native.prepare(tmp_path, output)
    assert result["audit"]["rows"] == result["context_control"]["rows"] == 5
    assert result["audit"]["class_segments"] == {"0": 2, "1": 2}
    assert result["audit"]["status_counts"]["mixed_phase"] == 1
    assert result["audit"]["callback_packet_counts"] == {"benign": 6, "attack": 6, "unknown": 0}
    assert result["audit"]["tuple_groups_with_both_labels"] == 1
    assert result["context_control"]["all_values_exact"]
    assert not result["training_authorized"]
    db = sqlite3.connect(output / "flows.sqlite")
    assert db.execute("SELECT COUNT(*) FROM flows WHERE label IS NULL").fetchone()[0] == 1
    db.execute("UPDATE flows SET label=1 WHERE ordinal=1")
    db.commit()
    db.close()
    with pytest.raises(ValueError, match="labels"):
        native.audit_database(output / "flows.sqlite")


def test_registered_phase_lab_refuses_training_or_time_range_changes(tmp_path):
    config = native.configuration()
    config["training_authorized"] = True
    path = tmp_path / "config.json"
    write_json(path, config)
    with pytest.raises(ValueError, match="registered"):
        native.configuration(path)


def test_real_linux_fixture_has_exact_native_values_and_hand_counted_packet_labels(tmp_path):
    if platform.system() != "Linux":
        pytest.skip("Verified exporter requires CPython Linux")
    pytest.importorskip("nfstream")
    result = native.verify_fixture(tmp_path / "fixture.json")
    assert result["passed"] and result["checked_values"] == 280
    assert result["mixed_segments_retained"] == 1

import json
import platform
from pathlib import Path
from types import SimpleNamespace

import pytest

from netattdetector.causal_context import (
    CONTEXT_FEATURES,
    CONTEXT_METADATA,
    FlowStartContext,
    endpoint,
    validate_context,
)
from netattdetector.context_cohort import create_database, independent_sql_snapshot
from netattdetector.context_meter import CausalFlowStartPlugin, iter_context_rows
from netattdetector.measurement import flow_key
from netattdetector.statistical import FEATURES

A, B, C = map(endpoint, ("192.0.2.1", "192.0.2.2", "2001:db8::1"))
D, E = map(endpoint, ("198.51.100.1", "198.51.100.2"))


def test_current_and_same_watermark_peers_are_excluded():
    state = FlowStartContext()
    first = state.observe(0, A, D, 80)
    peer = state.observe(0, A, D, 443)
    assert all(first[name] == peer[name] == 0 for name in CONTEXT_FEATURES)
    later = state.observe(1, A, D, 80)
    assert [later[name] for name in CONTEXT_FEATURES[:6]] == [2] * 6
    assert later["src_distinct_destinations_60s"] == 1
    assert later["src_distinct_dst_ports_60s"] == 2
    assert later["dst_distinct_sources_60s"] == 1
    validate_context(later)


def test_inclusive_lower_bound_expiry_and_distinct_counter_cleanup():
    state = FlowStartContext()
    state.observe(0, A, D, 80)
    boundary = state.observe(1000, A, D, 443)
    assert boundary["src_flow_starts_1s"] == 1
    other = state.observe(1001, B, D, 443)
    assert other["dst_flow_starts_1s"] == 1 and other["dst_flow_starts_10s"] == 2
    assert other["dst_distinct_sources_60s"] == 1
    sixty = state.observe(60_000, A, E, 123)
    assert sixty["src_flow_starts_60s"] == 2  # The start at0 remains on the inclusive boundary.
    assert sixty["src_distinct_dst_ports_60s"] == 2
    expired = state.observe(60_001, A, D, 80)
    assert expired["src_flow_starts_60s"] == 2
    assert expired["src_distinct_destinations_60s"] == 2
    assert expired["dst_distinct_sources_60s"] == 2
    empty = state.observe(200_000, C, E, 80)
    assert all(empty[name] == 0 for name in CONTEXT_FEATURES)
    assert not state.source_peers and not state.destination_peers and not state.source_ports
    assert empty["context_warmup_complete"] == 1


def test_late_clock_keeps_arrival_order_and_never_retroactively_reorders():
    state = FlowStartContext()
    state.observe(2000, A, D, 80)
    late = state.observe(1000, B, D, 80)
    assert late["context_watermark_ms"] == 2000 and late["context_late_start"] == 1
    assert all(late[name] == 0 for name in CONTEXT_FEATURES)
    next_row = state.observe(2001, C, D, 80)
    assert next_row["dst_flow_starts_1s"] == 2
    assert next_row["dst_distinct_sources_60s"] == 2
    assert next_row["context_init_sequence"] == 3


def test_retained_snapshots_are_unchanged_by_future_events():
    prefix = [(0, A, D, 80), (1000, B, D, 443)]
    first = FlowStartContext()
    saved = [first.observe(*event) for event in prefix]
    reference = [row.copy() for row in saved]
    for event in [(1001, A, E, 123), (500, C, D, 80), (100_000, A, D, 80)]:
        first.observe(*event)
    assert saved == reference
    second = FlowStartContext()
    assert [second.observe(*event) for event in prefix] == saved
    assert set(saved[0]) == set(CONTEXT_FEATURES + CONTEXT_METADATA)


def test_state_budget_is_terminal_instead_of_returning_missing_zeros():
    state = FlowStartContext(max_events=2)
    state.observe(0, A, D, 80)
    state.observe(0, B, D, 80)
    with pytest.raises(ValueError, match="bound reached"):
        state.observe(0, C, D, 80)
    with pytest.raises(ValueError, match="already failed"):
        state.observe(100_000, C, D, 80)


def test_callback_stores_only_primitive_snapshot_values_and_checks_start_time():
    plugin = CausalFlowStartPlugin()
    flow = SimpleNamespace(
        src_ip="192.0.2.1",
        dst_ip="198.51.100.1",
        dst_port=80,
        bidirectional_first_seen_ms=0,
        udps=SimpleNamespace(),
    )
    plugin.on_init(SimpleNamespace(time=0), flow)
    assert all(type(value) is int for value in vars(flow.udps).values())
    assert flow.udps.ctx_context_init_sequence == 1
    with pytest.raises(ValueError, match="packet time"):
        plugin.on_init(SimpleNamespace(time=1), flow)
    plugin.cleanup()
    assert plugin.state is None


def test_independent_sql_uses_prior_bins_and_sequence_without_future_events(tmp_path):
    db = create_database(tmp_path / "targets.sqlite", targets=True)
    try:
        rows = [
            (1, 1, 0, 0, A, D, 80),
            (2, 2, 0, 0, B, D, 443),
            (3, 3, 1000, 1000, A, E, 80),
            (4, 4, 500, 1000, C, D, 80),
            (5, 5, 1200, 1200, A, E, 443),
        ]
        db.executemany("INSERT INTO targets VALUES (?,?,?,?,?,?,?)", rows)
        first = independent_sql_snapshot(db, 3, 1000, A, E)
        assert [first[name] for name in CONTEXT_FEATURES] == [1, 1, 1, 0, 0, 0, 1, 1, 0]
        late = independent_sql_snapshot(db, 4, 1000, C, D)
        assert [late[name] for name in CONTEXT_FEATURES] == [0, 0, 0, 2, 2, 2, 0, 0, 2]
        next_row = independent_sql_snapshot(db, 5, 1200, A, E)
        assert [next_row[name] for name in CONTEXT_FEATURES] == [1, 2, 2, 1, 1, 1, 2, 1, 1]
    finally:
        db.close()


def test_real_linux_callback_matches_known_five_flow_initialization_history():
    if platform.system() != "Linux":
        pytest.skip("Pinned callback exporter is Linux-only")
    pytest.importorskip("nfstream")
    expected = json.loads(Path("tests/fixtures/pcap/statistical-expected.json").read_text())["rows"]
    actual = {flow_key(row): row for row in iter_context_rows("tests/fixtures/pcap/core.pcap")}
    assert len(actual) == 5
    for sequence, known in enumerate(expected, 1):
        row = actual[flow_key(known)]
        assert all(row[name] == pytest.approx(known[name]) for name in FEATURES)
        assert row["context_init_sequence"] == sequence
        assert row["context_watermark_ms"] == known["bidirectional_first_seen_ms"]
        assert row["context_warmup_complete"] == row["context_late_start"] == 0
        assert row["context_peak_events"] == sequence
        wanted = {name: 0 for name in CONTEXT_FEATURES}
        if sequence == 5:
            wanted.update(
                {
                    "src_flow_starts_10s": 1,
                    "src_flow_starts_60s": 1,
                    "dst_flow_starts_10s": 1,
                    "dst_flow_starts_60s": 1,
                    "src_distinct_destinations_60s": 1,
                    "src_distinct_dst_ports_60s": 1,
                    "dst_distinct_sources_60s": 1,
                }
            )
        assert {name: row[name] for name in CONTEXT_FEATURES} == wanted

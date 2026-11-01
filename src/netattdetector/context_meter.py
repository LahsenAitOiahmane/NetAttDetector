"""Offline NFPlugin adapter for causal initialization snapshots, not live capture."""

import json
import math
from pathlib import Path

from netattdetector.causal_context import (
    CONTEXT_FEATURES,
    CONTEXT_METADATA,
    FlowStartContext,
    endpoint,
    validate_context,
)
from netattdetector.io import sha256_file, write_json
from netattdetector.measurement import METADATA, _iter_meter_rows, flow_key
from netattdetector.statistical import FEATURES, FLOAT_FIELDS, validate_statistics

try:
    from nfstream import NFPlugin as _PluginBase
except ImportError:
    _PluginBase = (
        object  # Enables platform-neutral state/callback tests; export still requires Linux.
    )


class CausalFlowStartPlugin(_PluginBase):
    def __init__(self, max_events=100_000):
        self.max_events = max_events
        self.state = None

    def on_init(self, packet, flow):
        if packet.time != flow.bidirectional_first_seen_ms:
            raise ValueError("Initialization packet time differs from the native first-seen time.")
        if self.state is None:
            self.state = FlowStartContext(self.max_events)
        values = self.state.observe(
            int(packet.time), endpoint(flow.src_ip), endpoint(flow.dst_ip), int(flow.dst_port)
        )
        for name, value in values.items():
            setattr(flow.udps, "ctx_" + name, value)

    def cleanup(self):
        self.state = None


def iter_context_rows(path, max_nflows=10_000, *, max_pcap_bytes=16 * 2**20, max_events=100_000):
    plugin = CausalFlowStartPlugin(max_events)
    for row in _iter_meter_rows(
        path, max_nflows, max_pcap_bytes, [*FEATURES, *METADATA, "udps"], True, udps=plugin
    ):
        user = row.pop("udps")
        row.update(
            {name: getattr(user, "ctx_" + name) for name in [*CONTEXT_FEATURES, *CONTEXT_METADATA]}
        )
        validate_statistics(row)
        validate_context(row, max_events)
        yield row


def verify_context_fixture(output):
    if Path(output).exists():
        raise FileExistsError(output)
    known = json.loads(
        Path("tests/fixtures/pcap/statistical-expected.json").read_text(encoding="utf-8")
    )
    actual_rows = list(iter_context_rows("tests/fixtures/pcap/core.pcap"))
    actual = {flow_key(row): row for row in actual_rows}
    errors, checked = [], 0
    if (
        len(actual_rows) != len(actual)
        or len(actual) != len(known["rows"])
        or set(actual) != {flow_key(row) for row in known["rows"]}
    ):
        errors.append("Context fixture flow identities are incomplete/repeated.")
    for sequence, base in enumerate(known["rows"], 1):
        context = {name: 0 for name in CONTEXT_FEATURES}
        if sequence == 5:
            context.update(
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
        expected = {
            **base,
            **context,
            "context_init_sequence": sequence,
            "context_watermark_ms": base["bidirectional_first_seen_ms"],
            "context_warmup_complete": 0,
            "context_late_start": 0,
            "context_peak_events": sequence,
        }
        row = actual.get(flow_key(base))
        if row is None:
            continue
        if set(row) != set(expected):
            errors.append("Context fixture field contract differs.")
            continue
        for name, value in expected.items():
            checked += 1
            matches = (
                math.isclose(row[name], value, rel_tol=1e-9, abs_tol=1e-9)
                if name in FLOAT_FIELDS
                else row[name] == value
            )
            if not matches:
                errors.append(
                    f"Context fixture mismatch: flow {sequence}, {name}: {row[name]} != {value}."
                )
    result = {
        "passed": not errors,
        "flows": len(actual),
        "checked_fields": checked,
        "errors": errors,
        "pcap_sha256": sha256_file("tests/fixtures/pcap/core.pcap"),
        "training_authorized": False,
        "scope": "Known five-flow/twelve-packet initialization order; core/statistical fields plus independent reused-endpoint context expectations. Not attack detection or live loss evidence.",
    }
    write_json(output, result)
    if not result["passed"]:
        raise ValueError(f"Causal context fixture failed; see {output}.")
    return result

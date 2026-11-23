"""Candidate packet-phase labels, separate from unchanged native feature measurements."""

import hashlib
import ipaddress
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path

from netattdetector.causal_context import (
    CONTEXT_FEATURES,
    CONTEXT_METADATA,
    SCHEMA,
    validate_context,
)
from netattdetector.context_cohort import FEATURES
from netattdetector.context_meter import CausalFlowStartPlugin, iter_context_rows
from netattdetector.io import sha256_file, write_json
from netattdetector.kitsune import audit, verified_files
from netattdetector.measurement import METADATA, _iter_meter_rows, flow_key
from netattdetector.resources import ResourceMonitor
from netattdetector.statistical import FLOAT_FIELDS, STAT_SETTINGS, validate_statistics
from netattdetector.statistical_cohort import configure_database, connect_readonly, pattern_hash

CONFIG = "configs/data/kitsune-native.json"
LABEL_FIELDS = [
    "phase_mask",
    "phase_packets",
    "callback_benign_packets",
    "callback_attack_packets",
    "callback_unknown_packets",
    "callback_min_ms",
    "callback_max_ms",
]
COLUMNS = [
    "ordinal",
    "flow_hash",
    "group_hash",
    "label",
    "status",
    "first_ms",
    "last_ms",
    *LABEL_FIELDS,
    *FEATURES,
    *CONTEXT_METADATA,
    "feature_hash",
    "no_ports_hash",
]


def configuration(path=CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "study_id": "kitsune-native-phase-lab-v1",
        "source_profile": "configs/data/kitsune-development.json",
        "expected_pcap_sha256": "3bebe40dbf24546be36880f89908e315f8f858c8256aa00de0b3bbab8c3c69e6",
        "expected_labels_sha256": "190d5cc70c35e11000f98f5e508d0d5e50eed82c0ea3acd1abefda8d18a95817",
        "expected_packets": 764137,
        "expected_class_counts": {"0": 121621, "1": 642516},
        "label_time_ranges_ns": {
            "0": {"minimum_ns": 1540446382933899000, "maximum_ns": 1540450841432895000},
            "1": {"minimum_ns": 1540450841473288000, "maximum_ns": 1540453519842464000},
        },
        "schema": SCHEMA,
        "label_policy": "kitsune-packet-phase-candidate-v1",
        "maximum_flows": 250000,
        "maximum_pcap_bytes": 128 * 2**20,
        "maximum_context_events": 100000,
        "maximum_database_bytes": 512 * 2**20,
        "training_authorized": False,
        "deployment_authorized": False,
        "public_model_release_authorized": False,
    }
    if any(config.get(name) != value for name, value in expected.items()):
        raise ValueError("Use the registered training-disabled Kitsune phase-label lab.")
    return config


class PacketPhases:
    def __init__(self, ranges):
        if set(ranges) != {"0", "1"}:
            raise ValueError("Both proven packet-label time ranges are required.")
        self.windows = []
        for label in ("0", "1"):
            row = ranges[label]
            if (
                set(row) != {"minimum_ns", "maximum_ns"}
                or any(type(v) is not int or not 0 <= v < 2**63 for v in row.values())
                or row["minimum_ns"] > row["maximum_ns"]
            ):
                raise ValueError("Invalid packet-label time range.")
            self.windows.append((row["minimum_ns"] // 1_000_000, row["maximum_ns"] // 1_000_000))
        if self.windows[0][1] >= self.windows[1][0]:
            raise ValueError("Packet-label ranges overlap after millisecond quantization.")

    def label(self, timestamp):
        if type(timestamp) is not int:
            raise ValueError("Callback clock must be exact integer milliseconds.")
        for label, (low, high) in enumerate(self.windows):
            if low <= timestamp <= high:
                return label
        return None


class PhaseContextPlugin(CausalFlowStartPlugin):
    def __init__(self, ranges, max_events=100000):
        super().__init__(max_events)
        self.phases = PacketPhases(ranges)

    def on_init(self, packet, flow):
        super().on_init(packet, flow)
        for name in LABEL_FIELDS[:5]:
            setattr(flow.udps, name, 0)
        flow.udps.callback_min_ms = flow.udps.callback_max_ms = int(packet.time)
        self.on_update(packet, flow)

    def on_update(self, packet, flow):
        label = self.phases.label(packet.time)
        field = {
            0: "callback_benign_packets",
            1: "callback_attack_packets",
            None: "callback_unknown_packets",
        }[label]
        setattr(flow.udps, field, getattr(flow.udps, field) + 1)
        flow.udps.phase_mask |= {0: 1, 1: 2, None: 4}[label]
        flow.udps.phase_packets += 1
        flow.udps.callback_min_ms = min(flow.udps.callback_min_ms, int(packet.time))
        flow.udps.callback_max_ms = max(flow.udps.callback_max_ms, int(packet.time))


def label_row(row):
    if any(type(row[name]) is not int or not 0 <= row[name] < 2**63 for name in LABEL_FIELDS):
        raise ValueError("Phase callback fields must be exact nonnegative integers.")
    benign, attack, unknown = (row[name] for name in LABEL_FIELDS[2:5])
    expected_mask = (1 if benign else 0) | (2 if attack else 0) | (4 if unknown else 0)
    if (
        expected_mask != row["phase_mask"]
        or row["phase_packets"] != benign + attack + unknown
        or row["phase_packets"] != row["bidirectional_packets"]
        or row["phase_packets"] == 0
        or row["callback_min_ms"] > row["callback_max_ms"]
    ):
        raise ValueError("Phase callbacks do not reconcile with complete native packet counts.")
    if unknown:
        return None, "unknown_clock"
    if row["phase_mask"] == 3:
        return None, "mixed_phase"
    return (0, "uniform_benign") if benign else (1, "uniform_attack")


def iter_phase_rows(
    path, ranges, *, maximum_flows=250000, maximum_bytes=128 * 2**20, max_events=100000
):
    plugin = PhaseContextPlugin(ranges, max_events)
    for row in _iter_meter_rows(
        path, maximum_flows, maximum_bytes, [*FEATURES[:27], *METADATA, "udps"], True, udps=plugin
    ):
        user = row.pop("udps")
        row.update(
            {name: getattr(user, "ctx_" + name) for name in [*CONTEXT_FEATURES, *CONTEXT_METADATA]}
        )
        row.update({name: getattr(user, name) for name in LABEL_FIELDS})
        validate_statistics(row)
        validate_context(row, max_events)
        row["label"], row["status"] = label_row(row)
        yield row


def verify_fixture(output):
    known = json.loads(Path("tests/fixtures/pcap/statistical-expected.json").read_text())["rows"]
    ranges = {
        "0": {"minimum_ns": 1600000001000000000, "maximum_ns": 1600000003200000000},
        "1": {"minimum_ns": 1600000003320000000, "maximum_ns": 1600000010000000000},
    }
    rows = list(
        iter_phase_rows(
            "tests/fixtures/pcap/core.pcap", ranges, maximum_flows=10000, maximum_bytes=16 * 2**20
        )
    )
    control = list(iter_context_rows("tests/fixtures/pcap/core.pcap"))
    expected_counts = [(2, 0), (1, 0), (3, 3), (0, 2), (0, 1)]
    if len(rows) != len(control) or len(rows) != len(known):
        raise ValueError("Kitsune phase fixture export is incomplete.")
    checks = 0
    for row, old, base, (benign, attack) in zip(rows, control, known, expected_counts, strict=True):
        if any(row[name] != value for name, value in old.items()) or any(
            row[name] != value for name, value in base.items()
        ):
            raise ValueError("Phase labeling changed a known original fixture measurement.")
        checks += len(old)
        values = {
            "callback_benign_packets": benign,
            "callback_attack_packets": attack,
            "callback_unknown_packets": 0,
            "phase_packets": benign + attack,
            "phase_mask": (1 if benign else 0) | (2 if attack else 0),
            "callback_min_ms": base["bidirectional_first_seen_ms"],
            "callback_max_ms": base["bidirectional_last_seen_ms"],
            "label": None if benign and attack else 0 if benign else 1,
            "status": "mixed_phase"
            if benign and attack
            else "uniform_benign"
            if benign
            else "uniform_attack",
        }
        if any(row[name] != value for name, value in values.items()):
            raise ValueError("Phase fixture callback labels/counts differ from hand counts.")
        checks += len(values)
    result = {
        "passed": True,
        "checked_values": checks,
        "flows": 5,
        "packets": 12,
        "benign_callback_packets": 6,
        "attack_callback_packets": 6,
        "mixed_segments_retained": 1,
        "scope": "Known original 36-feature/event/context values plus hand-counted callback labels/masks/min/max. Not attack truth.",
    }
    write_json(output, result)
    return result


def digest_row(digest, values):
    digest.update(json.dumps(values, separators=(",", ":"), allow_nan=False).encode() + b"\n")


def grouping(row, source_hash):
    endpoints = sorted(
        (
            ipaddress.ip_address(row[address]).version,
            int(ipaddress.ip_address(row[address])),
            row[port],
        )
        for address, port in (("src_ip", "src_port"), ("dst_ip", "dst_port"))
    )
    return hashlib.sha256(
        json.dumps([source_hash, row["protocol"], endpoints], separators=(",", ":")).encode()
    ).hexdigest()


def audit_database(path):
    db = connect_readonly(path)
    try:
        configure_database(db)
        if (
            db.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
            or [row[1] for row in db.execute("PRAGMA table_info(flows)")] != COLUMNS
        ):
            raise ValueError("Kitsune native schema/integrity check failed.")
        logical, count = hashlib.sha256(), 0
        for values in db.execute("SELECT * FROM flows ORDER BY ordinal"):
            count += 1
            row = dict(zip(COLUMNS, values, strict=True))
            row["bidirectional_packets"] = row["src2dst_packets"] + row["dst2src_packets"]
            if (
                row["ordinal"] != count
                or (row["label"], row["status"]) != label_row(row)
                or row["feature_hash"] != pattern_hash([row[name] for name in FEATURES])
                or row["no_ports_hash"] != pattern_hash([row[name] for name in FEATURES[2:]])
            ):
                raise ValueError("Kitsune native labels, sequence or feature hash changed.")
            validate_context(row)
            digest_row(logical, list(values))
        patterns = {}
        for field in ("feature_hash", "no_ports_hash"):
            unique = mixed = affected = 0
            for rows, labels in db.execute(
                f"SELECT COUNT(*),COUNT(DISTINCT label) FROM flows WHERE label IS NOT NULL GROUP BY {field}"
            ):
                unique += 1
                mixed += labels > 1
                affected += rows if labels > 1 else 0
            patterns[field] = {
                "unique_vectors": unique,
                "mixed_label_vectors": mixed,
                "rows_in_mixed_vectors": affected,
            }
        return {
            "rows": count,
            "logical_sha256": logical.hexdigest(),
            "status_counts": dict(db.execute("SELECT status,COUNT(*) FROM flows GROUP BY status")),
            "class_segments": {
                str(label): n
                for label, n in db.execute(
                    "SELECT label,COUNT(*) FROM flows WHERE label IS NOT NULL GROUP BY label"
                )
            },
            "callback_packet_counts": dict(
                zip(
                    ("benign", "attack", "unknown"),
                    db.execute(
                        "SELECT COALESCE(SUM(callback_benign_packets),0),COALESCE(SUM(callback_attack_packets),0),COALESCE(SUM(callback_unknown_packets),0) FROM flows"
                    ).fetchone(),
                    strict=True,
                )
            ),
            "warmup_segments": db.execute(
                "SELECT COUNT(*) FROM flows WHERE context_warmup_complete=0"
            ).fetchone()[0],
            "late_segments": db.execute(
                "SELECT COALESCE(SUM(context_late_start),0) FROM flows"
            ).fetchone()[0],
            "peak_context_events": db.execute(
                "SELECT COALESCE(MAX(context_peak_events),0) FROM flows"
            ).fetchone()[0],
            "first_seen_minimum_disagreements": db.execute(
                "SELECT COUNT(*) FROM flows WHERE first_ms!=callback_min_ms"
            ).fetchone()[0],
            "last_seen_maximum_disagreements": db.execute(
                "SELECT COUNT(*) FROM flows WHERE last_ms!=callback_max_ms"
            ).fetchone()[0],
            "repeated_flow_identities": db.execute(
                "SELECT COUNT(*) FROM (SELECT flow_hash FROM flows GROUP BY flow_hash HAVING COUNT(*)>1)"
            ).fetchone()[0],
            "native_tuple_groups": db.execute(
                "SELECT COUNT(DISTINCT group_hash) FROM flows"
            ).fetchone()[0],
            "tuple_groups_with_both_labels": db.execute(
                "SELECT COUNT(*) FROM (SELECT group_hash FROM flows WHERE label IS NOT NULL GROUP BY group_hash HAVING COUNT(DISTINCT label)>1)"
            ).fetchone()[0],
            "protocol_segments": {
                str(protocol): n
                for protocol, n in db.execute(
                    "SELECT protocol,COUNT(*) FROM flows GROUP BY protocol"
                )
            },
            "patterns": patterns,
            "group_limitation": "Capture-scoped bidirectional native endpoint/port/protocol hashes, not Zeek UID sessions/devices/independent attacks.",
        }
    finally:
        db.close()


def prepare(source, output, config_path=CONFIG):
    config = configuration(config_path)
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _, receipt, paths = verified_files(source, config["source_profile"])
    source_audit = audit(source, output / "packet-audit.json", config["source_profile"])
    if (
        source_audit["packets"] != config["expected_packets"]
        or source_audit["binary_class_counts"] != config["expected_class_counts"]
        or source_audit["time_ranges_by_label"] != config["label_time_ranges_ns"]
        or not source_audit["single_benign_prefix_then_attack_suffix"]
    ):
        raise ValueError("Kitsune packet-label proof differs from registered phase assumptions.")
    PacketPhases(source_audit["time_ranges_by_label"])
    verify_fixture(output / "fixture.json")
    if shutil.disk_usage(output).free - config["maximum_database_bytes"] < 2 * 2**30:
        raise ValueError("Kitsune native preparation would breach disk reserve.")
    path = output / "flows.sqlite"
    with path.open("xb"):
        pass
    db = sqlite3.connect(path)
    configure_database(db)
    db.execute("PRAGMA max_page_count=131072")
    text = {"flow_hash", "group_hash", "status", "feature_hash", "no_ports_hash"}
    db.execute(
        "CREATE TABLE flows("
        + ",".join(
            "ordinal INTEGER PRIMARY KEY"
            if name == "ordinal"
            else "label INTEGER"
            if name == "label"
            else f"{name} TEXT NOT NULL"
            if name in text
            else f"{name} REAL NOT NULL"
            if name in FLOAT_FIELDS
            else f"{name} INTEGER NOT NULL"
            for name in COLUMNS
        )
        + ")"
    )
    logical, count = hashlib.sha256(), 0
    try:
        with ResourceMonitor(include_children=True) as resources:
            for count, row in enumerate(
                iter_phase_rows(paths["pcap"], config["label_time_ranges_ns"]), 1
            ):
                values = {
                    "ordinal": count,
                    "flow_hash": hashlib.sha256(
                        json.dumps(flow_key(row), separators=(",", ":")).encode()
                    ).hexdigest(),
                    "group_hash": grouping(row, config["expected_pcap_sha256"]),
                    "first_ms": row["bidirectional_first_seen_ms"],
                    "last_ms": row["bidirectional_last_seen_ms"],
                    "feature_hash": pattern_hash([row[name] for name in FEATURES]),
                    "no_ports_hash": pattern_hash([row[name] for name in FEATURES[2:]]),
                    **row,
                }
                ordered = [
                    float(values[name]) if name in FLOAT_FIELDS else values[name]
                    for name in COLUMNS
                ]
                db.execute(
                    "INSERT INTO flows VALUES(" + ",".join("?" for _ in COLUMNS) + ")", ordered
                )
                digest_row(logical, ordered)
                if count % 10000 == 0:
                    db.commit()
            for field in ("group_hash", "flow_hash", "feature_hash", "no_ports_hash"):
                db.execute(f"CREATE INDEX by_{field} ON flows({field})")
            db.commit()
    finally:
        db.close()
    with ResourceMonitor(include_children=True) as replay:
        db = connect_readonly(path)
        try:
            cursor = db.execute(
                "SELECT flow_hash,"
                + ",".join([*FEATURES, *CONTEXT_METADATA, "first_ms", "last_ms"])
                + " FROM flows ORDER BY ordinal"
            )
            compared, checked = 0, 0
            for row in iter_context_rows(
                paths["pcap"], max_nflows=250000, max_pcap_bytes=128 * 2**20
            ):
                stored = cursor.fetchone()
                expected = (
                    hashlib.sha256(
                        json.dumps(flow_key(row), separators=(",", ":")).encode()
                    ).hexdigest(),
                    *[row[name] for name in [*FEATURES, *CONTEXT_METADATA]],
                    row["bidirectional_first_seen_ms"],
                    row["bidirectional_last_seen_ms"],
                )
                if stored != expected:
                    raise ValueError(
                        "Phase callbacks changed an original native/context measurement or identity."
                    )
                compared += 1
                checked += len(expected)
            if cursor.fetchone() is not None or compared != count:
                raise ValueError("Context-only control export is incomplete.")
        finally:
            db.close()
    with ResourceMonitor() as auditing:
        verified = audit_database(path)
    if verified["logical_sha256"] != logical.hexdigest() or verified["rows"] != count:
        raise ValueError("Reopened Kitsune native rows differ from preparation.")
    callback_counts = verified["callback_packet_counts"]
    if (
        callback_counts["benign"] > config["expected_class_counts"]["0"]
        or callback_counts["attack"] > config["expected_class_counts"]["1"]
    ):
        raise ValueError("Native callbacks exceed the proven original class packet counts.")
    result = {
        "study_id": config["study_id"],
        "schema": SCHEMA,
        "features": FEATURES,
        "label_policy": config["label_policy"],
        "config_sha256": sha256_file(config_path),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "exporter_version": "6.6.0",
        "settings": {**STAT_SETTINGS, "max_nflows": 250000},
        "source_files": receipt["files"],
        "audit": verified,
        "source_packet_coverage": {
            "original_packet_classes": config["expected_class_counts"],
            "native_callback_classes": callback_counts,
            "benign_packet_fraction": callback_counts["benign"]
            / config["expected_class_counts"]["0"],
            "attack_packet_fraction": callback_counts["attack"]
            / config["expected_class_counts"]["1"],
            "scope": "Packets represented by native segments, including mixed/unknown segments; not detector accuracy or independent events.",
        },
        "context_control": {
            "passed": True,
            "rows": compared,
            "checked_values": checked,
            "all_values_exact": True,
        },
        "database": {"bytes": path.stat().st_size, "sha256": sha256_file(path)},
        "resources": {
            name: {"seconds": monitor.elapsed, "sampled_tree_rss_mib": monitor.peak_tree / 2**20}
            for name, monitor in (
                ("phase_export", resources),
                ("context_control", replay),
                ("read_only_audit", auditing),
            )
        },
        "training_authorized": False,
        "deployment_authorized": False,
        "public_model_release_authorized": False,
        "scope": "Candidate labels inherit publisher packet-phase annotations. Most original attack packets are non-IP and outside the model. Group hashes are not independent UID/session/device labels. No fitted model, source split or deployment acceptance.",
    }
    write_json(output / "manifest.json", result)
    return result

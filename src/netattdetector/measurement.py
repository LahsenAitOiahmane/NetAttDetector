"""Offline-only NFStream measurement adapter; incompatible with the UQ model contract."""

import importlib.metadata
import json
import platform
from pathlib import Path

from netattdetector.data import FEATURES
from netattdetector.io import sha256_file, write_json

SCHEMA = "nfstream-core-v1"
CORE_FEATURES = [
    "src_port",
    "dst_port",
    "protocol",
    "ip_version",
    "src2dst_packets",
    "dst2src_packets",
    "src2dst_bytes",
    "dst2src_bytes",
    "bidirectional_duration_ms",
    "src2dst_duration_ms",
    "dst2src_duration_ms",
]
METADATA = [
    "src_ip",
    "dst_ip",
    "bidirectional_first_seen_ms",
    "bidirectional_last_seen_ms",
    "bidirectional_packets",
    "bidirectional_bytes",
]
SETTINGS = {
    "decode_tunnels": False,
    "promiscuous_mode": False,
    "snapshot_length": 65535,
    "idle_timeout": 2,
    "active_timeout": 1800,
    "accounting_mode": 1,
    "n_dissections": 0,
    "statistical_analysis": False,
    "n_meters": 1,
    "max_nflows": 10_000,
    "system_visibility_mode": 0,
}


def contract_report():
    candidate = {
        "L4_SRC_PORT": "src_port",
        "L4_DST_PORT": "dst_port",
        "PROTOCOL": "protocol",
        "IN_BYTES": "src2dst_bytes",
        "IN_PKTS": "src2dst_packets",
        "OUT_BYTES": "dst2src_bytes",
        "OUT_PKTS": "dst2src_packets",
        "FLOW_DURATION_MILLISECONDS": "bidirectional_duration_ms",
        "DURATION_IN": "src2dst_duration_ms",
        "DURATION_OUT": "dst2src_duration_ms",
    }
    return {
        "schema": SCHEMA,
        "features": CORE_FEATURES,
        "settings": SETTINGS,
        "uq_model_compatible": False,
        "uq_fields": {
            name: {"conceptual_candidate": candidate.get(name), "equivalence_verified": False}
            for name in FEATURES
        },
        "decision": "Separate measurement contract. No aliases/zero-fill are emitted for UQ models. Require matching re-exported labeled training data and new preprocessing/model before prediction.",
        "limits": "Fixtures test selected core counters/durations, first-observed direction, timeout behavior and IPv4/IPv6. They do not validate nProbe parity, real traffic labels, retransmission semantics, VLANs, fragments, tunnels, payload inspection or live overload behavior.",
    }


def export_pcap(path, output):
    path, output = Path(path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not path.is_file() or path.suffix.casefold() not in {".pcap", ".pcapng"}:
        raise ValueError(
            "Provide an existing offline .pcap/.pcapng file; network interfaces are not accepted."
        )
    if not 0 < path.stat().st_size <= 16 * 2**20:
        raise ValueError("Measurement lab inputs must be nonempty and at most 16 MiB.")
    if platform.system() != "Linux":
        raise ValueError(
            "The verified NFStream adapter runs on Linux. Use the offline GitHub measurement workflow; no Windows capture driver is installed by this project."
        )
    try:
        from nfstream import NFStreamer
    except ImportError as error:
        raise ValueError("Install the locked Linux measurement extra.") from error
    from netattdetector.training import ResourceMonitor

    version = importlib.metadata.version("nfstream")
    if version != "6.6.0":
        raise ValueError("Measurement settings are registered for NFStream 6.6.0 only.")
    rows = []
    with ResourceMonitor() as resources:
        for flow in NFStreamer(source=str(path.resolve()), **SETTINGS):
            rows.append({name: getattr(flow, name) for name in [*CORE_FEATURES, *METADATA]})
            if len(rows) >= SETTINGS["max_nflows"]:
                raise ValueError(
                    "Flow limit reached; refuse to label a potentially partial export complete."
                )
    if not rows:
        raise ValueError("No supported flows were exported.")
    result = {
        "schema": SCHEMA,
        "pcap_sha256": sha256_file(path),
        "exporter": "nfstream",
        "exporter_version": version,
        "settings": SETTINGS,
        "features": CORE_FEATURES,
        "rows": rows,
        "scope": "offline measurement lab only; no attack scores or model predictions",
        "resources": {
            "wall_seconds": resources.elapsed,
            "sampled_peak_process_rss_mib": resources.peak / 2**20,
        },
    }
    write_json(output, result)
    return result


def flow_key(row):
    return tuple(
        row[name]
        for name in (
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
            "protocol",
            "bidirectional_first_seen_ms",
        )
    )


def verify_measurements(export_path, expected_path, output):
    if Path(output).exists():
        raise FileExistsError(output)
    actual = json.loads(Path(export_path).read_text(encoding="utf-8"))
    expected = json.loads(Path(expected_path).read_text(encoding="utf-8"))
    errors = []
    if (
        actual["schema"] != SCHEMA
        or actual["features"] != CORE_FEATURES
        or actual["settings"] != SETTINGS
        or actual["exporter_version"] != "6.6.0"
    ):
        errors.append("Exporter/schema/settings do not match the registered contract.")
    if actual["pcap_sha256"] != expected["pcap_sha256"]:
        errors.append("Fixture PCAP digest differs.")
    left = {flow_key(row): row for row in actual["rows"]}
    right = {flow_key(row): row for row in expected["rows"]}
    if len(left) != len(actual["rows"]) or len(right) != len(expected["rows"]):
        errors.append("Duplicate flow identities in expected/actual records.")
    if left.keys() != right.keys():
        errors.append(f"Flow identity sets differ: {len(left)} actual, {len(right)} expected.")
    checked = 0
    for key in left.keys() & right.keys():
        if set(right[key]) != set(CORE_FEATURES + METADATA):
            errors.append("Expected record must cover every registered feature and metadata field.")
        for name, value in right[key].items():
            checked += 1
            if left[key].get(name) != value:
                errors.append(
                    f"Mismatch for {key}, {name}: expected {value}, actual {left[key].get(name)}."
                )
    result = {
        "passed": not errors,
        "schema": SCHEMA,
        "export_sha256": sha256_file(export_path),
        "expected_sha256": sha256_file(expected_path),
        "checked_fields": checked,
        "flows": len(left),
        "errors": errors,
        "uq_model_compatible": False,
        "scope": "synthetic offline measurement regression, not network-attack detection evidence",
    }
    write_json(output, result)
    return result

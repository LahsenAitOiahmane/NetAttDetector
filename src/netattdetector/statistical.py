"""Separate pinned statistical feature contract and independent fixture verification."""

import json
import math
from pathlib import Path

from netattdetector.io import sha256_file, write_json
from netattdetector.measurement import CORE_FEATURES, METADATA, SETTINGS, _iter_meter_rows, flow_key
from netattdetector.resources import ResourceMonitor

SCHEMA = "nfstream-statistical-v1"
EXTRA_FEATURES = [
    "bidirectional_min_ps",
    "bidirectional_mean_ps",
    "bidirectional_stddev_ps",
    "bidirectional_max_ps",
    "bidirectional_min_piat_ms",
    "bidirectional_mean_piat_ms",
    "bidirectional_stddev_piat_ms",
    "bidirectional_max_piat_ms",
    *[
        f"bidirectional_{flag}_packets"
        for flag in ("syn", "cwr", "ece", "urg", "ack", "psh", "rst", "fin")
    ],
]
FEATURES = [*CORE_FEATURES, *EXTRA_FEATURES]
STAT_SETTINGS = {**SETTINGS, "statistical_analysis": True}
FLOAT_FIELDS = {
    "bidirectional_mean_ps",
    "bidirectional_stddev_ps",
    "bidirectional_mean_piat_ms",
    "bidirectional_stddev_piat_ms",
}


def iter_statistical_rows(path, max_nflows=10_000, *, max_pcap_bytes=16 * 2**20):
    yield from _iter_meter_rows(path, max_nflows, max_pcap_bytes, FEATURES + METADATA, True)


def validate_statistics(row):
    for name in [*CORE_FEATURES, *METADATA[2:]]:
        if type(row[name]) is not int or not 0 <= row[name] < 2**63:
            raise ValueError("Core measurement fields must remain exact nonnegative int64 values.")
    for name in EXTRA_FEATURES:
        value = row[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Statistical fields must be finite nonnegative numbers.")
        if name not in FLOAT_FIELDS and type(value) is not int:
            raise ValueError("Counts and min/max statistical fields must be integers.")
    packets = row["bidirectional_packets"]
    for name in EXTRA_FEATURES[8:]:
        if row[name] > packets or (row["protocol"] != 6 and row[name] != 0):
            raise ValueError("TCP flag counts exceed packets or appear on non-TCP flows.")
    for suffix in ("ps", "piat_ms"):
        low, mean, high = [row[f"bidirectional_{name}_{suffix}"] for name in ("min", "mean", "max")]
        if low > mean + 1e-9 or mean > high + 1e-9:
            raise ValueError("Statistical minimum/mean/maximum are inconsistent.")
    if packets == 1 and any(
        row[f"bidirectional_{name}_piat_ms"] != 0 for name in ("min", "mean", "stddev", "max")
    ):
        raise ValueError("Single-packet flows cannot have interarrival observations.")


def export_statistical_pcap(path, output):
    if Path(output).exists():
        raise FileExistsError(output)
    with ResourceMonitor(include_children=True) as resources:
        rows = []
        for row in iter_statistical_rows(path):
            validate_statistics(row)
            rows.append(row)
    if not rows:
        raise ValueError("No supported statistical flows exported.")
    result = {
        "schema": SCHEMA,
        "features": FEATURES,
        "settings": STAT_SETTINGS,
        "exporter_version": "6.6.0",
        "pcap_sha256": sha256_file(path),
        "rows": rows,
        "resources": {
            "wall_seconds": resources.elapsed,
            "sampled_peak_process_tree_rss_sum_mib": resources.peak_tree / 2**20,
        },
        "scope": "Bounded offline statistical measurement only; no model scores.",
    }
    write_json(output, result)
    return result


def verify_statistical_measurements(export, expected, output):
    if Path(output).exists():
        raise FileExistsError(output)
    actual = json.loads(Path(export).read_text(encoding="utf-8"))
    known = json.loads(Path(expected).read_text(encoding="utf-8"))
    errors = []
    if (
        actual["schema"] != SCHEMA
        or actual["features"] != FEATURES
        or actual["settings"] != STAT_SETTINGS
        or actual["exporter_version"] != "6.6.0"
        or actual["pcap_sha256"] != known["pcap_sha256"]
    ):
        errors.append("Statistical input/schema/settings/version mismatch.")
    left = {flow_key(row): row for row in actual["rows"]}
    right = {flow_key(row): row for row in known["rows"]}
    if (
        len(left) != len(actual["rows"])
        or len(right) != len(known["rows"])
        or left.keys() != right.keys()
    ):
        errors.append("Statistical fixture identity sets are incomplete or repeated.")
    checked = 0
    for key in left.keys() & right.keys():
        if set(left[key]) != set(FEATURES + METADATA) or set(right[key]) != set(
            FEATURES + METADATA
        ):
            errors.append("Statistical fixture must cover every declared field.")
            continue
        try:
            validate_statistics(left[key])
            validate_statistics(right[key])
        except ValueError as error:
            errors.append(str(error))
        for name, value in right[key].items():
            checked += 1
            found = left[key][name]
            matches = (
                math.isclose(found, value, rel_tol=1e-9, abs_tol=1e-9)
                if name in FLOAT_FIELDS and type(found) in (int, float)
                else found == value
            )
            if not matches:
                errors.append(f"Statistical fixture mismatch: {key}, {name}: {found} != {value}.")
    result = {
        "passed": not errors,
        "schema": SCHEMA,
        "flows": len(left),
        "checked_fields": checked,
        "errors": errors,
        "export_sha256": sha256_file(export),
        "expected_sha256": sha256_file(expected),
        "training_authorized": False,
    }
    write_json(output, result)
    return result

"""Bounded IoT-23 acquisition and conservative offline Zeek/NFStream label linkage."""

import hashlib
import ipaddress
import json
import math
import os
import re
import shutil
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from netattdetector.io import sha256_file, write_json
from netattdetector.label_index import SQLiteLabelIndex
from netattdetector.measurement import CORE_FEATURES, SCHEMA, SETTINGS, flow_key, iter_pcap_rows
from netattdetector.resources import ResourceMonitor

CATALOG = Path("configs/data/iot23-development.json")
BASE = "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/"
POLICY = "iot23-interval-v1"


def configuration(config_path, capture_id):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if (
        config.get("schema_version") != 1
        or config.get("source_id") != "iot23"
        or config.get("base_url") != BASE
        or config.get("join_policy") != POLICY
        or config.get("time_tolerance_ms") != 1
        or config.get("training_enabled") is not False
        or config.get("max_audit_flows") not in (100_000, 250_000)
    ):
        raise ValueError("Use the registered, training-disabled IoT-23 audit policy.")
    for name, minimum, maximum in (
        ("max_file_bytes", 1, 128 * 2**20),
        ("max_capture_bytes", 1, 160 * 2**20),
        ("disk_reserve_bytes", 2 * 2**30, 100 * 2**30),
    ):
        value = config.get(name)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"Invalid registered budget: {name}.")
    if (
        config.get("max_pcap_bytes", 16 * 2**20) not in (16 * 2**20, 128 * 2**20)
        or config.get("max_label_bytes", 16 * 2**20) not in (16 * 2**20, 32 * 2**20)
        or config.get("max_label_rows", 100_000) not in (100_000, 250_000)
    ):
        raise ValueError("Use registered PCAP/label-size/label-row budgets.")
    if capture_id not in config["captures"] or not re.fullmatch(
        r"CTU-IoT-Malware-Capture-\d+-1", capture_id
    ):
        raise ValueError("Unknown registered development capture.")
    capture = config["captures"][capture_id]
    if Path(capture["pcap"]).name != capture["pcap"] or not capture["pcap"].endswith(".pcap"):
        raise ValueError("Invalid registered PCAP filename.")
    return config, capture


def bounded_fetch(url, target, limit, expected_sha256=None):
    """Read at most limit+1 response bytes; never follow a redirect off the publisher host."""

    class PublisherRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            parsed = urlsplit(newurl)
            if parsed.scheme != "https" or parsed.hostname != "mcfp.felk.cvut.cz":
                raise ValueError("Publisher redirected off the registered HTTPS host.")
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    request = urllib.request.Request(url, headers={"User-Agent": "NetAttDetector-research/0.3"})
    if type(limit) is not int or not 0 < limit <= 128 * 2**20:
        raise ValueError("Invalid bounded download budget.")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    created = False
    deadline = time.monotonic() + 300
    try:
        with urllib.request.build_opener(PublisherRedirect()).open(request, timeout=45) as response:
            if response.status != 200:
                raise ValueError("Expected a complete publisher response.")
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isdigit() or not 0 < int(length) <= limit):
                raise ValueError("Publisher file exceeds the registered byte budget.")
            digest, total = hashlib.sha256(), 0
            with partial.open("xb") as stream:
                created = True
                read_chunk = getattr(response, "read1", response.read)
                while True:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Publisher download exceeded the five-minute deadline.")
                    chunk = read_chunk(min(65536, limit + 1 - total))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > limit:
                        raise ValueError("Over-budget publisher response.")
                    digest.update(chunk)
                    stream.write(chunk)
            if not total or (length and total != int(length)):
                raise ValueError("Empty or truncated publisher response.")
        checksum = digest.hexdigest()
        if expected_sha256 is not None and checksum != expected_sha256:
            raise ValueError("Publisher bytes changed from the registered SHA-256.")
        # Atomic creation refuses an existing destination on Windows and Linux.
        os.link(partial, target)
        return {"url": url, "bytes": total, "sha256": checksum}
    finally:
        if created:
            partial.unlink(missing_ok=True)


def acquire_capture(capture_id, output, config_path=CATALOG, require_pins=True):
    config, capture = configuration(config_path, capture_id)
    roles = json.loads(Path("configs/data/source_roles.json").read_text(encoding="utf-8"))
    if roles["sources"]["iot23"]["role"] != "development":
        raise ValueError("IoT-23 must remain registered as development.")
    names = [capture["pcap"], "bro/conn.log.labeled", "README.md"]
    if require_pins and set(capture["sha256"]) != set(names):
        raise ValueError(
            "First acquisition requires explicit --record-first; pin receipts before replay."
        )
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    parent = output.resolve().parent
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < config["disk_reserve_bytes"] + config["max_capture_bytes"]:
        raise ValueError("Insufficient disk reserve for bounded capture acquisition.")
    output.mkdir(parents=True)
    files, total = {}, 0
    for name in names:
        file_budget = (
            config.get("max_pcap_bytes", 16 * 2**20)
            if name == capture["pcap"]
            else config.get("max_label_bytes", 16 * 2**20)
            if name == "bro/conn.log.labeled"
            else 65536
        )
        limit = min(config["max_file_bytes"], config["max_capture_bytes"] - total, file_budget)
        files[name] = bounded_fetch(
            f"{BASE}{capture_id}/{name}", output / name, limit, capture["sha256"].get(name)
        )
        total += files[name]["bytes"]
    receipt = {
        "source_id": "iot23",
        "capture_id": capture_id,
        "role": "development",
        "acquired_at_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256_file(config_path),
        "source_roles_sha256": sha256_file("configs/data/source_roles.json"),
        "files": files,
        "bytes": total,
        "publisher_member_checksum_verified": False,
        "registered_hashes_verified": require_pins,
        "license_note": config["license_note"],
        "training_authorized": False,
    }
    write_json(output / "receipt.json", receipt)
    return receipt


def connection_key(src, sport, dst, dport, protocol):
    if protocol not in (6, 17):
        return None  # ICMP's exporter port conventions are not interchangeable with Zeek.
    endpoints = []
    for address, port in ((src, sport), (dst, dport)):
        ip = ipaddress.ip_address(address)
        if str(int(port)) != str(port) or not 0 <= int(port) <= 65535:
            raise ValueError("Invalid endpoint port.")
        endpoints.append((ip.version, int(ip), int(port)))
    return (protocol, *sorted(endpoints))


def read_zeek_labels(path, *, index=None, max_bytes=16 * 2**20, max_rows=100_000):
    """IoT-23 uses tabs plus spaces in its last three fields; no inferred binary labels."""
    path = Path(path)
    if (
        max_bytes not in (16 * 2**20, 32 * 2**20)
        or max_rows not in (100_000, 250_000)
        or not 0 < path.stat().st_size <= max_bytes
    ):
        raise ValueError("Label log exceeds the registered nonempty size/row budget.")
    columns, counts, uids = None, Counter(), set()
    disk_index = index is not None
    if index is None:
        index = defaultdict(list)
    required = {
        "ts",
        "uid",
        "id.orig_h",
        "id.orig_p",
        "id.resp_h",
        "id.resp_p",
        "proto",
        "duration",
        "label",
        "detailed-label",
    }
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if len(line) > 65536:
                raise ValueError("Oversized label-log line.")
            if line.startswith("#fields"):
                if columns is not None:
                    raise ValueError("Repeated label header.")
                columns = line.split()[1:]
                if "det_label" in columns:
                    if "detailed-label" in columns:
                        raise ValueError("Ambiguous detailed-label header aliases.")
                    columns[columns.index("det_label")] = "detailed-label"
                    counts["header_alias:det_label"] = 1
                if len(columns) != len(set(columns)) or not required <= set(columns):
                    raise ValueError("Unsupported label-log header.")
                continue
            if line.startswith("#") or not line.strip():
                continue
            values = line.split()
            if columns is None or len(values) != len(columns):
                raise ValueError(f"Malformed label row at line {number}.")
            counts["rows"] += 1
            if counts["rows"] > max_rows:
                raise ValueError("Label-log row limit exceeded.")
            row = dict(zip(columns, values, strict=True))
            if not disk_index and row["uid"] in uids:
                raise ValueError("Duplicate Zeek UID within a capture.")
            if not disk_index:
                uids.add(row["uid"])
            label = {"Benign": 0, "Malicious": 1}.get(row["label"])
            counts[f"coarse:{row['label']}"] += 1
            counts[f"detail:{row['detailed-label']}"] += 1
            try:
                start = Decimal(row["ts"]) * 1000
                duration = Decimal(row["duration"]) * 1000 if row["duration"] != "-" else None
                if duration is None:
                    counts[f"missing_duration:{row['label']}"] += 1
                if (
                    not start.is_finite()
                    or start < 0
                    or (duration is not None and (not duration.is_finite() or duration < 0))
                ):
                    raise ValueError("Invalid label timestamp/duration.")
            except InvalidOperation as error:
                raise ValueError("Invalid label timestamp/duration.") from error
            key = connection_key(
                row["id.orig_h"],
                row["id.orig_p"],
                row["id.resp_h"],
                row["id.resp_p"],
                {"tcp": 6, "udp": 17}.get(row["proto"]),
            )
            # Unknown labels and unknown durations stay in the index as blockers.
            candidate = {
                "uid": row["uid"],
                "start": start,
                "end": start + duration if duration is not None else None,
                "label": label,
                "detail": row["detailed-label"],
            }
            if disk_index:
                index.add(key, candidate)
            if key is None:
                counts["unsupported_protocol_rows"] += 1
                continue
            if not disk_index:
                index[key].append(candidate)
    if columns is None or not counts["rows"]:
        raise ValueError("No label records found.")
    if disk_index:
        index.finish()
    return index, dict(counts)


def match_flow(flow, index):
    key = connection_key(
        flow["src_ip"], flow["src_port"], flow["dst_ip"], flow["dst_port"], flow["protocol"]
    )
    if key is None:
        return "unsupported_protocol", None
    start, end = flow["bidirectional_first_seen_ms"], flow["bidirectional_last_seen_ms"]
    if (
        any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
            for x in (start, end)
        )
        or end < start
    ):
        raise ValueError("Invalid exported flow interval.")
    # Decimal preserves publisher timestamps; only one registered millisecond of quantization slack.
    start, end, slack = Decimal(str(start)), Decimal(str(end)), Decimal(1)
    candidates = []
    overlaps = (
        index.overlapping(key, start, end, slack)
        if isinstance(index, SQLiteLabelIndex)
        else index.get(key, [])
    )
    for row in overlaps:
        # An unknown end can span later same-tuple flows; exclude it only when it starts after this flow.
        if row["start"] <= end + slack and (row["end"] is None or row["end"] >= start - slack):
            candidates.append(row)
    if not candidates:
        return "unmatched", None
    if len(candidates) != 1:
        return "ambiguous", None
    row = candidates[0]
    if row["end"] is None or row["label"] is None:
        return "unknown_label_or_duration", None
    if row["start"] > start + slack or row["end"] < end - slack:
        return "partial_interval", None
    return "matched", row


def verified_inputs(capture_dir, config_path):
    capture_dir = Path(capture_dir)
    receipt = json.loads((capture_dir / "receipt.json").read_text(encoding="utf-8"))
    config, capture = configuration(config_path, receipt["capture_id"])
    if receipt["source_id"] != "iot23" or receipt["role"] != "development":
        raise ValueError("Only registered IoT-23 development receipts are accepted.")
    for name in (capture["pcap"], "bro/conn.log.labeled", "README.md"):
        digest = sha256_file(capture_dir / name)
        if digest != receipt["files"][name]["sha256"] or digest != capture["sha256"].get(name):
            raise ValueError("Capture/label bytes differ from the pinned development inputs.")
    return config, capture, receipt


def audit_capture_labels(export_path, capture_dir, output, config_path=CATALOG):
    if Path(output).exists():
        raise FileExistsError(output)
    config, capture, receipt = verified_inputs(capture_dir, config_path)
    if Path(export_path).stat().st_size > 16 * 2**20:
        raise ValueError("Export JSON exceeds the audit byte limit.")
    exported = json.loads(Path(export_path).read_text(encoding="utf-8"))
    if (
        exported["schema"] != SCHEMA
        or exported["features"] != CORE_FEATURES
        or exported["settings"] != SETTINGS
        or exported["exporter_version"] != "6.6.0"
        or exported["pcap_sha256"] != capture["sha256"][capture["pcap"]]
    ):
        raise ValueError("Export contract/settings/input identity mismatch.")
    if not 0 < len(exported["rows"]) < SETTINGS["max_nflows"]:
        raise ValueError("Export must contain a bounded complete set of flows.")
    result = summarize_flows(exported["rows"], capture_dir, config_path, config, receipt)
    result["export_sha256"] = sha256_file(export_path)
    result["export_settings"] = SETTINGS
    write_json(output, result)
    return result


def summarize_flows(flows, capture_dir, config_path, config, receipt, *, parsed_labels=None):
    capture_dir = Path(capture_dir)
    index, log_counts = (
        parsed_labels
        if parsed_labels is not None
        else read_zeek_labels(capture_dir / "bro/conn.log.labeled")
    )
    statuses, classes, details, matched_uids = Counter(), Counter(), Counter(), defaultdict(set)
    first, last, count, identities = math.inf, -math.inf, 0, set()
    for flow in flows:
        count += 1
        if count >= config["max_audit_flows"]:
            raise ValueError(
                "Streaming audit flow limit reached; completeness cannot be established."
            )
        identity = flow_key(flow)
        if identity in identities:
            raise ValueError("Duplicate exported flow identity.")
        identities.add(identity)
        status, label = match_flow(flow, index)
        statuses[status] += 1
        first = min(first, flow["bidirectional_first_seen_ms"])
        last = max(last, flow["bidirectional_last_seen_ms"])
        if label is not None:
            classes[str(label["label"])] += 1
            details[label["detail"]] += 1
            matched_uids[str(label["label"])].add(label["uid"])
    if not count:
        raise ValueError("No supported flows were exported.")
    return {
        "source_id": "iot23",
        "capture_id": receipt["capture_id"],
        "policy": POLICY,
        "config_sha256": sha256_file(config_path),
        "receipt_sha256": sha256_file(capture_dir / "receipt.json"),
        "input_files": receipt["files"],
        "exported_flows": count,
        "flow_status_counts": dict(statuses),
        "matched_flow_class_counts": dict(classes),
        "matched_zeek_connection_class_counts": {
            key: len(value) for key, value in matched_uids.items()
        },
        "connection_support_screen": {
            "minimum_per_class": 25,
            "both_classes_supported": min(len(matched_uids["0"]), len(matched_uids["1"])) >= 25,
            "scope": "distinct capture-scoped Zeek UIDs, not independent events or a training authorization",
        },
        "matched_detailed_labels": dict(details),
        "label_log_counts": log_counts,
        "event_start_utc": datetime.fromtimestamp(first / 1000, UTC).isoformat(),
        "event_end_utc": datetime.fromtimestamp(last / 1000, UTC).isoformat(),
        "training_authorized": False,
        "limitations": [
            "Matching is a conservative candidate linkage, not independent ground-truth verification.",
            "NFStream timeout splits may share one Zeek UID; flow counts are not independent label support.",
            "Unknown/ambiguous/partial/unmatched records are never assigned benign labels.",
            "Device identity and capture-held-out splits are not established by this audit.",
            config["license_note"],
        ],
    }


def audit_pcap_labels(capture_dir, output, config_path=CATALOG):
    """Streaming development audit; rows are never accumulated or written to output."""
    if Path(output).exists():
        raise FileExistsError(output)
    config, capture, receipt = verified_inputs(capture_dir, config_path)
    pcap = Path(capture_dir) / capture["pcap"]
    # The fixture exporter retains its 10k materialization limit. This distinct audit
    # consumes at most 100k flows one by one, while retaining only keys/UIDs/counts.
    with ResourceMonitor(include_children=True) as resources:
        index_path = Path(capture_dir) / f".label-index-{uuid4().hex}.sqlite"
        index = SQLiteLabelIndex(index_path)
        try:
            parsed = read_zeek_labels(
                Path(capture_dir) / "bro/conn.log.labeled",
                index=index,
                max_bytes=config.get("max_label_bytes", 16 * 2**20),
                max_rows=config.get("max_label_rows", 100_000),
            )
            index_bytes = index_path.stat().st_size
            result = summarize_flows(
                iter_pcap_rows(
                    pcap,
                    max_nflows=config["max_audit_flows"],
                    max_pcap_bytes=config.get("max_pcap_bytes", 16 * 2**20),
                ),
                capture_dir,
                config_path,
                config,
                receipt,
                parsed_labels=parsed,
            )
        finally:
            index.close()
            index_path.unlink(missing_ok=True)
    result["export_settings"] = {**SETTINGS, "max_nflows": config["max_audit_flows"]}
    result["label_index"] = {
        "backend": "sqlite",
        "cache_kib": 8192,
        "file_bytes": index_bytes,
        "max_file_bytes": 512 * 2**20,
        "exact_time_unit": "nanoseconds",
    }
    result["schema"] = SCHEMA
    result["exporter_version"] = "6.6.0"
    result["resources"] = {
        "wall_seconds": resources.elapsed,
        "sampled_peak_process_rss_mib": resources.peak / 2**20,
        "sampled_peak_process_tree_rss_sum_mib": resources.peak_tree / 2**20,
        "measurement_scope": "50 ms samples covering label parsing, exporter imports, iteration and matching; child RSS sums may double-count shared pages; not an exact peak or hard OS memory limit",
    }
    write_json(output, result)
    return result

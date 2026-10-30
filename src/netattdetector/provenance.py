"""Inspect available metadata and validate declared capture split plans without fitting."""

import csv
import io
import json
from datetime import datetime
from itertools import pairwise
from pathlib import Path

from netattdetector.data import csv_stream
from netattdetector.io import sha256_file, write_json

METADATA_NAMES = {
    "event_time": {
        "timestamp",
        "ts",
        "starttime",
        "endtime",
        "flowstartmilliseconds",
        "flowendmilliseconds",
        "bidirectionalfirstseenms",
        "bidirectionallastseenms",
        "stime",
        "ltime",
    },
    "capture": {"captureid", "pcapid", "scenarioid", "capture", "scenario"},
    "host": {
        "ipv4srcaddr",
        "ipv4dstaddr",
        "srcip",
        "dstip",
        "sourceip",
        "destinationip",
        "idorigh",
        "idresph",
    },
    "session": {"flowid", "sessionid", "uid"},
}


def inspect_provenance(path, source_id, roles_path, output):
    path, output = Path(path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    roles = json.loads(Path(roles_path).read_text(encoding="utf-8"))
    source = roles["sources"].get(source_id)
    if source is None or path.stem != source_id:
        raise ValueError("Use a registered source ID matching the local archive/CSV name.")
    with csv_stream(path) as stream:
        member = str(getattr(stream, "name", path.name))
        header = next(csv.reader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")))
    header = [column.strip() for column in header]
    if len(header) != len(set(header)):
        raise ValueError("Duplicate normalized header columns.")
    candidates = {
        role: [
            column
            for column in header
            if "".join(x for x in column.casefold() if x.isalnum()) in names
        ]
        for role, names in METADATA_NAMES.items()
    }
    blockers = []
    if not candidates["event_time"]:
        blockers.append("No event timestamp columns; durations are not timestamps.")
    if not candidates["capture"]:
        blockers.append("No row-linked capture/scenario identifier.")
    blockers.append(
        "Header presence alone does not validate metadata values, timezones, label timing or capture/device independence."
    )
    result = {
        "source_id": source_id,
        "role": source["role"],
        "lineage": source["lineage"],
        "publisher": source["publisher"],
        "archive": path.name,
        "sha256": sha256_file(path),
        "member": member,
        "columns": header,
        "candidate_metadata_columns": candidates,
        "capture_time_split_ready": False,
        "blockers": blockers,
        "scope": "header/provenance inspection only; no attack rows, label distributions or model scores inspected",
        "source_roles_sha256": sha256_file(roles_path),
    }
    write_json(output, result)
    return result


def utc_time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Capture timestamps must contain an explicit UTC offset.")
    return parsed.timestamp()


def validate_capture_plan(plan_path, output):
    """Validate declarations; never assert row-label accuracy or authorize model fitting."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1 or plan.get("training_enabled") is not False:
        raise ValueError(
            "Use schema 1 with training explicitly disabled for this preparation phase."
        )
    roles_path = Path(plan["source_roles"])
    roles = json.loads(roles_path.read_text(encoding="utf-8"))["sources"]
    purge, minimum = plan["purge_seconds"], plan["min_rows_per_class"]
    if purge < 0 or minimum < 1 or plan["require_host_disjoint"] is not True:
        raise ValueError("Invalid purge/support/host-isolation policy.")
    captures = plan["captures"]
    if len(captures) > 10_000:
        raise ValueError("Capture manifest exceeds the bounded preparation limit.")
    errors, seen, files, host_splits, intervals, support = [], {}, {}, {}, {}, {}
    order = {name: i for i, name in enumerate(("train", "calibration", "validation", "test"))}
    used_lineages = {}
    for item in captures:
        source_id, capture_id, split = item["source_id"], item["capture_id"], item["split"]
        if source_id not in roles or split not in order or not capture_id:
            raise ValueError("Unknown source, split or empty capture ID.")
        role = roles[source_id]["role"]
        if (split == "test" and role != "reserved_unacquired") or (
            split != "test" and role != "development"
        ):
            errors.append(f"Source role does not permit {source_id} in {split}.")
        identity = (source_id, capture_id)
        if identity in seen:
            errors.append(f"Capture repeated or shared across splits: {identity}.")
        seen[identity] = split
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            errors.append(f"Capture/export input missing or checksum mismatch: {identity}.")
        if item["sha256"] in files:
            errors.append(f"Same input bytes registered multiple times: {identity}.")
        files[item["sha256"]] = identity
        start, end = utc_time(item["start_utc"]), utc_time(item["end_utc"])
        if end < start:
            errors.append(f"Reversed capture interval: {identity}.")
        intervals.setdefault(source_id, {}).setdefault(split, []).append((start, end))
        if not item["host_groups"] or any(
            not isinstance(host, str) or not host for host in item["host_groups"]
        ):
            errors.append(f"Missing declared host/device groups: {identity}.")
        for host in item["host_groups"]:
            host_splits.setdefault(host, set()).add(split)
        for label, field in ((0, "benign_rows"), (1, "attack_rows")):
            count = item[field]
            if not isinstance(count, int) or count < 0:
                raise ValueError("Declared class support must be nonnegative integers.")
            support.setdefault(split, {0: 0, 1: 0})[label] += count
        for lineage in roles[source_id]["lineage"]:
            used_lineages.setdefault(lineage, set()).add(split == "test")
    for host, splits in host_splits.items():
        if len(splits) > 1:
            errors.append(f"Host/device group crosses partitions: {host}.")
    for lineage, uses in used_lineages.items():
        if len(uses) > 1:
            errors.append(f"Reserved source lineage appears in development: {lineage}.")
    for source_id, parts in intervals.items():
        available = sorted(parts, key=order.get)
        for left, right in pairwise(available):
            if max(end for _, end in parts[left]) + purge > min(start for start, _ in parts[right]):
                errors.append(f"Chronology/purge failure for {source_id}: {left} -> {right}.")
    for split in order:
        if split not in support or min(support[split].values()) < minimum:
            errors.append(
                f"Insufficient declared benign/attack support in {split} (minimum {minimum}/class)."
            )
    result = {
        "schema_version": 1,
        "plan_sha256": sha256_file(plan_path),
        "source_roles_sha256": sha256_file(roles_path),
        "capture_count": len(captures),
        "structure_valid": not errors,
        "training_authorized": False,
        "declared_support": support,
        "blockers": errors,
        "limitation": "Validates declared metadata/checksums only. Does not prove label counts, endpoint identities, time semantics or row-to-capture linkage. These require raw-data/export audit before fitting.",
    }
    write_json(output, result)
    return result

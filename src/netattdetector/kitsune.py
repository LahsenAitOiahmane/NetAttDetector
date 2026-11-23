"""Bounded development acquisition and independent packet/label cardinality audit."""

import csv
import hashlib
import json
import os
import shutil
import stat
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit

from netattdetector.io import sha256_file, write_json
from netattdetector.packet_reader import packet_records
from netattdetector.resources import ResourceMonitor

CONFIG = "configs/data/kitsune-development.json"
API = "https://www.kaggle.com/api/v1/datasets/download/ymirsky/network-attack-dataset-kitsune/"
FILES = [
    ("pcap", "Mirai Botnet/Mirai_pcap.pcap", 75075844, 96 * 2**20),
    ("labels", "Mirai Botnet/mirai_labels.csv", 2292409, 4 * 2**20),
]


def configuration(path=CONFIG):
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = [
        {
            "id": identity,
            "name": name,
            "expected_uncompressed_bytes": size,
            "maximum_transfer_bytes": limit,
        }
        for identity, name, size, limit in FILES
    ]
    if (
        policy.get("source_id") != "kitsune"
        or policy.get("profile_id") != "kitsune-mirai-development-v1"
        or policy.get("role") != "development"
        or policy.get("files") != expected
        or policy.get("maximum_total_uncompressed_bytes") != 96 * 2**20
        or policy.get("disk_reserve_bytes") != 2 * 2**30
        or policy.get("maximum_label_rows") != 1_000_000
        or policy.get("training_authorized") is not False
        or policy.get("deployment_authorized") is not False
    ):
        raise ValueError("Use the registered training-disabled Kitsune development profile.")
    return policy


class DownloadRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https" or urlsplit(newurl).hostname not in {
            "www.kaggle.com",
            "kaggle.com",
            "storage.googleapis.com",
        }:
            raise ValueError("Kitsune download redirected off the registered HTTPS storage hosts.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def publish_download(download, target, specification):
    """Validate a raw file or exactly one CRC-checked ZIP member; never extract paths."""
    download, target = Path(download), Path(target)
    if target.exists():
        raise FileExistsError(target)
    temporary = target.with_name(target.name + ".expanded.partial")
    created = False
    try:
        if zipfile.is_zipfile(download):
            with zipfile.ZipFile(download) as archive:
                members = archive.infolist()
                if len(members) != 1:
                    raise ValueError("Kitsune file response contains unexpected ZIP members.")
                member = members[0]
                mode = stat.S_IFMT(member.external_attr >> 16)
                if (
                    member.filename
                    not in {specification["name"], PurePosixPath(specification["name"]).name}
                    or member.is_dir()
                    or mode not in (0, stat.S_IFREG)
                    or member.file_size != specification["expected_uncompressed_bytes"]
                ):
                    raise ValueError(
                        "Kitsune ZIP member identity/type/size differs from registration."
                    )
                count = 0
                with archive.open(member) as source, temporary.open("xb") as output:
                    created = True
                    while chunk := source.read(65536):
                        count += len(chunk)
                        if count > specification["expected_uncompressed_bytes"]:
                            raise ValueError("Kitsune ZIP expansion exceeds the registered bound.")
                        output.write(chunk)
                if count != specification["expected_uncompressed_bytes"]:
                    raise ValueError("Truncated Kitsune ZIP member.")
            source_path, container = temporary, "single-member-zip-crc-checked"
        else:
            if download.stat().st_size != specification["expected_uncompressed_bytes"]:
                raise ValueError("Kitsune raw byte count differs from publisher file metadata.")
            source_path, container = download, "raw"
        os.link(source_path, target)
        return {
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
            "container": container,
            "publisher_checksum_verified": False,
        }
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def fetch_file(specification, output):
    output = Path(output)
    target = output / PurePosixPath(specification["name"]).name
    if target.exists():
        raise FileExistsError(target)
    temporary = output / (target.name + ".download.partial")
    created = False
    deadline = time.monotonic() + 600
    limit = specification["maximum_transfer_bytes"]
    request = urllib.request.Request(
        API + quote(specification["name"], safe="") + "?raw=true",
        headers={"User-Agent": "NetAttDetector-research/0.3", "Accept-Encoding": "identity"},
    )
    try:
        with urllib.request.build_opener(DownloadRedirect()).open(request, timeout=45) as response:
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
            length = response.headers.get("Content-Length")
            if (
                response.status != 200
                or content_type
                not in {
                    "application/octet-stream",
                    "application/zip",
                    "text/csv",
                    "application/vnd.tcpdump.pcap",
                }
                or (length is not None and (not length.isdigit() or not 0 < int(length) <= limit))
            ):
                raise ValueError("Kitsune did not return a complete bounded binary/CSV file.")
            count, digest = 0, hashlib.sha256()
            with temporary.open("xb") as stream:
                created = True
                read_chunk = getattr(response, "read1", response.read)
                while True:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Kitsune file transfer exceeded its deadline.")
                    chunk = read_chunk(min(65536, limit + 1 - count))
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > limit:
                        raise ValueError("Kitsune file response exceeds its transfer bound.")
                    digest.update(chunk)
                    stream.write(chunk)
            if not count or (length is not None and count != int(length)):
                raise ValueError("Kitsune file transfer is empty or truncated.")
        receipt = publish_download(temporary, target, specification)
        return {
            **receipt,
            "id": specification["id"],
            "source_name": specification["name"],
            "local_name": target.name,
            "transfer_bytes": count,
            "transfer_sha256": digest.hexdigest(),
        }
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def acquire(output, config_path=CONFIG):
    policy = configuration(config_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reserve = sum(
        row["maximum_transfer_bytes"] + row["expected_uncompressed_bytes"]
        for row in policy["files"]
    )
    if shutil.disk_usage(output).free - reserve < policy["disk_reserve_bytes"]:
        raise ValueError("Kitsune acquisition would breach the 2 GiB disk reserve.")
    records = []
    with ResourceMonitor() as resources:
        for specification in policy["files"]:
            records.append(fetch_file(specification, output))
            write_json(output / "progress.json", {"files": records, "training_authorized": False})
    if sum(row["bytes"] for row in records) > policy["maximum_total_uncompressed_bytes"]:
        raise ValueError("Kitsune total expanded inputs exceed their registered limit.")
    receipt = {
        "profile_id": policy["profile_id"],
        "source_role": "development",
        "config_sha256": sha256_file(config_path),
        "files": records,
        "resources": {"seconds": resources.elapsed, "sampled_rss_mib": resources.peak / 2**20},
        "publisher_checksum_verified": False,
        "training_authorized": False,
        "deployment_authorized": False,
        "scope": "Complete bounded author-mirror originals. Observed hashes, not publisher-authenticated checksums. No packet-feature CSV or whole collection.",
    }
    write_json(output / "acquisition.json", receipt)
    return receipt


def verified_files(directory, config_path=CONFIG):
    directory = Path(directory)
    policy = configuration(config_path)
    receipt = json.loads((directory / "acquisition.json").read_text(encoding="utf-8"))
    if (
        receipt["config_sha256"] != sha256_file(config_path)
        or receipt["profile_id"] != policy["profile_id"]
        or receipt["source_role"] != "development"
        or len(receipt["files"]) != 2
    ):
        raise ValueError("Kitsune acquisition differs from the registered profile.")
    paths = {}
    for specification in policy["files"]:
        record = next(row for row in receipt["files"] if row["id"] == specification["id"])
        name = PurePosixPath(specification["name"]).name
        path = directory / name
        if (
            record["local_name"] != name
            or record["source_name"] != specification["name"]
            or record["bytes"] != specification["expected_uncompressed_bytes"]
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError("Kitsune input bytes or source identity changed after acquisition.")
        paths[specification["id"]] = path
    return policy, receipt, paths


def binary_labels(path, *, max_rows=1_000_000):
    """One column, literal0/1 only; numeric first rows are data, never guessed headers."""
    count = 0
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for number, row in enumerate(csv.reader(stream), 1):
            if number == 1 and row in (["label"], ["Label"]):
                continue
            if len(row) != 1 or row[0] not in ("0", "1"):
                raise ValueError(
                    "Kitsune labels require a single literal binary column; no guessed numeric header/index."
                )
            count += 1
            if count > max_rows:
                raise ValueError("Kitsune label row bound exceeded.")
            yield int(row[0])
    if count == 0:
        raise ValueError("Kitsune labels are empty.")


def audit_pairs(pcap, labels, *, max_rows=1_000_000):
    counts, statuses = Counter(), defaultdict(Counter)
    count = regressions = transitions = 0
    first_attack = last_benign = previous_label = previous_time = None
    time_ranges = {}
    missing = object()
    for packet, label in zip_longest(
        packet_records(pcap, max_packets=max_rows),
        binary_labels(labels, max_rows=max_rows),
        fillvalue=missing,
    ):
        if packet is missing or label is missing:
            raise ValueError("Kitsune packet/label cardinality differs; refuse ordinal alignment.")
        count += 1
        counts[str(label)] += 1
        statuses[packet["status"]][str(label)] += 1
        timestamp = packet["time_ns"]
        regressions += int(previous_time is not None and timestamp < previous_time)
        transitions += int(previous_label is not None and label != previous_label)
        if label == 1 and first_attack is None:
            first_attack = count
        if label == 0:
            last_benign = count
        name = str(label)
        if name not in time_ranges:
            time_ranges[name] = {"minimum_ns": timestamp, "maximum_ns": timestamp}
        else:
            time_ranges[name]["minimum_ns"] = min(time_ranges[name]["minimum_ns"], timestamp)
            time_ranges[name]["maximum_ns"] = max(time_ranges[name]["maximum_ns"], timestamp)
        previous_time, previous_label = timestamp, label
    return {
        "packets": count,
        "labels": count,
        "binary_class_counts": dict(counts),
        "class_counts_by_packet_status": {
            name: dict(value) for name, value in sorted(statuses.items())
        },
        "timestamp_regressions": regressions,
        "label_transitions": transitions,
        "first_attack_packet_ordinal": first_attack,
        "last_benign_packet_ordinal": last_benign,
        "single_benign_prefix_then_attack_suffix": first_attack is not None
        and last_benign is not None
        and last_benign < first_attack,
        "time_ranges_by_label": time_ranges,
        "packet_label_cardinality_verified": True,
        "native_flow_labels_authorized": False,
        "scope": "Structural ordinal alignment under publisher one-label-per-packet documentation; not independent validation of attack truth, checksums/payloads or native flow mapping. Truncated/unsupported packet labels are retained in coverage counts.",
    }


def audit(directory, output, config_path=CONFIG):
    policy, receipt, paths = verified_files(directory, config_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    with ResourceMonitor() as resources:
        result = audit_pairs(paths["pcap"], paths["labels"], max_rows=policy["maximum_label_rows"])
    result.update(
        {
            "profile_id": policy["profile_id"],
            "config_sha256": sha256_file(config_path),
            "acquisition_sha256": sha256_file(Path(directory) / "acquisition.json"),
            "input_files": receipt["files"],
            "resources": {"seconds": resources.elapsed, "sampled_rss_mib": resources.peak / 2**20},
            "training_authorized": False,
            "deployment_authorized": False,
        }
    )
    write_json(output, result)
    return result

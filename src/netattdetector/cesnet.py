"""Bounded CESNET-CC25 capture acquisition and frozen positive-only evaluation.

No fitting, source-derived thresholds, negative labels or endpoint model features.
"""

import hashlib
import ipaddress
import json
import os
import shutil
import sqlite3
import tarfile
import time
import urllib.request
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from netattdetector.causal_context import SCHEMA
from netattdetector.context_cohort import FEATURES
from netattdetector.io import sha256_file, write_json
from netattdetector.resources import ResourceMonitor
from netattdetector.statistical import FLOAT_FIELDS, STAT_SETTINGS
from netattdetector.statistical_cohort import configure_database, connect_readonly

CONFIG = "configs/data/cesnet-frozen-test.json"
MODEL_SHA256 = "a504167cf9dbd384c17b192d2f1209714c8f4ef5c27e6b892b64fcbe04e19cc7"
THRESHOLD = 0.9999190519374116
RESERVE = 2 * 2**30
MAX_DATABASE = 512 * 2**20
COLUMNS = ["ordinal", "capture_id", "eligible", *FEATURES]


def config(path=CONFIG):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        value.get("study_id") != "cesnet-cc25-frozen-recall-v1"
        or value.get("expected_model_sha256") != MODEL_SHA256
        or value.get("threshold") != THRESHOLD
        or value.get("feature_contract") != SCHEMA
        or value.get("model_policy") != "statistics27_control"
        or value.get("prediction_threads") != 1
        or value.get("batch_size") != 1024
        or value.get("training_authorized") is not False
        or value.get("deployment_authorized") is not False
        or value.get("maximum_archive_bytes") != 64 * 2**20
        or value.get("maximum_total_uncompressed_bytes") != 512 * 2**20
        or value.get("maximum_capture_bytes") != 128 * 2**20
        or value.get("maximum_flows_per_capture") != 250_000
        or value.get("maximum_context_events") != 100_000
        or value.get("bootstrap")
        != {"replicates": 10000, "seed": 20261123, "unit": "capture_macro_recall"}
        or value.get("archive")
        != {
            "url": "https://zenodo.org/api/records/16752462/files/botnet.tar.gz/content",
            "bytes": 43421271,
            "publisher_md5": "f66a6410bc1e35881f2de2077788847f",
        }
        or len(value.get("captures", [])) != 32
        or len({row["capture_id"] for row in value["captures"]}) != 32
        or hashlib.sha256(
            json.dumps(value["captures"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        != "13d891902f788b3311b6804d7313c84e480d2876dbcf08e915ce5b52c5ffe668"
    ):
        raise ValueError("Use the registered frozen CESNET recall policy.")
    for row in value["captures"]:
        ipaddress.ip_address(row["cc_ip"])
        if type(row["cc_port"]) is not int or not 0 < row["cc_port"] <= 65535:
            raise ValueError("Invalid registered C&C endpoint.")
    return value


def reserve_space(directory, additional):
    if shutil.disk_usage(directory).free - additional < RESERVE:
        raise ValueError("CESNET work would breach the 2 GiB disk reserve.")


def archive_identity(path, policy):
    path = Path(path)
    if path.stat().st_size != policy["archive"]["bytes"]:
        raise ValueError("CESNET archive byte count differs from registration.")
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != policy["archive"]["publisher_md5"]:
        raise ValueError("CESNET publisher archive checksum mismatch.")
    return {
        "bytes": path.stat().st_size,
        "publisher_md5_verified": True,
        "sha256": sha256_file(path),
    }


def download_archive(output, policy):
    """Exclusive complete download, at most64MiB; never print redirect URLs/content."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "botnet.tar.gz"
    if target.exists():
        return archive_identity(target, policy)
    reserve_space(output, policy["maximum_archive_bytes"])

    class Redirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            url = urlsplit(newurl)
            if url.scheme != "https" or url.hostname != "zenodo.org":
                raise ValueError("CESNET redirected off the registered publisher host.")
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    partial = output / "botnet.tar.gz.partial"
    created = False
    deadline = time.monotonic() + 600
    try:
        request = urllib.request.Request(
            policy["archive"]["url"], headers={"User-Agent": "NetAttDetector-research/0.3"}
        )
        with urllib.request.build_opener(Redirect()).open(request, timeout=45) as response:
            if response.status != 200:
                raise ValueError("Expected the complete CESNET archive.")
            count = 0
            with partial.open("xb") as stream:
                created = True
                while True:
                    if time.monotonic() > deadline:
                        raise TimeoutError("CESNET archive transfer deadline exceeded.")
                    chunk = response.read(min(65536, policy["maximum_archive_bytes"] + 1 - count))
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > policy["maximum_archive_bytes"]:
                        raise ValueError("CESNET transfer exceeds its byte bound.")
                    stream.write(chunk)
        receipt = archive_identity(partial, policy)
        os.link(partial, target)
        return receipt
    finally:
        if created:
            partial.unlink(missing_ok=True)


def inventory_archive(path, policy):
    """Inspect headers only; reject unsafe types/paths and overlapping split copies."""
    samples = {row["capture_id"] for row in policy["captures"]}
    selected, names, count, total = {}, set(), 0, 0
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            count += 1
            name = PurePosixPath(member.name)
            if (
                count > 512
                or member.name in names
                or name.is_absolute()
                or ".." in name.parts
                or "\\" in member.name
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError("Unsafe, duplicate or oversized CESNET archive inventory.")
            names.add(member.name)
            if member.isfile():
                total += member.size
                if member.size < 0 or total > policy["maximum_total_uncompressed_bytes"]:
                    raise ValueError("CESNET archive expansion exceeds its bound.")
                if (
                    member.name.endswith(".pcapng")
                    and not name.name.startswith("._")
                    and not member.name.endswith(
                        ("-train.pcapng", "-test.pcapng", "_train.pcapng", "_test.pcapng")
                    )
                ):
                    matches = [part for part in name.parts[:-1] if part in samples]
                    if (
                        len(matches) != 1
                        or matches[0] in selected
                        or not 0 < member.size <= policy["maximum_capture_bytes"]
                    ):
                        raise ValueError("Ambiguous or oversized original CESNET capture.")
                    selected[matches[0]] = {"member": member.name, "bytes": member.size}
    if set(selected) != samples:
        raise ValueError("Original CESNET capture inventory is incomplete.")
    return {"members": count, "declared_uncompressed_bytes": total, "selected": selected}


def acquire(output, config_path=CONFIG):
    policy = config(config_path)
    output = Path(output)
    receipt = download_archive(output, policy)
    inventory = inventory_archive(output / "botnet.tar.gz", policy)
    manifest = {
        "study_id": policy["study_id"],
        "config_sha256": sha256_file(config_path),
        "archive": receipt,
        "inventory": inventory,
        "training_authorized": False,
        "scoring_started": False,
    }
    path = output / "acquisition.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("CESNET acquisition inventory changed.")
    else:
        write_json(path, manifest)
    return manifest


def extract_originals(source, output, acquisition, policy):
    """Stream only exact frozen regular members into generated fresh basenames."""
    source, output = Path(source), Path(output)
    if (
        archive_identity(source / "botnet.tar.gz", policy) != acquisition["archive"]
        or inventory_archive(source / "botnet.tar.gz", policy) != acquisition["inventory"]
    ):
        raise ValueError("CESNET archive/inventory differs from acquisition freeze.")
    output.mkdir(parents=True, exist_ok=False)
    selected = acquisition["inventory"]["selected"]
    reserve_space(output, sum(row["bytes"] for row in selected.values()) + MAX_DATABASE)
    wanted = {row["member"]: capture for capture, row in selected.items()}
    records = {}
    with tarfile.open(source / "botnet.tar.gz", "r|gz") as archive:
        for member in archive:
            if member.name not in wanted:
                continue
            capture = wanted[member.name]
            target = output / f"{capture}.pcapng"
            digest, count = hashlib.sha256(), 0
            with archive.extractfile(member) as original, target.open("xb") as stream:
                while chunk := original.read(65536):
                    count += len(chunk)
                    if count > selected[capture]["bytes"]:
                        raise ValueError("CESNET member expansion exceeds frozen size.")
                    digest.update(chunk)
                    stream.write(chunk)
            if count != selected[capture]["bytes"]:
                raise ValueError("Truncated CESNET original member.")
            records[capture] = {"bytes": count, "sha256": digest.hexdigest()}
    if set(records) != set(selected):
        raise ValueError("CESNET original capture extraction is incomplete.")
    return records


def endpoint_eligible(row, sample):
    return (row["src_ip"] == sample["cc_ip"] and row["src_port"] == sample["cc_port"]) or (
        row["dst_ip"] == sample["cc_ip"] and row["dst_port"] == sample["cc_port"]
    )


def row_digest(digest, values):
    digest.update(json.dumps(values, separators=(",", ":"), allow_nan=False).encode() + b"\n")


def prepare(source, output, config_path=CONFIG):
    from netattdetector.context_meter import iter_context_rows, verify_context_fixture

    policy = config(config_path)
    source, output = Path(source), Path(output)
    acquisition = json.loads((source / "acquisition.json").read_text(encoding="utf-8"))
    if acquisition["config_sha256"] != sha256_file(config_path):
        raise ValueError("CESNET acquisition used a different registered configuration.")
    output.mkdir(parents=True, exist_ok=False)
    verify_context_fixture(output / "fixture.json")
    files = extract_originals(source, output / "captures", acquisition, policy)
    path = output / "flows.sqlite"
    with path.open("xb"):
        pass
    db = sqlite3.connect(path)
    configure_database(db)
    db.execute("PRAGMA max_page_count=131072")
    db.execute(
        "CREATE TABLE flows("
        + ",".join(
            "ordinal INTEGER PRIMARY KEY"
            if name == "ordinal"
            else f"{name} TEXT NOT NULL"
            if name == "capture_id"
            else f"{name} REAL NOT NULL"
            if name in FLOAT_FIELDS
            else f"{name} INTEGER NOT NULL"
            for name in COLUMNS
        )
        + ")"
    )
    records, ordinal, logical = [], 0, hashlib.sha256()
    try:
        for sample in policy["captures"]:
            capture = sample["capture_id"]
            rows = eligible = packets = eligible_packets = warmup = late = peak = 0
            with ResourceMonitor(include_children=True) as resources:
                for row in iter_context_rows(
                    output / "captures" / f"{capture}.pcapng",
                    max_nflows=policy["maximum_flows_per_capture"],
                    max_pcap_bytes=policy["maximum_capture_bytes"],
                    max_events=policy["maximum_context_events"],
                ):
                    ordinal += 1
                    rows += 1
                    positive = int(endpoint_eligible(row, sample))
                    eligible += positive
                    packets += row["bidirectional_packets"]
                    eligible_packets += positive * row["bidirectional_packets"]
                    warmup += 1 - row["context_warmup_complete"]
                    late += row["context_late_start"]
                    peak = max(peak, row["context_peak_events"])
                    values = [ordinal, capture, positive, *[row[name] for name in FEATURES]]
                    row_digest(logical, values)
                    db.execute(
                        "INSERT INTO flows VALUES(" + ",".join("?" for _ in COLUMNS) + ")", values
                    )
                    if rows % 10000 == 0:
                        db.commit()
                db.commit()
            records.append(
                {
                    "capture_id": capture,
                    "family": sample["family"],
                    "exported_flows": rows,
                    "eligible_positive_flows": eligible,
                    "unknown_flows": rows - eligible,
                    "exported_packets": packets,
                    "eligible_packets": eligible_packets,
                    "warmup_flows": warmup,
                    "late_flows": late,
                    "peak_context_events": peak,
                    "file": files[capture],
                    "publisher_endpoint_annotated": sample["publisher_endpoint_annotation"]
                    is not None,
                    "resources": {
                        "seconds": resources.elapsed,
                        "sampled_tree_rss_mib": resources.peak_tree / 2**20,
                    },
                }
            )
            write_json(
                output / "progress.json",
                {"captures": records, "model_fits": 0, "model_scored": False},
            )
        db.execute("CREATE INDEX by_capture ON flows(capture_id,ordinal)")
        db.commit()
    finally:
        db.close()
    manifest = {
        "study_id": policy["study_id"],
        "config_sha256": sha256_file(config_path),
        "acquisition_sha256": sha256_file(source / "acquisition.json"),
        "schema": SCHEMA,
        "features": FEATURES,
        "exporter_version": "6.6.0",
        "settings": {**STAT_SETTINGS, "max_nflows": policy["maximum_flows_per_capture"]},
        "records": records,
        "rows": ordinal,
        "logical_sha256": logical.hexdigest(),
        "database": {"bytes": path.stat().st_size, "sha256": sha256_file(path)},
        "model_fits": 0,
        "model_scored": False,
        "deployment_authorized": False,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def verify_database(directory, policy, config_path=CONFIG):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    path = directory / "flows.sqlite"
    if (
        manifest["config_sha256"] != sha256_file(config_path)
        or manifest["schema"] != SCHEMA
        or manifest["features"] != FEATURES
        or manifest["exporter_version"] != "6.6.0"
        or manifest["settings"]
        != {**STAT_SETTINGS, "max_nflows": policy["maximum_flows_per_capture"]}
        or manifest["database"] != {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        or path.stat().st_size > MAX_DATABASE
    ):
        raise ValueError("CESNET measurement contract or database bytes changed.")
    db = connect_readonly(path)
    try:
        configure_database(db)
        if (
            db.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
            or [row[1] for row in db.execute("PRAGMA table_info(flows)")] != COLUMNS
        ):
            raise ValueError("CESNET database schema/integrity failed.")
        logical, rows = hashlib.sha256(), 0
        for row in db.execute("SELECT * FROM flows ORDER BY ordinal"):
            rows += 1
            if row[0] != rows or row[2] not in (0, 1):
                raise ValueError("CESNET row sequence/eligibility changed.")
            row_digest(logical, list(row))
        if rows != manifest["rows"] or logical.hexdigest() != manifest["logical_sha256"]:
            raise ValueError("CESNET logical row digest differs from preparation.")
        actual = {
            row[0]: row[1:]
            for row in db.execute(
                "SELECT capture_id,COUNT(*),SUM(eligible) FROM flows GROUP BY capture_id"
            )
        }
        samples = {row["capture_id"] for row in policy["captures"]}
        if {row["capture_id"] for row in manifest["records"]} != samples or set(actual) - samples:
            raise ValueError("CESNET sample coverage changed.")
        for row in manifest["records"]:
            if actual.get(row["capture_id"], (0, 0)) != (
                row["exported_flows"],
                row["eligible_positive_flows"],
            ):
                raise ValueError("CESNET capture counts differ from their retained audit.")
    finally:
        db.close()
    return manifest


def recall_summary(records, *, replicates=10000, seed=20261123):
    import numpy as np

    eligible = sum(row["eligible_positive_flows"] for row in records)
    detected = sum(row["detected_positive_flows"] for row in records)
    macro = [
        row["detected_positive_flows"] / row["eligible_positive_flows"]
        for row in records
        if row["eligible_positive_flows"]
    ]
    if macro:
        rng = np.random.default_rng(seed)
        values = np.asarray(macro)
        bootstrap = values[rng.integers(0, len(values), (replicates, len(values)))].mean(axis=1)
        interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    else:
        interval = None
    return {
        "captures": len(records),
        "assessed_captures": len(macro),
        "unassessed_captures": len(records) - len(macro),
        "eligible_positive_flows": eligible,
        "detected_positive_flows": detected,
        "pooled_positive_flow_recall": detected / eligible if eligible else None,
        "capture_macro_recall": float(np.mean(macro)) if macro else None,
        "capture_bootstrap_95": interval,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "uncertainty_limitation": "Percentile resampling of observed captures with eligible endpoint-matched flows. Captures can share devices/infrastructure/families; not independent-attack or population uncertainty.",
        "precision": None,
        "false_positive_rate": None,
        "accuracy": None,
        "balanced_accuracy": None,
        "reason": "No compatible labeled benign packet captures. Native flow segments are not Zeek UID connections.",
    }


def evaluate(measurements, model_path, output, config_path=CONFIG):
    import importlib.metadata
    import platform
    import subprocess

    import joblib
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits

    from netattdetector.context_study import ContextFeatures

    policy = config(config_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    freeze = json.loads(Path(policy["freeze"]).read_text(encoding="utf-8"))
    if (
        freeze["artifact_sha256"] != MODEL_SHA256
        or freeze["threshold"] != THRESHOLD
        or sha256_file(model_path) != MODEL_SHA256
    ):
        raise ValueError("CESNET model differs from the independently committed freeze.")
    manifest = verify_database(measurements, policy, config_path)
    # Hash and contract checks precede any model pickle load.
    bundle = joblib.load(model_path)
    if (
        bundle["format"] != "context-diagnostic-v1"
        or bundle["features"] != FEATURES
        or bundle["schema"] != SCHEMA
        or bundle["threshold"] != THRESHOLD
        or bundle["specification"] != freeze["specification"]
        or bundle["model"].named_steps["features"].policy != policy["model_policy"]
        or bundle["deployment_authorized"] is not False
    ):
        raise ValueError("Frozen model contract differs from the registered external test.")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    with ResourceMonitor(include_children=True) as resources:
        db = connect_readonly(Path(measurements) / "flows.sqlite")
        try:
            configure_database(db)
            for record in manifest["records"]:
                capture = record["capture_id"]
                detected = eligible = unknown = 0
                cursor = db.execute(
                    "SELECT eligible,"
                    + ",".join(FEATURES)
                    + " FROM flows WHERE capture_id=? ORDER BY ordinal",
                    (capture,),
                )
                with threadpool_limits(limits=1):
                    while rows := cursor.fetchmany(policy["batch_size"]):
                        frame = pd.DataFrame([row[1:] for row in rows], columns=FEATURES)
                        ContextFeatures().transform(frame)
                        scores = bundle["model"].predict_proba(frame)[:, 1]
                        if (
                            not np.isfinite(scores).all()
                            or (scores < 0).any()
                            or (scores > 1).any()
                        ):
                            raise ValueError("Invalid frozen model scores.")
                        positive = np.asarray([row[0] for row in rows], dtype=bool)
                        eligible += int(positive.sum())
                        unknown += int((~positive).sum())
                        detected += int(np.count_nonzero(positive & (scores >= THRESHOLD)))
                if (
                    eligible != record["eligible_positive_flows"]
                    or eligible + unknown != record["exported_flows"]
                ):
                    raise ValueError("CESNET scoring did not cover every measured row.")
                records.append(
                    {
                        **record,
                        "detected_positive_flows": detected,
                        "missed_positive_flows": eligible - detected,
                        "positive_flow_recall": detected / eligible if eligible else None,
                    }
                )
                write_json(
                    output / "progress.json",
                    {"records": records, "model_fits": 0, "threshold": THRESHOLD},
                )
        finally:
            db.close()
    summary = recall_summary(
        records, **{key: policy["bootstrap"][key] for key in ("replicates", "seed")}
    )
    families = {
        family: recall_summary(
            [row for row in records if row["family"] == family],
            **{key: policy["bootstrap"][key] for key in ("replicates", "seed")},
        )
        for family in sorted({row["family"] for row in records})
    }
    result = {
        "study_id": policy["study_id"],
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config_sha256": sha256_file(config_path),
        "uv_lock_sha256": sha256_file("uv.lock"),
        "model_sha256": MODEL_SHA256,
        "threshold": THRESHOLD,
        "measurements_manifest_sha256": sha256_file(Path(measurements) / "manifest.json"),
        "measurement_logical_sha256": manifest["logical_sha256"],
        "summary": summary,
        "families": families,
        "records": records,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "joblib")
        },
        "platform": platform.system(),
        "resources": {
            "seconds": resources.elapsed,
            "sampled_tree_rss_mib": resources.peak_tree / 2**20,
            "scope": "Bounded scoring and count reconciliation only; excludes imports, input/hash verification, model load and bootstrap. Sampled RSS is not an OS-enforced limit.",
        },
        "source_now_observed": True,
        "model_fits": 0,
        "threshold_changed": False,
        "deployment_authorized": False,
        "development_research_screen_passed": freeze["passes_development_research_screen"],
        "limitations": "Positive-only endpoint-matched native segment recall. Unknown endpoint flows are retained and never called benign. No precision/FPR or production acceptance. Independent acquisition source does not establish device/infrastructure independence.",
    }
    write_json(output / "evaluation.json", result)
    return result


def seal_measurements(
    directory, output, recipient="configs/security/model-recipient.json", config_path=CONFIG
):
    """Only recipient-encrypted numeric measurements leave the private Linux runner."""
    import io
    import zipfile

    from netattdetector.private_artifacts import encode, seal_payload

    policy = config(config_path)
    manifest = verify_database(directory, policy, config_path)
    compressed = io.BytesIO()
    with zipfile.ZipFile(
        compressed, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        archive.write(Path(directory) / "flows.sqlite", "flows.sqlite")
    if compressed.tell() > 64 * 2**20:
        raise ValueError("Encrypted measurement archive exceeds its compressed byte bound.")
    return seal_payload(
        {
            "format": "cesnet-numeric-transfer-v1",
            "manifest": manifest,
            "sqlite_zip": encode(compressed.getvalue()),
        },
        recipient,
        output,
        max_payload=96 * 2**20,
    )


def recover_measurements(envelope, private_key, trusted_manifest, output):
    import io
    import zipfile

    from netattdetector.private_artifacts import decode, protect_directory, recover_payload

    manifest = json.loads(Path(trusted_manifest).read_text(encoding="utf-8"))
    if not 0 < manifest["database"]["bytes"] <= MAX_DATABASE:
        raise ValueError("Trusted measurement database exceeds its bound.")
    payload = recover_payload(envelope, private_key, max_payload=96 * 2**20)
    if (
        set(payload) != {"format", "manifest", "sqlite_zip"}
        or payload["format"] != "cesnet-numeric-transfer-v1"
        or payload["manifest"] != manifest
    ):
        raise ValueError(
            "Recovered measurement metadata differs from the separately trusted manifest."
        )
    raw = decode(payload["sqlite_zip"])
    if len(raw) > 64 * 2**20:
        raise ValueError("Compressed measurement payload exceeds its bound.")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        if (
            len(entries) != 1
            or entries[0].filename != "flows.sqlite"
            or entries[0].file_size != manifest["database"]["bytes"]
        ):
            raise ValueError("Measurement payload does not contain the exact frozen SQLite member.")
        output = protect_directory(output)
        reserve_space(output, entries[0].file_size)
        digest, count = hashlib.sha256(), 0
        with (
            archive.open(entries[0]) as source,
            (output / "flows.sqlite").open("xb") as destination,
        ):
            while chunk := source.read(65536):
                count += len(chunk)
                if count > manifest["database"]["bytes"]:
                    raise ValueError("Measurement decompression exceeds the frozen bound.")
                digest.update(chunk)
                destination.write(chunk)
        if (
            count != manifest["database"]["bytes"]
            or digest.hexdigest() != manifest["database"]["sha256"]
        ):
            raise ValueError("Recovered database differs from the separately trusted hash.")
    write_json(output / "manifest.json", manifest)
    return {
        "database_sha256": digest.hexdigest(),
        "rows": manifest["rows"],
        "verified_before_sqlite_open": True,
    }

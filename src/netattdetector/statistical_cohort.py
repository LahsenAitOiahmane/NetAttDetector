"""Private statistical sidecars bound to an unchanged, verified native core cohort."""

import hashlib
import json
import shutil
import sqlite3
import struct
import subprocess
from pathlib import Path

from netattdetector.capture_labels import verified_inputs
from netattdetector.io import sha256_file, write_json
from netattdetector.measurement import CORE_FEATURES, flow_key
from netattdetector.native_cohort import SPLITS, digest, load_config, verify_native_cohort
from netattdetector.resources import ResourceMonitor
from netattdetector.statistical import (
    EXTRA_FEATURES,
    FEATURES,
    FLOAT_FIELDS,
    SCHEMA,
    STAT_SETTINGS,
    iter_statistical_rows,
    validate_statistics,
)

CONFIG = "configs/data/statistical-features.json"
COLUMNS = ["ordinal", "flow_hash", *EXTRA_FEATURES, "full_hash", "no_ports_hash"]


def pattern_hash(values):
    return hashlib.sha256(struct.pack(f"<{len(values)}d", *values)).hexdigest()


def connect_readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def configure_database(db):
    db.execute("PRAGMA cache_size=-8192")
    db.execute("PRAGMA temp_store=FILE")


def create_sidecar(path):
    with Path(path).open("xb"):
        pass
    db = sqlite3.connect(path)
    configure_database(db)
    db.execute("PRAGMA page_size=4096")
    db.execute("PRAGMA max_page_count=131072")
    types = {
        "ordinal": "INTEGER PRIMARY KEY",
        "flow_hash": "TEXT UNIQUE",
        "full_hash": "TEXT",
        "no_ports_hash": "TEXT",
    }
    db.execute(
        "CREATE TABLE features("
        + ",".join(
            f"{name} {types.get(name, 'REAL' if name in FLOAT_FIELDS else 'INTEGER')}"
            for name in COLUMNS
        )
        + ")"
    )
    return db


def joined_audit(path, core_path):
    if not 0 < Path(path).stat().st_size <= 512 * 2**20:
        raise ValueError("Statistical sidecar exceeds its main-file bound.")
    db = connect_readonly(path)
    try:
        configure_database(db)
        db.execute("ATTACH DATABASE ? AS core", (Path(core_path).resolve().as_uri() + "?mode=ro",))
        if (
            db.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
            or [row[1] for row in db.execute("PRAGMA table_info(features)")] != COLUMNS
        ):
            raise ValueError("Statistical sidecar integrity or schema differs.")
        joined = (
            " FROM features f JOIN core.flows c ON f.ordinal=c.ordinal AND f.flow_hash=c.flow_hash"
        )
        count = db.execute("SELECT COUNT(*)" + joined).fetchone()[0]
        if (
            count != db.execute("SELECT COUNT(*) FROM core.flows").fetchone()[0]
            or count != db.execute("SELECT COUNT(*) FROM features").fetchone()[0]
        ):
            raise ValueError("Statistical/core row identities are incomplete or inconsistent.")
        logical = hashlib.sha256()
        query = (
            "SELECT f.*, "
            + ",".join(f"c.{name}" for name in CORE_FEATURES)
            + ",c.start_ms,c.end_ms,c.packets,c.ip_bytes"
            + joined
            + " ORDER BY f.ordinal"
        )
        for number, values in enumerate(db.execute(query), 1):
            sidecar, native = values[: len(COLUMNS)], values[len(COLUMNS) :]
            stored = dict(zip(COLUMNS, sidecar, strict=True))
            row = dict(zip(CORE_FEATURES, native[: len(CORE_FEATURES)], strict=True))
            row.update({name: stored[name] for name in EXTRA_FEATURES})
            row.update(
                dict(
                    zip(
                        (
                            "bidirectional_first_seen_ms",
                            "bidirectional_last_seen_ms",
                            "bidirectional_packets",
                            "bidirectional_bytes",
                        ),
                        native[len(CORE_FEATURES) :],
                        strict=True,
                    )
                )
            )
            validate_statistics(row)
            if (
                stored["ordinal"] != number
                or stored["full_hash"] != pattern_hash([row[name] for name in FEATURES])
                or stored["no_ports_hash"] != pattern_hash([row[name] for name in FEATURES[2:]])
            ):
                raise ValueError("Stored statistical values or pattern hashes changed.")
            logical.update(
                (json.dumps(sidecar, separators=(",", ":"), allow_nan=False) + "\n").encode()
            )
        patterns = {}
        for field in ("full_hash", "no_ports_hash"):
            row = db.execute(
                "SELECT COUNT(*),COALESCE(SUM(classes>1),0),COALESCE(SUM(CASE WHEN classes>1 THEN n ELSE 0 END),0) FROM (SELECT f."
                + field
                + ",COUNT(*) AS n,COUNT(DISTINCT c.label) AS classes"
                + joined
                + " WHERE c.label IS NOT NULL GROUP BY f."
                + field
                + ")"
            ).fetchone()
            patterns[field] = dict(
                zip(
                    ("unique_vectors", "conflicting_label_vectors", "rows_in_conflicting_vectors"),
                    row,
                    strict=True,
                )
            )
        details = {}
        for detail, rows, groups in db.execute(
            "SELECT c.detail,COUNT(*),COUNT(DISTINCT c.group_hash)"
            + joined
            + " WHERE c.label IS NOT NULL GROUP BY c.detail"
        ):
            flags = {}
            for name in EXTRA_FEATURES[8:]:
                flags[name] = dict(
                    zip(
                        ("flows_with_flag", "connections_with_flag"),
                        db.execute(
                            "SELECT COUNT(*),COUNT(DISTINCT c.group_hash)"
                            + joined
                            + f" WHERE c.detail=? AND f.{name}>0",
                            (detail,),
                        ).fetchone(),
                        strict=True,
                    )
                )
            summaries = {}
            for name in EXTRA_FEATURES[:8]:
                summaries[name] = dict(
                    zip(
                        ("min", "flow_weighted_mean", "max"),
                        db.execute(
                            f"SELECT MIN(f.{name}),AVG(f.{name}),MAX(f.{name})"
                            + joined
                            + " WHERE c.detail=?",
                            (detail,),
                        ).fetchone(),
                        strict=True,
                    )
                )
            details[detail] = {
                "accepted_flows": rows,
                "accepted_connections": groups,
                "flags": flags,
                "statistics": summaries,
            }
        return {
            "rows": count,
            "ordered_sidecar_rows_sha256": logical.hexdigest(),
            "complete_identity_core_join": True,
            "patterns": patterns,
            "details": details,
            "scope": "Exact float64 inputs; full/no-port patterns include the sixteen statistics. All accepted conflicting/repeated patterns retained; excluded flows stay present. Connection flag presence means any matched segment, not independent incidents.",
        }
    finally:
        db.close()


def prepare_statistical_cohort(capture_dirs, cohort, output, config_path=CONFIG):
    policy = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if any(
        policy.get(name) != value
        for name, value in {
            "schema_version": 1,
            "schema": SCHEMA,
            "cohort_config": "configs/data/native-cohort.json",
            "expected_core_logical_sha256": "9be2a2d5c39b3cb78a31fec40118971ba03b0e62b398e1151f7480ce8c312d59",
            "source_config": "configs/data/iot23-expansion.json",
            "exporter_version": "6.6.0",
            "statistical_analysis": True,
            "max_database_bytes": 512 * 2**20,
            "cache_kib": 8192,
            "training_enabled": False,
        }.items()
    ):
        raise ValueError("Use the registered statistical measurement policy.")
    output, cohort = Path(output), Path(cohort)
    if output.exists():
        raise FileExistsError(output)
    _, _, plan = load_config(policy["cohort_config"])
    if set(capture_dirs) != set(plan["partitions"].values()):
        raise ValueError("Statistical export requires exactly the registered development captures.")
    output.mkdir(parents=True)
    if shutil.disk_usage(output).free < 2**31 + 512 * 2**20:
        raise ValueError("Statistical sidecar would breach the disk reserve.")
    verified = verify_native_cohort(cohort, output / "core-verification.json")
    if verified["audit"]["rows_logical_sha256"] != policy["expected_core_logical_sha256"]:
        raise ValueError("Statistical export core cohort changed.")
    core_path = cohort / "cohort.sqlite"
    core = connect_readonly(core_path)
    configure_database(core)
    path = output / "statistics.sqlite"
    db = create_sidecar(path)
    records, count = [], 0
    try:
        for split in SPLITS:
            capture_id = plan["partitions"][split]
            directory = Path(capture_dirs[capture_id])
            source, capture, receipt = verified_inputs(directory, policy["source_config"])
            if receipt["capture_id"] != capture_id:
                raise ValueError("Statistical source receipt differs from its planned capture.")
            rows = 0
            with ResourceMonitor(include_children=True) as resources:
                for row in iter_statistical_rows(
                    directory / capture["pcap"],
                    max_nflows=source["max_audit_flows"],
                    max_pcap_bytes=source["max_pcap_bytes"],
                ):
                    validate_statistics(row)
                    flow_hash = digest(flow_key(row))
                    stored = core.execute(
                        "SELECT ordinal,capture_id,"
                        + ",".join(CORE_FEATURES)
                        + ",start_ms,end_ms,packets,ip_bytes FROM flows WHERE flow_hash=?",
                        (flow_hash,),
                    ).fetchone()
                    actual = tuple(row[name] for name in CORE_FEATURES) + tuple(
                        row[name]
                        for name in (
                            "bidirectional_first_seen_ms",
                            "bidirectional_last_seen_ms",
                            "bidirectional_packets",
                            "bidirectional_bytes",
                        )
                    )
                    if (
                        stored is None
                        or stored[0] != count + 1
                        or stored[1] != capture_id
                        or stored[2:] != actual
                    ):
                        raise ValueError(
                            "Statistical export changed a core value, identity, order or capture."
                        )
                    values = [
                        stored[0],
                        flow_hash,
                        *[row[name] for name in EXTRA_FEATURES],
                        pattern_hash([row[name] for name in FEATURES]),
                        pattern_hash([row[name] for name in FEATURES[2:]]),
                    ]
                    db.execute(
                        "INSERT INTO features VALUES (" + ",".join("?" for _ in COLUMNS) + ")",
                        values,
                    )
                    rows += 1
                    count += 1
                db.commit()
            expected = core.execute(
                "SELECT COUNT(*) FROM flows WHERE split=?", (split,)
            ).fetchone()[0]
            record = {
                "capture_id": capture_id,
                "split": split,
                "rows": rows,
                "expected_rows": expected,
                "all_core_values_and_identities_preserved": rows == expected,
                "resources": {
                    "wall_seconds": resources.elapsed,
                    "sampled_peak_process_tree_rss_sum_mib": resources.peak_tree / 2**20,
                },
                "resource_screen_met": resources.elapsed < 600 and resources.peak_tree < 2**30,
            }
            records.append(record)
            write_json(output / "progress.json", {"records": records, "training_authorized": False})
            if rows != expected or not record["resource_screen_met"]:
                raise ValueError("Statistical completeness/resource screen failed.")
        for field in ("full_hash", "no_ports_hash"):
            db.execute(f"CREATE INDEX by_{field} ON features({field})")
        db.commit()
    finally:
        core.close()
        db.close()
    with ResourceMonitor(include_children=True) as resources:
        audit = joined_audit(path, core_path)
    manifest = {
        "schema": SCHEMA,
        "features": FEATURES,
        "settings": {**STAT_SETTINGS, "max_nflows": 250_000},
        "exporter_version": "6.6.0",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config_sha256": sha256_file(config_path),
        "core_logical_sha256": verified["audit"]["rows_logical_sha256"],
        "core_database_sha256": sha256_file(core_path),
        "sidecar": {"bytes": path.stat().st_size, "sha256": sha256_file(path)},
        "records": records,
        "audit": audit,
        "audit_resources": {
            "wall_seconds": resources.elapsed,
            "sampled_peak_process_tree_rss_sum_mib": resources.peak_tree / 2**20,
        },
        "training_authorized": False,
        "candidate_policy_promoted": False,
        "external_test_accessed": False,
        "device_isolation_verified": False,
        "limitations": "Core/label preservation proves compatible measurements, not attack truth or model quality. No checksums, reassembly, DPI, live sensing or new model fits. Cache/file caps and sampled resources are not OS-enforced total limits.",
    }
    write_json(output / "manifest.json", manifest)
    return manifest

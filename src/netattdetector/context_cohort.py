"""Numeric causal context sidecars with full arrival replay and independent SQL samples."""

import hashlib
import json
import math
import shutil
import sqlite3
import subprocess
from pathlib import Path
from uuid import uuid4

from netattdetector.capture_labels import verified_inputs
from netattdetector.causal_context import (
    CONTEXT_FEATURES,
    CONTEXT_METADATA,
    SCHEMA,
    WINDOWS,
    FlowStartContext,
    endpoint,
    validate_context,
)
from netattdetector.io import sha256_file, write_json
from netattdetector.measurement import CORE_FEATURES, flow_key
from netattdetector.native_cohort import SPLITS, digest
from netattdetector.native_cohort import load_config as native_config
from netattdetector.resources import ResourceMonitor
from netattdetector.statistical import EXTRA_FEATURES, FLOAT_FIELDS, STAT_SETTINGS
from netattdetector.statistical import FEATURES as STAT_FEATURES
from netattdetector.statistical_cohort import (
    configure_database,
    connect_readonly,
    pattern_hash,
    verify_statistical_inputs,
)

CONFIG = "configs/data/causal-context.json"
FEATURES = [*STAT_FEATURES, *CONTEXT_FEATURES]
COLUMNS = [
    "ordinal",
    "flow_hash",
    *CONTEXT_FEATURES,
    *CONTEXT_METADATA,
    "full_hash",
    "no_ports_hash",
]


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if any(
        config.get(name) != value
        for name, value in {
            "schema_version": 1,
            "schema": SCHEMA,
            "plugin": "causal-flow-start-v1",
            "expected_core_logical_sha256": "9be2a2d5c39b3cb78a31fec40118971ba03b0e62b398e1151f7480ce8c312d59",
            "expected_statistical_logical_sha256": "a4e6d0ff836814acadd4cc778b0b63320fe5dc38cc080f040caaeb95db189259",
            "statistical_config": "configs/data/statistical-features.json",
            "source_config": "configs/data/iot23-expansion.json",
            "exporter_version": "6.6.0",
            "windows_ms": list(WINDOWS),
            "max_active_events": 100_000,
            "max_database_bytes": 512 * 2**20,
            "cache_kib": 8192,
            "independent_sql_checks_per_capture": 100,
            "training_enabled": False,
            "deployment_authorized": False,
        }.items()
    ):
        raise ValueError("Use the registered training-disabled causal context policy.")
    return config


def create_database(path, *, targets=False):
    with Path(path).open("xb"):
        pass
    db = sqlite3.connect(path, uri=True)
    configure_database(db)
    db.execute("PRAGMA page_size=4096")
    db.execute("PRAGMA max_page_count=131072")
    if targets:
        db.execute(
            "CREATE TABLE targets(ordinal INTEGER PRIMARY KEY,sequence INTEGER UNIQUE,start_ms INTEGER,watermark INTEGER,source BLOB,destination BLOB,port INTEGER)"
        )
    else:
        db.execute(
            "CREATE TABLE features("
            + ",".join(
                "ordinal INTEGER PRIMARY KEY"
                if name == "ordinal"
                else "flow_hash TEXT UNIQUE"
                if name == "flow_hash"
                else f"{name} TEXT"
                if name in ("full_hash", "no_ports_hash")
                else f"{name} INTEGER"
                for name in COLUMNS
            )
            + ")"
        )
    return db


def independent_sql_snapshot(db, sequence, watermark, source, destination):
    values = []
    for field, endpoint_value in (("source", source), ("destination", destination)):
        for window in WINDOWS:
            values.append(
                db.execute(
                    f"SELECT COUNT(*) FROM targets WHERE {field}=? AND watermark>=? AND watermark<? AND sequence<?",
                    (endpoint_value, watermark - window, watermark, sequence),
                ).fetchone()[0]
            )
    for distinct, field, endpoint_value in (
        ("destination", "source", source),
        ("port", "source", source),
        ("source", "destination", destination),
    ):
        values.append(
            db.execute(
                f"SELECT COUNT(DISTINCT {distinct}) FROM targets WHERE {field}=? AND watermark>=? AND watermark<? AND sequence<?",
                (endpoint_value, watermark - WINDOWS[-1], watermark, sequence),
            ).fetchone()[0]
        )
    return dict(zip(CONTEXT_FEATURES, values, strict=True))


def verify_capture_replay(targets, context_path, expected_rows, samples=100):
    targets.execute(
        "ATTACH DATABASE ? AS context", (Path(context_path).resolve().as_uri() + "?mode=ro",)
    )
    total = targets.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    if total != expected_rows or not total:
        raise ValueError("Context target replay is incomplete.")
    targets.execute("CREATE INDEX target_source_time ON targets(source,watermark)")
    targets.execute("CREATE INDEX target_destination_time ON targets(destination,watermark)")
    targets.commit()
    count = min(samples, total)
    chosen = {1} if count == 1 else {1 + i * (total - 1) // (count - 1) for i in range(count)}
    state = FlowStartContext()
    late = warmup = checked = 0
    query = (
        "SELECT t.ordinal,t.sequence,t.start_ms,t.watermark,t.source,t.destination,t.port,"
        + ",".join(f"f.{name}" for name in [*CONTEXT_FEATURES, *CONTEXT_METADATA])
        + " FROM targets t JOIN context.features f ON t.ordinal=f.ordinal ORDER BY t.sequence"
    )
    replayed = 0
    for row in targets.execute(query):
        ordinal, sequence, start, watermark, source, destination, port = row[:7]
        actual = dict(zip([*CONTEXT_FEATURES, *CONTEXT_METADATA], row[7:], strict=True))
        replayed += 1
        expected = state.observe(start, source, destination, port)
        if (
            sequence != replayed
            or watermark != expected["context_watermark_ms"]
            or actual != expected
        ):
            raise ValueError(f"Causal callback/replay mismatch at stored ordinal {ordinal}.")
        if sequence in chosen:
            sql = independent_sql_snapshot(targets, sequence, watermark, source, destination)
            if sql != {name: actual[name] for name in CONTEXT_FEATURES}:
                raise ValueError("Independent SQL context snapshot differs from the callback.")
            checked += 1
        late += actual["context_late_start"]
        warmup += not actual["context_warmup_complete"]
    if replayed != expected_rows or checked != count:
        raise ValueError("Causal full/sample verification did not cover its declared rows.")
    return {
        "rows_replayed": replayed,
        "full_callback_replay_identical": True,
        "independent_sql_snapshots": checked,
        "independent_sql_snapshots_identical": True,
        "late_flow_starts": late,
        "warmup_flow_rows": warmup,
        "peak_active_plus_pending_events": state.peak,
        "scope": "Full arrival-sequence replay shares the tested window routine; the deterministic SQL samples are algorithmically independent. Original times/labels are unchanged.",
    }


def joined_context_audit(path, cohort, statistics):
    db = connect_readonly(path)
    try:
        configure_database(db)
        db.execute(
            "ATTACH DATABASE ? AS core",
            ((Path(cohort) / "cohort.sqlite").resolve().as_uri() + "?mode=ro",),
        )
        db.execute(
            "ATTACH DATABASE ? AS statistics",
            ((Path(statistics) / "statistics.sqlite").resolve().as_uri() + "?mode=ro",),
        )
        if (
            Path(path).stat().st_size > 512 * 2**20
            or db.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
            or [row[1] for row in db.execute("PRAGMA table_info(features)")] != COLUMNS
        ):
            raise ValueError("Context database size/integrity/schema failed.")
        join = " FROM features f JOIN core.flows c ON f.ordinal=c.ordinal AND f.flow_hash=c.flow_hash JOIN statistics.features s ON c.ordinal=s.ordinal AND c.flow_hash=s.flow_hash"
        count = db.execute("SELECT COUNT(*)" + join).fetchone()[0]
        if (
            count != db.execute("SELECT COUNT(*) FROM core.flows").fetchone()[0]
            or count != db.execute("SELECT COUNT(*) FROM features").fetchone()[0]
        ):
            raise ValueError("Context/core/statistical identities are not complete.")
        logical = hashlib.sha256()
        fields = [
            *[f"c.{name}" for name in CORE_FEATURES],
            *[f"s.{name}" for name in EXTRA_FEATURES],
        ]
        for row in db.execute("SELECT f.*," + ",".join(fields) + join + " ORDER BY f.ordinal"):
            stored = dict(zip(COLUMNS, row[: len(COLUMNS)], strict=True))
            validate_context(stored)
            values = [*row[len(COLUMNS) :], *[stored[name] for name in CONTEXT_FEATURES]]
            if stored["full_hash"] != pattern_hash(values) or stored[
                "no_ports_hash"
            ] != pattern_hash(values[2:]):
                raise ValueError("Context numeric values or pattern hashes changed.")
            logical.update(
                (
                    json.dumps(row[: len(COLUMNS)], separators=(",", ":"), allow_nan=False) + "\n"
                ).encode()
            )
        # Independent metadata consistency remains verifiable after endpoint targets are removed.
        for (capture_id,) in db.execute("SELECT DISTINCT capture_id FROM core.flows"):
            watermark = first = None
            sequence = peak = 0
            for start, seq, clock, warm, late, current_peak in db.execute(
                "SELECT c.start_ms,f.context_init_sequence,f.context_watermark_ms,f.context_warmup_complete,f.context_late_start,f.context_peak_events"
                + join
                + " WHERE c.capture_id=? ORDER BY f.context_init_sequence",
                (capture_id,),
            ):
                previous = watermark
                watermark = start if watermark is None else max(start, watermark)
                first = watermark if first is None else first
                sequence += 1
                if (
                    seq != sequence
                    or clock != watermark
                    or warm != int(watermark - first >= WINDOWS[-1])
                    or late != int(previous is not None and start < previous)
                    or current_peak < peak
                ):
                    raise ValueError("Context arrival/prefix-clock/warm-up metadata differs.")
                peak = current_peak
        patterns = {}
        for field in ("full_hash", "no_ports_hash"):
            values = db.execute(
                "SELECT COUNT(*),COALESCE(SUM(classes>1),0),COALESCE(SUM(CASE WHEN classes>1 THEN n ELSE 0 END),0) FROM (SELECT f."
                + field
                + ",COUNT(*) AS n,COUNT(DISTINCT c.label) AS classes"
                + join
                + " WHERE c.label IS NOT NULL GROUP BY f."
                + field
                + ")"
            ).fetchone()
            patterns[field] = dict(
                zip(
                    ("unique_vectors", "conflicting_label_vectors", "rows_in_conflicting_vectors"),
                    values,
                    strict=True,
                )
            )
        warmup = []
        for split, ready, label, rows, groups in db.execute(
            "SELECT c.split,f.context_warmup_complete,c.label,COUNT(*),COUNT(DISTINCT c.group_hash)"
            + join
            + " GROUP BY c.split,f.context_warmup_complete,c.label"
        ):
            warmup.append(
                {
                    "split": split,
                    "warmup_complete": bool(ready),
                    "label": label,
                    "flows": rows,
                    "accepted_uid_groups": groups,
                }
            )
        return {
            "rows": count,
            "complete_identity_core_statistical_join": True,
            "ordered_context_rows_sha256": logical.hexdigest(),
            "patterns": patterns,
            "warmup_counts": warmup,
            "scope": "All flows retained; patterns use accepted labels only. Warm-up metadata stays outside model features. Packet-callback/SQL checks occurred before private endpoint targets were removed.",
        }
    finally:
        db.close()


def prepare_context_cohort(capture_dirs, cohort, statistics, output, config_path=CONFIG):
    config = load_config(config_path)
    cohort, statistics, output = map(Path, (cohort, statistics, output))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    if shutil.disk_usage(output).free < 2**31 + 2 * config["max_database_bytes"]:
        raise ValueError("Context sidecar/temporary targets would breach the disk reserve.")
    verify_statistical_inputs(cohort, statistics, output, config)
    _, _, plan = native_config("configs/data/native-cohort.json")
    if set(capture_dirs) != set(plan["partitions"].values()):
        raise ValueError("Context lab requires exactly the three registered development captures.")
    core = connect_readonly(cohort / "cohort.sqlite")
    configure_database(core)
    core.execute(
        "ATTACH DATABASE ? AS statistics",
        ((statistics / "statistics.sqlite").resolve().as_uri() + "?mode=ro",),
    )
    path = output / "context.sqlite"
    writer = create_database(path)
    records, count = [], 0
    fields = [
        *[f"c.{name}" for name in CORE_FEATURES],
        *[f"s.{name}" for name in EXTRA_FEATURES],
        "c.start_ms",
        "c.end_ms",
        "c.packets",
        "c.ip_bytes",
    ]
    select_original = (
        "SELECT c.ordinal,c.capture_id,"
        + ",".join(fields)
        + " FROM flows c JOIN statistics.features s ON c.ordinal=s.ordinal AND c.flow_hash=s.flow_hash WHERE c.flow_hash=?"
    )
    insert_context = "INSERT INTO features VALUES (" + ",".join("?" for _ in COLUMNS) + ")"
    measurement_names = [
        *STAT_FEATURES,
        "bidirectional_first_seen_ms",
        "bidirectional_last_seen_ms",
        "bidirectional_packets",
        "bidirectional_bytes",
    ]
    try:
        for split in SPLITS:
            capture_id = plan["partitions"][split]
            directory = Path(capture_dirs[capture_id])
            source, capture, receipt = verified_inputs(directory, config["source_config"])
            if receipt["capture_id"] != capture_id:
                raise ValueError("Context source differs from the planned capture.")
            target_path = output / f".context-targets-{uuid4().hex}.sqlite"
            targets = create_database(target_path, targets=True)
            try:
                from netattdetector.context_meter import iter_context_rows

                rows = nonexact_floats = 0
                max_float_difference = 0.0
                with ResourceMonitor(include_children=True) as resources:
                    for row in iter_context_rows(
                        directory / capture["pcap"],
                        max_nflows=source["max_audit_flows"],
                        max_pcap_bytes=source["max_pcap_bytes"],
                    ):
                        identity = digest(flow_key(row))
                        old = core.execute(select_original, (identity,)).fetchone()
                        actual = tuple(row[name] for name in measurement_names)
                        if old is None or old[0] != count + 1 or old[1] != capture_id:
                            raise ValueError(
                                "Context callback changed a flow identity, order or capture."
                            )
                        for name, expected_value, value in zip(
                            measurement_names, old[2:], actual, strict=True
                        ):
                            if name in FLOAT_FIELDS:
                                difference = abs(expected_value - value)
                                max_float_difference = max(max_float_difference, difference)
                                nonexact_floats += expected_value != value
                                if not math.isclose(
                                    expected_value, value, rel_tol=1e-9, abs_tol=1e-9
                                ):
                                    raise ValueError(
                                        "Context statistical float differs beyond its declared measurement tolerance."
                                    )
                            elif expected_value != value:
                                raise ValueError(
                                    "Context callback changed an exact core/statistical counter or time."
                                )
                        numeric = [
                            *old[2 : 2 + len(STAT_FEATURES)],
                            *[row[name] for name in CONTEXT_FEATURES],
                        ]
                        values = [
                            old[0],
                            identity,
                            *[row[name] for name in [*CONTEXT_FEATURES, *CONTEXT_METADATA]],
                            pattern_hash(numeric),
                            pattern_hash(numeric[2:]),
                        ]
                        writer.execute(insert_context, values)
                        targets.execute(
                            "INSERT INTO targets VALUES (?,?,?,?,?,?,?)",
                            (
                                old[0],
                                row["context_init_sequence"],
                                row["bidirectional_first_seen_ms"],
                                row["context_watermark_ms"],
                                endpoint(row["src_ip"]),
                                endpoint(row["dst_ip"]),
                                row["dst_port"],
                            ),
                        )
                        count += 1
                        rows += 1
                    writer.commit()
                    targets.commit()
                    expected = core.execute(
                        "SELECT COUNT(*) FROM flows WHERE split=?", (split,)
                    ).fetchone()[0]
                    replay = verify_capture_replay(
                        targets, path, expected, config["independent_sql_checks_per_capture"]
                    )
                record = {
                    "capture_id": capture_id,
                    "split": split,
                    "rows": rows,
                    "all_core_values_and_statistical_tolerances_preserved": rows == expected,
                    "nonexact_statistical_float_fields": nonexact_floats,
                    "maximum_statistical_float_absolute_difference": max_float_difference,
                    "replay": replay,
                    "resources": {
                        "wall_seconds": resources.elapsed,
                        "sampled_peak_process_tree_rss_sum_mib": resources.peak_tree / 2**20,
                    },
                    "resource_screen_met": resources.elapsed < 600 and resources.peak_tree < 2**30,
                }
                records.append(record)
                write_json(
                    output / "progress.json", {"records": records, "training_authorized": False}
                )
                if rows != expected or not record["resource_screen_met"]:
                    raise ValueError("Context capture completeness/resource screen failed.")
            finally:
                targets.close()
                target_path.unlink(missing_ok=True)
        for field in ("full_hash", "no_ports_hash"):
            writer.execute(f"CREATE INDEX by_{field} ON features({field})")
        writer.commit()
    finally:
        core.close()
        writer.close()
    with ResourceMonitor(include_children=True) as resources:
        audit = joined_context_audit(path, cohort, statistics)
    manifest = {
        "schema": SCHEMA,
        "features": FEATURES,
        "exporter_version": "6.6.0",
        "settings": {**STAT_SETTINGS, "max_nflows": 250_000},
        "context_policy": config,
        "config_sha256": sha256_file(config_path),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "core_logical_sha256": config["expected_core_logical_sha256"],
        "statistical_logical_sha256": config["expected_statistical_logical_sha256"],
        "core_database_sha256": sha256_file(cohort / "cohort.sqlite"),
        "statistical_database_sha256": sha256_file(statistics / "statistics.sqlite"),
        "database": {"bytes": path.stat().st_size, "sha256": sha256_file(path)},
        "records": records,
        "audit": audit,
        "audit_resources": {
            "wall_seconds": resources.elapsed,
            "sampled_peak_process_tree_rss_sum_mib": resources.peak_tree / 2**20,
        },
        "training_authorized": False,
        "deployment_authorized": False,
        "candidate_policy_promoted": False,
        "external_test_accessed": False,
        "limitations": "Observed flow-start watermark clock, not verified physical uptime/device identity or live loss. Same-watermark peers are excluded; warm-up rows are retained. Context uses every exported flow independent of labels. No model fits or label repair.",
    }
    write_json(output / "manifest.json", manifest)
    return manifest

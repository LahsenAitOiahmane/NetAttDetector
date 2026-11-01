"""Bounded, private native-flow candidate cohorts; never authorizes model training."""

import hashlib
import json
import platform
import shutil
import sqlite3
import subprocess
from itertools import combinations, pairwise
from pathlib import Path
from uuid import uuid4

from netattdetector.capture_labels import indexed_labels, match_flow, verified_inputs
from netattdetector.io import sha256_file, write_json
from netattdetector.measurement import CORE_FEATURES, SCHEMA, SETTINGS, flow_key, iter_pcap_rows
from netattdetector.packet_evidence import (
    POLICY,
    MatchCounts,
    PacketIndex,
    PointLabels,
    candidate_match,
    load_packet_policy,
)
from netattdetector.resources import ResourceMonitor

COHORT_SCHEMA = "native-cohort-v1"
CONFIG = "configs/data/native-cohort.json"
SPLITS = ("train", "calibration", "validation")
COLUMNS = [
    "ordinal",
    "split",
    "capture_id",
    "flow_hash",
    "group_hash",
    "uid_hash",
    "status",
    "label",
    "detail",
    "feature_hash",
    "start_ms",
    "end_ms",
    "packets",
    "ip_bytes",
    *CORE_FEATURES,
]
EXCLUDED = {
    "unsupported_protocol",
    "unmatched",
    "ambiguous",
    "unknown_label_or_duration",
    "partial_interval",
    "point_flow_counter_mismatch",
}


def canonical(value):
    return (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def integer(value):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError("Native fields must be exact nonnegative int64 integers.")
    return value


def validate_values(values):
    for name in ["start_ms", "end_ms", "packets", "ip_bytes", *CORE_FEATURES]:
        integer(values[name])
    if (
        values["src_port"] > 65535
        or values["dst_port"] > 65535
        or values["protocol"] > 255
        or values["ip_version"] not in (4, 6)
        or values["end_ms"] - values["start_ms"] != values["bidirectional_duration_ms"]
        or values["packets"] != values["src2dst_packets"] + values["dst2src_packets"]
        or values["ip_bytes"] != values["src2dst_bytes"] + values["dst2src_bytes"]
        or values["src2dst_duration_ms"] > values["bidirectional_duration_ms"]
        or values["dst2src_duration_ms"] > values["bidirectional_duration_ms"]
        or values["packets"] == 0
    ):
        raise ValueError("Invalid native feature ranges, times or directional counters.")


class CohortWriter:
    def __init__(self, path):
        self.path = Path(path)
        # Exclusive creation also protects an existing empty database.
        with self.path.open("xb"):
            pass
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA page_size=4096")
        self.db.execute("PRAGMA cache_size=-8192")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.execute("PRAGMA max_page_count=131072")
        text = {
            "split",
            "capture_id",
            "flow_hash",
            "group_hash",
            "uid_hash",
            "status",
            "detail",
            "feature_hash",
        }
        self.db.execute(
            "CREATE TABLE flows("
            + ",".join(
                "ordinal INTEGER PRIMARY KEY"
                if name == "ordinal"
                else f"{name} {'TEXT' if name in text else 'INTEGER'}"
                for name in COLUMNS
            )
            + ", UNIQUE(flow_hash))"
        )
        self.count = 0

    def add(self, split, capture_id, flow, status, row):
        if split not in SPLITS or not isinstance(capture_id, str):
            raise ValueError("Unknown split or capture identity.")
        values = {
            **{name: flow[name] for name in CORE_FEATURES},
            "start_ms": flow["bidirectional_first_seen_ms"],
            "end_ms": flow["bidirectional_last_seen_ms"],
            "packets": flow["bidirectional_packets"],
            "ip_bytes": flow["bidirectional_bytes"],
        }
        validate_values(values)
        if (status == "matched") != (row is not None) or (
            status != "matched" and status not in EXCLUDED
        ):
            raise ValueError("Unknown/inconsistent linkage status.")
        if row is not None and (
            type(row["label"]) is not int
            or row["label"] not in (0, 1)
            or not isinstance(row["uid"], str)
            or not row["uid"]
            or not isinstance(row["detail"], str)
            or not row["detail"]
        ):
            raise ValueError("Matched rows require an explicit binary label, UID and detail.")
        uid_hash = digest(["iot23", "uid", row["uid"]]) if row else None
        values.update(
            {
                "ordinal": self.count + 1,
                "split": split,
                "capture_id": capture_id,
                "flow_hash": digest(flow_key(flow)),
                "uid_hash": uid_hash,
                "group_hash": digest(["iot23", capture_id, uid_hash]) if row else None,
                "status": status,
                "label": row["label"] if row else None,
                "detail": row["detail"] if row else None,
                "feature_hash": digest([values[name] for name in CORE_FEATURES]),
            }
        )
        self.db.execute(
            "INSERT INTO flows VALUES (" + ",".join("?" for _ in COLUMNS) + ")",
            [values[name] for name in COLUMNS],
        )
        self.count += 1
        if self.count >= 750_000:
            raise ValueError("Cohort reached its registered completeness/flow limit.")

    def finish(self):
        for name in ("group_hash", "feature_hash", "uid_hash", "split"):
            self.db.execute(f"CREATE INDEX by_{name} ON flows({name})")
        self.db.commit()

    def close(self):
        self.db.close()


def scalar(db, query, parameters=()):
    return db.execute(query, parameters).fetchone()[0]


def audit_database(path, plan):
    """Reopen read-only; validate every row and compute bounded SQL aggregate diagnostics."""
    path = Path(path)
    if not 0 < path.stat().st_size <= 512 * 2**20:
        raise ValueError("Cohort database exceeds the registered byte limit.")
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        db.execute("PRAGMA cache_size=-8192")
        db.execute("PRAGMA temp_store=FILE")
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Cohort SQLite integrity check failed.")
        if [row[1] for row in db.execute("PRAGMA table_info(flows)")] != COLUMNS:
            raise ValueError("Unexpected native cohort column contract.")
        logical = hashlib.sha256()
        count = 0
        for record in db.execute("SELECT * FROM flows ORDER BY ordinal"):
            count += 1
            row = dict(zip(COLUMNS, record, strict=True))
            validate_values(row)
            if (
                row["ordinal"] != count
                or row["split"] not in SPLITS
                or plan["partitions"].get(row["split"]) != row["capture_id"]
            ):
                raise ValueError("Row ordinal or actual capture/partition membership differs.")
            if row["feature_hash"] != digest([row[name] for name in CORE_FEATURES]):
                raise ValueError("Native feature hash differs from stored feature values.")
            if row["status"] == "matched":
                if (
                    type(row["label"]) is not int
                    or row["label"] not in (0, 1)
                    or not row["uid_hash"]
                    or not row["detail"]
                    or row["group_hash"] != digest(["iot23", row["capture_id"], row["uid_hash"]])
                ):
                    raise ValueError("Invalid accepted label or connection group.")
            elif row["status"] not in EXCLUDED or any(
                row[name] is not None for name in ("label", "uid_hash", "group_hash", "detail")
            ):
                raise ValueError("Excluded rows must not acquire labels or connection groups.")
            logical.update(canonical(record))
            if count >= 750_000:
                raise ValueError("Native row limit reached.")
        if not count:
            raise ValueError("Empty native cohort.")
        partitions = []
        for split in SPLITS:
            where = " WHERE split=?"
            rows = scalar(db, "SELECT COUNT(*) FROM flows" + where, (split,))
            if not rows:
                raise ValueError("Every registered native partition must contain rows.")
            first, last = db.execute(
                "SELECT MIN(start_ms), MAX(end_ms) FROM flows" + where, (split,)
            ).fetchone()
            partitions.append(
                {
                    "split": split,
                    "capture_id": plan["partitions"][split],
                    "rows": rows,
                    "start_ms": first,
                    "end_ms": last,
                    "status_counts": dict(
                        db.execute(
                            "SELECT status,COUNT(*) FROM flows" + where + " GROUP BY status",
                            (split,),
                        )
                    ),
                    "matched_class_rows": dict(
                        (str(label), total)
                        for label, total in db.execute(
                            "SELECT label,COUNT(*) FROM flows"
                            + where
                            + " AND label IS NOT NULL GROUP BY label",
                            (split,),
                        )
                    ),
                    "matched_class_connections": dict(
                        (str(label), total)
                        for label, total in db.execute(
                            "SELECT label,COUNT(DISTINCT group_hash) FROM flows"
                            + where
                            + " AND label IS NOT NULL GROUP BY label",
                            (split,),
                        )
                    ),
                    "matched_detail_connections": dict(
                        db.execute(
                            "SELECT detail,COUNT(DISTINCT group_hash) FROM flows"
                            + where
                            + " AND label IS NOT NULL GROUP BY detail",
                            (split,),
                        )
                    ),
                }
            )
        bad_groups = scalar(
            db,
            "SELECT COUNT(*) FROM (SELECT group_hash FROM flows WHERE group_hash IS NOT NULL GROUP BY group_hash HAVING COUNT(DISTINCT split)>1 OR COUNT(DISTINCT capture_id)>1 OR COUNT(DISTINCT label)>1 OR COUNT(DISTINCT detail)>1)",
        )
        repeated_ids = scalar(
            db,
            "SELECT COUNT(*) FROM (SELECT flow_hash FROM flows GROUP BY flow_hash HAVING COUNT(*)>1)",
        )
        repeated_uids = scalar(
            db,
            "SELECT COUNT(*) FROM (SELECT uid_hash FROM flows WHERE uid_hash IS NOT NULL GROUP BY uid_hash HAVING COUNT(DISTINCT capture_id)>1)",
        )
        feature_summary = db.execute(
            "SELECT COUNT(*),COALESCE(SUM(n),0),COALESCE(SUM(n>1),0),COALESCE(SUM(classes>1),0),COALESCE(SUM(CASE WHEN classes>1 THEN n ELSE 0 END),0) FROM (SELECT feature_hash,COUNT(*) AS n,COUNT(DISTINCT label) AS classes FROM flows WHERE label IS NOT NULL GROUP BY feature_hash)"
        ).fetchone()
        overlaps = []
        for left, right in combinations(SPLITS, 2):
            shared, lrows, rrows, conflict = db.execute(
                "WITH a AS (SELECT feature_hash,COUNT(*) AS n,MIN(label) AS lo,MAX(label) AS hi FROM flows WHERE split=? AND label IS NOT NULL GROUP BY feature_hash), b AS (SELECT feature_hash,COUNT(*) AS n,MIN(label) AS lo,MAX(label) AS hi FROM flows WHERE split=? AND label IS NOT NULL GROUP BY feature_hash) SELECT COUNT(*),COALESCE(SUM(a.n),0),COALESCE(SUM(b.n),0),COALESCE(SUM(a.lo!=b.hi OR a.hi!=b.lo),0) FROM a JOIN b USING(feature_hash)",
                (left, right),
            ).fetchone()
            overlaps.append(
                {
                    "left": left,
                    "right": right,
                    "shared_feature_vectors": shared,
                    "left_rows": lrows,
                    "right_rows": rrows,
                    "shared_vectors_with_conflicting_labels": conflict,
                }
            )
        checks = {
            "flow_identities_unique": repeated_ids == 0,
            "connection_groups_disjoint_and_consistent": bad_groups == 0,
            "capture_time_order_and_purge_valid": all(
                left["end_ms"] + plan["purge_seconds"] * 1000 <= right["start_ms"]
                for left, right in pairwise(partitions)
            ),
            "distinct_connection_class_support_met": all(
                min(
                    p["matched_class_connections"].get("0", 0),
                    p["matched_class_connections"].get("1", 0),
                )
                >= plan["minimum_distinct_connections_per_class"]
                for p in partitions
            ),
        }
        return {
            "rows": count,
            "rows_logical_sha256": logical.hexdigest(),
            "partitions": partitions,
            "checks": checks,
            "preparation_checks_passed": all(checks.values()),
            "invalid_or_cross_partition_groups": bad_groups,
            "repeated_flow_identities": repeated_ids,
            "uid_pseudonyms_reused_across_captures": repeated_uids,
            "matched_feature_patterns": dict(
                zip(
                    (
                        "unique_vectors",
                        "rows",
                        "repeated_vectors",
                        "conflicting_label_vectors",
                        "rows_in_conflicting_vectors",
                    ),
                    feature_summary,
                    strict=True,
                )
            ),
            "exact_feature_overlap": overlaps,
            "overlap_interpretation": "Matched rows only; repeated feature patterns are retained. These diagnostics do not establish device independence or authorize fitting.",
        }
    finally:
        db.close()


def load_config(config_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if any(
        config.get(name) != value
        for name, value in {
            "schema_version": 1,
            "cohort_schema": COHORT_SCHEMA,
            "feature_contract": SCHEMA,
            "label_policy": POLICY,
            "training_enabled": False,
            "max_database_bytes": 512 * 2**20,
            "cache_kib": 8192,
            "max_total_flows": 750_000,
            "disk_reserve_bytes": 2**31,
            "minimum_distinct_connections_per_class": 25,
            "purge_seconds": 300,
            "source_config": "configs/data/iot23-expansion.json",
        }.items()
    ):
        raise ValueError("Use the registered training-disabled native cohort configuration.")
    policy = load_packet_policy(config["packet_policy"])
    plan = json.loads(Path(config["plan"]).read_text(encoding="utf-8"))
    if (
        policy["source_config"] != config["source_config"]
        or plan["source_config"] != config["source_config"]
        or plan["training_enabled"] is not False
        or plan["source_id"] != "iot23"
        or plan["label_policy"] != POLICY
        or plan["feature_contract"] != SCHEMA
        or set(plan["partitions"]) != set(SPLITS)
        or len(set(plan["partitions"].values())) != 3
        or plan["purge_seconds"] != 300
        or plan["minimum_distinct_connections_per_class"] != 25
        or plan["device_isolation_verified"] is not False
        or plan["external_test_acquired"] is not False
    ):
        raise ValueError("Native plan/policy differs from the registered preparation scope.")
    return config, policy, plan


def check_replay_audit(records, audit):
    """Tie stored labels/groups back to the counters from the independently replayed matcher."""
    for record, partition in zip(records, audit["partitions"], strict=True):
        candidate = record["candidate"]
        if (
            record["split"] != partition["split"]
            or record["capture_id"] != partition["capture_id"]
            or record["exported_flows"] != partition["rows"]
            or candidate["flow_status_counts"] != partition["status_counts"]
            or candidate["matched_flow_class_counts"] != partition["matched_class_rows"]
            or candidate["matched_zeek_connection_class_counts"]
            != partition["matched_class_connections"]
            or candidate["matched_connections_by_detail"] != partition["matched_detail_connections"]
        ):
            raise ValueError("Stored native rows/groups differ from the archived matcher replay.")


def prepare_native_cohort(capture_dirs, output, config_path=CONFIG):
    if platform.system() != "Linux":
        raise ValueError(
            "Native export requires the pinned Linux NFStream measurement environment."
        )
    config, policy, plan = load_config(config_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if set(capture_dirs) != set(plan["partitions"].values()):
        raise ValueError("Provide exactly the three registered native development captures.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if (
        shutil.disk_usage(output.parent).free
        < config["disk_reserve_bytes"] + config["max_database_bytes"]
    ):
        raise ValueError("Native cohort would breach the disk reserve.")
    # Verify all pins before creating any cohort output.
    verified = {
        key: verified_inputs(value, config["source_config"]) for key, value in capture_dirs.items()
    }
    if any(key != item[2]["capture_id"] for key, item in verified.items()):
        raise ValueError("Capture directory receipt does not match its planned partition.")
    output.mkdir()
    writer = CohortWriter(output / "cohort.sqlite")
    records = []
    try:
        for split in SPLITS:
            capture_id = plan["partitions"][split]
            capture_dir = Path(capture_dirs[capture_id])
            source, capture, receipt = verified[capture_id]
            packet_path = output / f".packets-{uuid4().hex}.sqlite"
            packets = PacketIndex(packet_path)
            try:
                with (
                    ResourceMonitor(include_children=True) as resources,
                    indexed_labels(capture_dir, source) as (parsed, _),
                ):
                    packet_counts = packets.build(capture_dir / capture["pcap"])
                    points = packets.establish_points(capture_dir / "bro/conn.log.labeled")
                    labels = PointLabels(parsed[0], packets)
                    before, after = MatchCounts(), MatchCounts()
                    preserved = True
                    count = 0
                    for flow in iter_pcap_rows(
                        capture_dir / capture["pcap"],
                        max_nflows=source["max_audit_flows"],
                        max_pcap_bytes=source["max_pcap_bytes"],
                    ):
                        old_status, old = match_flow(flow, parsed[0])
                        status, row = candidate_match(flow, labels)
                        before.add(flow_key(flow), old_status, old)
                        after.add(flow_key(flow), status, row)
                        if old is not None and (
                            row is None
                            or any(old[name] != row[name] for name in ("uid", "label", "detail"))
                        ):
                            preserved = False
                        writer.add(split, capture_id, flow, status, row)
                        count += 1
                    writer.db.commit()
                saved = json.loads(
                    (
                        Path("reports/evidence/packet-labels/cloud")
                        / f"{capture_id}-packet-evidence.json"
                    ).read_text(encoding="utf-8")
                )
                parity = (
                    before.report() == saved["baseline"] and after.report() == saved["candidate"]
                )
                record = {
                    "capture_id": capture_id,
                    "split": split,
                    "exported_flows": count,
                    "input_files": receipt["files"],
                    "receipt_sha256": sha256_file(capture_dir / "receipt.json"),
                    "baseline": before.report(),
                    "candidate": after.report(),
                    "archived_assignments_identical": parity,
                    "previous_matches_preserved": preserved,
                    "packet_status_counts": packet_counts,
                    "point_evidence": points,
                    "resources": {
                        "wall_seconds": resources.elapsed,
                        "sampled_peak_process_rss_mib": resources.peak / 2**20,
                        "sampled_peak_process_tree_rss_sum_mib": resources.peak_tree / 2**20,
                    },
                    "resource_screen_met": resources.elapsed < 600 and resources.peak_tree < 2**30,
                }
                records.append(record)
                write_json(output / f"{capture_id}-preparation.json", record)
                if not parity or not preserved or not record["resource_screen_met"]:
                    raise ValueError(
                        "Native preparation replay/preservation/resource screen failed."
                    )
            finally:
                packets.close()
                packet_path.unlink(missing_ok=True)
        writer.finish()
    finally:
        writer.close()
    with ResourceMonitor(include_children=True) as validation_resources:
        audit = audit_database(output / "cohort.sqlite", plan)
    if not audit["preparation_checks_passed"]:
        write_json(output / "failed-audit.json", audit)
        raise ValueError("Native partition acceptance failed; no completed manifest written.")
    check_replay_audit(records, audit)
    hashes = {
        name: sha256_file(name)
        for name in (
            str(config_path),
            config["plan"],
            config["packet_policy"],
            config["source_config"],
            "configs/data/source_roles.json",
        )
    }
    for name, value in (
        ("cohort-config.json", config),
        ("dataset-plan.json", plan),
        ("packet-policy.json", policy),
    ):
        write_json(output / name, value)
    manifest = {
        "schema": COHORT_SCHEMA,
        "status": "complete_experimental_candidate",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config_hashes": hashes,
        "feature_contract": SCHEMA,
        "features": CORE_FEATURES,
        "exporter": "nfstream",
        "exporter_version": "6.6.0",
        "export_settings": {**SETTINGS, "max_nflows": 250_000},
        "label_policy": POLICY,
        "database": {
            "name": "cohort.sqlite",
            "bytes": (output / "cohort.sqlite").stat().st_size,
            "sha256": sha256_file(output / "cohort.sqlite"),
        },
        "records": records,
        "audit": audit,
        "validation_resources": {
            "wall_seconds": validation_resources.elapsed,
            "sampled_peak_process_tree_rss_sum_mib": validation_resources.peak_tree / 2**20,
        },
        "training_authorized": False,
        "candidate_policy_promoted": False,
        "device_isolation_verified": False,
        "external_test_acquired": False,
        "limitations": [
            "Capture/time separation is not device independence.",
            "Point labels are an experimental linkage candidate, not independently verified attack truth.",
            "Zero-counter DDoS contradictions, packet tails and unsupported protocols remain unresolved.",
            "Repeated patterns and conflicting labels are retained and reported.",
            "Raw/derived rows remain private; mirror/release terms remain unresolved.",
            "Sampled RSS can miss brief peaks/double-count shared pages; the SQLite cap excludes journals/temp files. No OS-enforced process limit.",
        ],
    }
    # Completion marker is written last. Fresh output directories prohibit silent overwrites.
    write_json(output / "manifest.json", manifest)
    return manifest


def verify_native_cohort(directory, output):
    directory, output = Path(directory), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["schema"] != COHORT_SCHEMA
        or manifest["status"] != "complete_experimental_candidate"
        or manifest["training_authorized"] is not False
        or manifest["candidate_policy_promoted"] is not False
        or manifest["features"] != CORE_FEATURES
        or manifest["feature_contract"] != SCHEMA
        or manifest["label_policy"] != POLICY
        or manifest["database"]["name"] != "cohort.sqlite"
    ):
        raise ValueError("Unexpected or training-enabled native cohort manifest.")
    for path, expected in manifest["config_hashes"].items():
        if sha256_file(path) != expected:
            raise ValueError("Native cohort configuration changed since materialization.")
    config_path = next(
        path for path in manifest["config_hashes"] if path.endswith("native-cohort.json")
    )
    _, _, plan = load_config(config_path)
    path = directory / "cohort.sqlite"
    if (
        path.stat().st_size != manifest["database"]["bytes"]
        or sha256_file(path) != manifest["database"]["sha256"]
    ):
        raise ValueError("Native cohort database bytes changed.")
    with ResourceMonitor(include_children=True) as resources:
        audit = audit_database(path, plan)
    if audit != manifest["audit"] or not audit["preparation_checks_passed"]:
        raise ValueError("Native cohort logical rows/partition audit differs from its manifest.")
    check_replay_audit(manifest["records"], audit)
    result = {
        "verified": True,
        "schema": COHORT_SCHEMA,
        "database_sha256": manifest["database"]["sha256"],
        "audit": audit,
        "training_authorized": False,
        "candidate_policy_promoted": False,
        "resources": {
            "wall_seconds": resources.elapsed,
            "sampled_peak_process_tree_rss_sum_mib": resources.peak_tree / 2**20,
        },
    }
    write_json(output, result)
    return result

"""Development-only exact packet witnesses and a paired candidate point-label audit."""

import hashlib
import ipaddress
import json
import platform
import sqlite3
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from netattdetector.capture_labels import (
    connection_key,
    indexed_labels,
    match_flow,
    verified_inputs,
)
from netattdetector.io import sha256_file, write_json
from netattdetector.label_index import SQLiteLabelIndex, nanoseconds
from netattdetector.measurement import SETTINGS, flow_key, iter_pcap_rows
from netattdetector.packet_reader import packet_records
from netattdetector.resources import ResourceMonitor

POLICY = "iot23-packet-point-candidate-v1"


def endpoint(address, port):
    ip = ipaddress.ip_address(address)
    return json.dumps((ip.version, int(ip), int(port)), separators=(",", ":"))


def label_rows(path):
    """Second pass after the standard parser validates header, rows, UIDs and bounds."""
    with Path(path).open(encoding="utf-8") as stream:
        columns = None
        for line in stream:
            if line.startswith("#fields"):
                columns = [
                    "detailed-label" if item == "det_label" else item for item in line.split()[1:]
                ]
            elif not line.startswith("#") and line.strip():
                if columns is None:
                    raise ValueError("Label header missing.")
                yield dict(zip(columns, line.split(), strict=True))


class PacketIndex:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA page_size=4096")
        self.db.execute("PRAGMA cache_size=-8192")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.execute("PRAGMA max_page_count=131072")
        self.db.execute(
            "CREATE TABLE packets(id INTEGER PRIMARY KEY, tuple_key TEXT, ts INTEGER, source TEXT, ip_bytes INTEGER, status TEXT)"
        )
        self.db.execute(
            "CREATE TABLE claims(uid TEXT PRIMARY KEY, packet_id INTEGER, ts INTEGER, source TEXT, ip_bytes INTEGER, missing INTEGER, detail TEXT)"
        )

    def build(self, pcap):
        counts = Counter()
        for packet in packet_records(pcap):
            counts[packet["status"]] += 1
            if "src_ip" not in packet:
                continue
            key = connection_key(
                packet["src_ip"],
                packet["src_port"],
                packet["dst_ip"],
                packet["dst_port"],
                packet["protocol"],
            )
            self.db.execute(
                "INSERT INTO packets VALUES (?, ?, ?, ?, ?, ?)",
                (
                    packet["packet_id"],
                    json.dumps(key, separators=(",", ":")),
                    packet["time_ns"],
                    endpoint(packet["src_ip"], packet["src_port"]),
                    packet["ip_bytes"],
                    packet["status"],
                ),
            )
        self.db.execute("CREATE INDEX packet_lookup ON packets(tuple_key, ts)")
        self.db.commit()
        return dict(counts)

    def establish_points(self, path):
        statuses, witnesses = defaultdict(Counter), defaultdict(Counter)
        for row in label_rows(path):
            missing, detail = row["duration"] == "-", row["detailed-label"]
            counters = [
                row.get(name, "-")
                for name in ("orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes")
            ]
            valid_counts = all(value.isdigit() for value in counters)
            counts = [int(value) for value in counters] if valid_counts else None
            # Known one-packet claims also participate in collision detection.
            if not missing and (counts is None or sum(counts[:2]) != 1):
                continue
            key = connection_key(
                row["id.orig_h"],
                row["id.orig_p"],
                row["id.resp_h"],
                row["id.resp_p"],
                {"tcp": 6, "udp": 17}.get(row["proto"]),
            )
            if key is None:
                if missing:
                    statuses[detail]["unsupported_protocol"] += 1
                continue
            timestamp = nanoseconds(Decimal(row["ts"]) * 1000)
            packets = self.db.execute(
                "SELECT id, ts, source, ip_bytes, status FROM packets WHERE tuple_key=? AND ts=? LIMIT 2",
                (json.dumps(key, separators=(",", ":")), timestamp),
            ).fetchall()
            witness = (
                "no_exact_packet"
                if not packets
                else "multiple_exact_packets"
                if len(packets) > 1
                else packets[0][4]
            )
            if missing:
                witnesses[detail][witness] += 1
            reason = None
            if counts is None:
                reason = "unknown_packet_or_byte_counts"
            elif sum(counts[:2]) == 0:
                reason = "zero_packet_claim"
            elif sum(counts[:2]) != 1:
                reason = "multi_packet_claim"
            elif len(packets) != 1:
                reason = witness
            elif packets[0][4] != "decoded_tcp_udp":
                reason = "unsupported_packet_structure"
            else:
                origin = counts[0] == 1
                source = endpoint(
                    row["id.orig_h"] if origin else row["id.resp_h"],
                    row["id.orig_p"] if origin else row["id.resp_p"],
                )
                if (
                    packets[0][2] != source
                    or counts[2 if origin else 3] != packets[0][3]
                    or counts[3 if origin else 2] != 0
                ):
                    reason = "direction_or_ip_bytes_mismatch"
            if reason:
                if missing:
                    statuses[detail][reason] += 1
                continue
            packet = packets[0]
            self.db.execute(
                "INSERT INTO claims VALUES (?, ?, ?, ?, ?, ?, ?)",
                (row["uid"], *packet[:4], int(missing), detail),
            )
        self.db.execute("CREATE INDEX claimed_packet ON claims(packet_id)")
        self.db.execute(
            "CREATE TABLE points AS SELECT * FROM claims WHERE missing=1 AND packet_id NOT IN (SELECT packet_id FROM claims GROUP BY packet_id HAVING COUNT(*)>1)"
        )
        self.db.execute("CREATE UNIQUE INDEX point_uid ON points(uid)")
        for detail, count in self.db.execute(
            "SELECT detail, COUNT(*) FROM claims WHERE missing=1 AND uid NOT IN (SELECT uid FROM points) GROUP BY detail"
        ):
            statuses[detail]["shared_packet_claim"] += count
        proven = dict(self.db.execute("SELECT detail, COUNT(*) FROM points GROUP BY detail"))
        self.db.commit()
        return {
            "proven_points_by_detail": proven,
            "excluded_missing_records_by_detail": {
                key: dict(value) for key, value in statuses.items()
            },
            "missing_record_packet_witnesses_by_detail": {
                key: dict(value) for key, value in witnesses.items()
            },
        }

    def close(self):
        self.db.close()


class PointLabels(SQLiteLabelIndex):
    """Candidate view of the baseline index; ownership stays with its original context."""

    def __init__(self, baseline, packet_index):
        self.connection = baseline.connection
        self.connection.execute("ATTACH DATABASE ? AS packet_evidence", (str(packet_index.path),))

    def overlapping(self, key, start, end, slack):
        rows = self.connection.execute(
            """SELECT l.uid,l.start_ns,
            CASE WHEN p.uid IS NOT NULL THEN l.start_ns ELSE l.end_ns END AS stop,
            l.label,l.detail,p.source,p.ip_bytes
            FROM labels l LEFT JOIN packet_evidence.points p ON l.uid=p.uid
            WHERE l.tuple_key=? AND l.start_ns<=? AND
            (CASE WHEN p.uid IS NOT NULL THEN l.start_ns ELSE l.end_ns END IS NULL OR
             CASE WHEN p.uid IS NOT NULL THEN l.start_ns ELSE l.end_ns END >=?) LIMIT 2""",
            (
                json.dumps(key, separators=(",", ":")),
                nanoseconds(end + slack),
                nanoseconds(max(Decimal(0), start - slack)),
            ),
        )
        return [
            {
                "uid": uid,
                "start": Decimal(first) / 1_000_000,
                "end": Decimal(last) / 1_000_000 if last is not None else None,
                "label": label,
                "detail": detail,
                "point_source": source,
                "point_ip_bytes": ip_bytes,
            }
            for uid, first, last, label, detail, source, ip_bytes in rows
        ]


def candidate_match(flow, labels):
    status, row = match_flow(flow, labels)
    if status == "matched" and row.get("point_ip_bytes") is not None:
        expected_ms = nanoseconds(row["start"]) // 1_000_000
        if (
            flow["bidirectional_packets"] != 1
            or flow["src2dst_packets"] != 1
            or flow["dst2src_packets"] != 0
            or flow["bidirectional_duration_ms"] != 0
            or flow["bidirectional_first_seen_ms"] != expected_ms
            or flow["bidirectional_last_seen_ms"] != expected_ms
            or flow["src2dst_bytes"] != row["point_ip_bytes"]
            or flow["dst2src_bytes"] != 0
            or endpoint(flow["src_ip"], flow["src_port"]) != row["point_source"]
        ):
            return "point_flow_counter_mismatch", None
    return status, row


class MatchCounts:
    def __init__(self):
        self.statuses, self.classes, self.details = Counter(), Counter(), Counter()
        self.uids, self.detail_uids = defaultdict(set), defaultdict(set)
        self.digest = hashlib.sha256()

    def add(self, key, status, row):
        self.statuses[status] += 1
        self.digest.update(
            (
                json.dumps(
                    [
                        key,
                        status,
                        row["uid"] if row else None,
                        row["label"] if row else None,
                        row["detail"] if row else None,
                    ],
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode()
        )
        if row is not None:
            self.classes[str(row["label"])] += 1
            self.details[row["detail"]] += 1
            self.uids[str(row["label"])].add(row["uid"])
            self.detail_uids[row["detail"]].add(row["uid"])

    def report(self):
        return {
            "flow_status_counts": dict(self.statuses),
            "matched_flow_class_counts": dict(self.classes),
            "matched_detailed_labels": dict(self.details),
            "matched_zeek_connection_class_counts": {
                key: len(value) for key, value in self.uids.items()
            },
            "matched_connections_by_detail": {
                key: len(value) for key, value in self.detail_uids.items()
            },
            "ordered_assignment_sha256": self.digest.hexdigest(),
        }


def audit_packet_labels(
    capture_dir, output, *, compare_flows=False, config_path="configs/data/packet-evidence.json"
):
    if Path(output).exists():
        raise FileExistsError(output)
    if compare_flows and platform.system() != "Linux":
        raise ValueError("Candidate flow comparison uses the verified Linux NFStream exporter.")
    policy = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if (
        policy["policy"] != POLICY
        or policy["training_enabled"] is not False
        or policy["packet_timestamp_tolerance_ns"] != 0
        or policy["max_pcap_bytes"] != 128 * 2**20
        or policy["max_packets"] != 2_000_000
        or policy["max_database_bytes"] != 512 * 2**20
        or policy["cache_kib"] != 8192
        or policy["flow_timestamp_tolerance_ms"] != 1
    ):
        raise ValueError("Use the registered training-disabled exact packet-witness policy.")
    config, capture, receipt = verified_inputs(capture_dir, policy["source_config"])
    path = Path(capture_dir) / f".packet-evidence-{uuid4().hex}.sqlite"
    packet_index = PacketIndex(path)
    try:
        with (
            ResourceMonitor(include_children=True) as resources,
            indexed_labels(capture_dir, config) as (parsed, _),
        ):
            packet_counts = packet_index.build(Path(capture_dir) / capture["pcap"])
            evidence = packet_index.establish_points(Path(capture_dir) / "bro/conn.log.labeled")
            result = {
                "policy": POLICY,
                "capture_id": receipt["capture_id"],
                "source_id": "iot23",
                "policy_sha256": sha256_file(config_path),
                "source_config_sha256": sha256_file(policy["source_config"]),
                "input_files": receipt["files"],
                "packet_status_counts": packet_counts,
                "label_log_counts": parsed[1],
                "point_evidence": evidence,
                "training_authorized": False,
                "candidate_policy_promoted": False,
                "flows_compared": False,
            }
            if compare_flows:
                candidate = PointLabels(parsed[0], packet_index)
                before, after, changes, seen = MatchCounts(), MatchCounts(), Counter(), set()
                for flow in iter_pcap_rows(
                    Path(capture_dir) / capture["pcap"],
                    max_nflows=config["max_audit_flows"],
                    max_pcap_bytes=config.get("max_pcap_bytes", 16 * 2**20),
                ):
                    key = flow_key(flow)
                    if key in seen:
                        raise ValueError("Duplicate flow identity in packet-backed replay.")
                    seen.add(key)
                    old_status, old = match_flow(flow, parsed[0])
                    new_status, new = candidate_match(flow, candidate)
                    before.add(key, old_status, old)
                    after.add(key, new_status, new)
                    if old is not None and (
                        new is None
                        or (old["uid"], old["label"], old["detail"])
                        != (new["uid"], new["label"], new["detail"])
                    ):
                        changes["lost_or_changed_previous_match"] += 1
                    elif old is None and new is not None:
                        changes["newly_matched_flows"] += 1
                    else:
                        changes["unchanged_flows"] += 1
                result.update(
                    {
                        "flows_compared": True,
                        "exported_flows": len(seen),
                        "export_settings": {**SETTINGS, "max_nflows": config["max_audit_flows"]},
                        "baseline": before.report(),
                        "candidate": after.report(),
                        "changes": dict(changes),
                        "previous_matches_preserved": not changes["lost_or_changed_previous_match"],
                    }
                )
                saved = (
                    Path("reports/evidence/capture-expansion/expanded")
                    / f"{receipt['capture_id']}-audit.json"
                )
                baseline = json.loads(saved.read_text(encoding="utf-8"))
                fields = [
                    "flow_status_counts",
                    "matched_flow_class_counts",
                    "matched_detailed_labels",
                    "matched_zeek_connection_class_counts",
                    "matched_connections_by_detail",
                    "ordered_assignment_sha256",
                ]
                result["saved_v1_replay_identical"] = all(
                    result["baseline"][field] == baseline[field] for field in fields
                )
            result["packet_index_file_bytes"] = path.stat().st_size
        result["resources"] = {
            "wall_seconds": resources.elapsed,
            "sampled_peak_process_rss_mib": resources.peak / 2**20,
            "sampled_peak_process_tree_rss_sum_mib": resources.peak_tree / 2**20,
            "scope": "50 ms samples; includes parsing/indexing/matching, excludes downloads; shared child pages may be counted multiple times; no OS-enforced memory bound",
        }
        result["resource_screen_met"] = resources.elapsed < 600 and resources.peak_tree < 2**30
        write_json(output, result)
        if compare_flows and (
            not result["saved_v1_replay_identical"] or not result["previous_matches_preserved"]
        ):
            raise ValueError(
                "Baseline preservation failed; see the written packet evidence report."
            )
        return result
    finally:
        packet_index.close()
        path.unlink(missing_ok=True)

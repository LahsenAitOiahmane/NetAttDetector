"""Run the registered packet-witness experiment sequentially on pinned development data."""

import argparse
import json
import subprocess
from itertools import pairwise
from pathlib import Path

from netattdetector.capture_labels import acquire_capture
from netattdetector.io import write_json
from netattdetector.packet_evidence import audit_packet_labels
from netattdetector.provenance import utc_time


def run(data_root, output):
    data_root, output = Path(data_root), Path(output)
    if data_root.exists() or output.exists():
        raise FileExistsError("Use fresh input and evidence directories.")
    output.mkdir(parents=True)
    policy_path = Path("configs/data/packet-evidence.json")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    source_path = Path(policy["source_config"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    write_json(output / "policy.json", policy)
    write_json(output / "source-config.json", source)
    records = []
    for capture_id in source["captures"]:
        capture_dir = data_root / capture_id
        receipt = acquire_capture(capture_id, capture_dir, source_path)
        write_json(output / f"{capture_id}-receipt.json", receipt)
        result = audit_packet_labels(
            capture_dir, output / f"{capture_id}-packet-evidence.json", compare_flows=True
        )
        records.append(
            {
                "capture_id": capture_id,
                "exported_flows": result["exported_flows"],
                "packet_status_counts": result["packet_status_counts"],
                "baseline": result["baseline"],
                "candidate": result["candidate"],
                "changes": result["changes"],
                "saved_v1_replay_identical": result["saved_v1_replay_identical"],
                "previous_matches_preserved": result["previous_matches_preserved"],
                "resource_screen_met": result["resource_screen_met"],
                "resources": result["resources"],
            }
        )
    summary = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "policy": policy["policy"],
        "records": records,
        "all_acceptance_screens_met": all(
            item["saved_v1_replay_identical"]
            and item["previous_matches_preserved"]
            and item["resource_screen_met"]
            for item in records
        ),
        "training_authorized": False,
        "candidate_policy_promoted": False,
    }
    plan_path = Path("configs/data/native-dataset-plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    by_capture = {item["capture_id"]: item for item in records}
    timeline = []
    support = True
    for split in ("train", "calibration", "validation"):
        capture_id = plan["partitions"][split]
        original = json.loads(
            (
                Path("reports/evidence/capture-expansion/expanded") / f"{capture_id}-audit.json"
            ).read_text()
        )
        timeline.append(
            {
                "split": split,
                "capture_id": capture_id,
                "start_utc": original["event_start_utc"],
                "end_utc": original["event_end_utc"],
            }
        )
        counts = by_capture[capture_id]["candidate"]["matched_zeek_connection_class_counts"]
        support = (
            support
            and min(counts.get("0", 0), counts.get("1", 0))
            >= plan["minimum_distinct_connections_per_class"]
        )
    chronology = all(
        utc_time(left["end_utc"]) + plan["purge_seconds"] <= utc_time(right["start_utc"])
        for left, right in pairwise(timeline)
    )
    summary["native_partition_readiness"] = {
        "timeline": timeline,
        "capture_time_order_and_purge_valid": chronology,
        "connection_support_screen_met": support,
        "device_isolation_verified": False,
        "native_cohort_materialized": False,
        "external_test_acquired": False,
        "training_authorized": False,
        "blockers": [
            "Device identities are not verified across captures.",
            "Point-label rule remains a candidate after this experiment.",
            "A compatible row-level native training cohort with UID groups is not yet materialized.",
            "Dataset mirror/release terms require resolution before a public model/data release.",
        ],
    }
    write_json(output / "native-dataset-plan.json", plan)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if not summary["all_acceptance_screens_met"]:
        raise ValueError("Packet evidence acceptance screen failed; see summary.json.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.data_root, args.output)

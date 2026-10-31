"""Sequential bounded development-only audit. Retain aggregate evidence, not raw traffic."""

import argparse
import json
from pathlib import Path

from netattdetector.capture_labels import CATALOG, acquire_capture, audit_pcap_labels
from netattdetector.io import sha256_file, write_json


def run(output, data_root, record_first=False, config_path=CATALOG):
    output, data_root = Path(output), Path(data_root)
    if output.exists() or data_root.exists():
        raise FileExistsError("Use fresh evidence and data directories.")
    output.mkdir(parents=True)
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for capture_id, capture in config["captures"].items():
        receipt = acquire_capture(
            capture_id, data_root / capture_id, config_path, require_pins=not record_first
        )
        write_json(output / f"{capture_id}-receipt.json", receipt)
        if record_first:
            capture["sha256"] = {name: item["sha256"] for name, item in receipt["files"].items()}
    frozen_config = output / "inputs-frozen.json"
    write_json(frozen_config, config)
    records = []
    for capture_id in config["captures"]:
        capture_dir = data_root / capture_id
        result = audit_pcap_labels(capture_dir, output / f"{capture_id}-audit.json", frozen_config)
        records.append(
            {
                "capture_id": capture_id,
                "flows": result["exported_flows"],
                "statuses": result["flow_status_counts"],
                "matched_zeek_connections": result["matched_zeek_connection_class_counts"],
                "connection_support_screen": result["connection_support_screen"],
                "resource_screen_met": result["resources"]["wall_seconds"] < 600
                and result["resources"]["sampled_peak_process_tree_rss_sum_mib"] < 1024,
                "resources": result["resources"],
            }
        )
    summary = {
        "source_id": "iot23",
        "role": "development",
        "training_authorized": False,
        "first_acquisition": record_first,
        "registered_config_sha256": sha256_file(config_path),
        "frozen_config_sha256": sha256_file(frozen_config),
        "records": records,
        "raw_or_derived_rows_published": False,
        "all_connection_support_screens_met": all(
            item["connection_support_screen"]["both_classes_supported"] for item in records
        ),
        "all_resource_screens_met": all(item["resource_screen_met"] for item in records),
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--record-first", action="store_true")
    parser.add_argument("--config", default=str(CATALOG))
    args = parser.parse_args()
    run(args.output, args.data_root, args.record_first, args.config)

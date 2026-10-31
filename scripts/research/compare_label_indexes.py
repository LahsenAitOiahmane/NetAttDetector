"""Compare index backends in fresh processes on the largest registered label log."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from netattdetector.io import sha256_file, write_json


def compare(config_path, data_root, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    capture_id = max(
        config["captures"],
        key=lambda key: (Path(data_root) / key / "bro/conn.log.labeled").stat().st_size,
    )
    reports = {}
    for backend in ("memory", "sqlite"):
        path = output / f"{backend}.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "netattdetector",
                "audit-pcap-labels",
                "--capture-dir",
                str(Path(data_root) / capture_id),
                "--config",
                str(config_path),
                "--output",
                str(path),
                "--label-index-backend",
                backend,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        reports[backend] = json.loads(path.read_text(encoding="utf-8"))
    fields = [
        "schema",
        "exporter_version",
        "config_sha256",
        "export_settings",
        "input_files",
        "exported_flows",
        "ordered_assignment_sha256",
        "flow_status_counts",
        "matched_flow_class_counts",
        "matched_zeek_connection_class_counts",
        "matched_connections_by_detail",
        "event_start_utc",
        "event_end_utc",
    ]
    equality = {field: reports["memory"][field] == reports["sqlite"][field] for field in fields}
    result = {
        "capture_id": capture_id,
        "config_sha256": sha256_file(config_path),
        "fresh_process_per_backend": True,
        "matching_results_identical": all(equality.values()),
        "checks": equality,
        "resources": {name: report["resources"] for name, report in reports.items()},
        "training_authorized": False,
        "scope": "Offline index trade-off and ordered assignment comparison; no model training or detection metrics. One run per mode, not a repeated performance benchmark.",
    }
    write_json(output / "comparison.json", result)
    print(json.dumps(result, indent=2))
    if not result["matching_results_identical"]:
        raise ValueError("Label-index backend decisions differ; see comparison.json.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    compare(args.config, args.data_root, args.output)
